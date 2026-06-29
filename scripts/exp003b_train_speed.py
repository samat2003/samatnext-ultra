#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 003B real synthetic training speed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--optimizer", choices=["adamw", "fused-adamw", "sgd", "8bit-adamw"], default="adamw")
    parser.add_argument("--update-every", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--overfit-fixed-batch", action="store_true")
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


@contextmanager
def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        yield
        return
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def static_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    inputs = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device, generator=generator)
    targets = torch.roll(inputs, shifts=-1, dims=1)
    return inputs, targets


def make_optimizer(model: DynamicDnaSsmLM, args: argparse.Namespace, device: torch.device):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-2), "sgd", None, False
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", None, False
    if args.optimizer == "fused-adamw":
        if device.type == "cuda":
            try:
                return torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True), "fused-adamw", None, False
            except Exception as exc:
                reason = f"fused-adamw unavailable, fell back to adamw: {type(exc).__name__}: {exc}"
                return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", reason, True
        return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", "fused-adamw requires CUDA, fell back to adamw", True

    try:
        import bitsandbytes as bnb  # type: ignore

        return bnb.optim.AdamW8bit(model.parameters(), lr=1e-3), "8bit-adamw", None, False
    except Exception as exc:
        reason = f"8bit-adamw unavailable, fell back to adamw: {type(exc).__name__}: {exc}"
        return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", reason, True


def loss_for(model: DynamicDnaSsmLM, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device, amp: str):
    with autocast_context(device, amp):
        output = model(inputs, return_metadata=False)
        loss = F.cross_entropy(output.logits.reshape(-1, 256), targets.reshape(-1))
    return loss, int(output.layers_used)


def time_forward_only(model: DynamicDnaSsmLM, inputs: torch.Tensor, args: argparse.Namespace, device: torch.device):
    model.eval()
    for _ in range(args.warmup_iters):
        with torch.no_grad():
            with autocast_context(device, args.amp):
                output = model(inputs, return_metadata=False)
    synchronize(device)
    started = time.perf_counter()
    layers_used = 0
    for _ in range(args.measure_iters):
        with torch.no_grad():
            with autocast_context(device, args.amp):
                output = model(inputs, return_metadata=False)
        layers_used += int(output.layers_used)
    synchronize(device)
    return time.perf_counter() - started, layers_used / args.measure_iters


def time_forward_loss(model: DynamicDnaSsmLM, inputs: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace, device: torch.device):
    model.eval()
    for _ in range(args.warmup_iters):
        with torch.no_grad():
            loss_for(model, inputs, targets, device, args.amp)
    synchronize(device)
    started = time.perf_counter()
    layers_used = 0
    first_loss = None
    last_loss = None
    for _ in range(args.measure_iters):
        with torch.no_grad():
            loss, used = loss_for(model, inputs, targets, device, args.amp)
        layers_used += used
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
    synchronize(device)
    return time.perf_counter() - started, layers_used / args.measure_iters, first_loss, last_loss


def time_train(model: DynamicDnaSsmLM, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace, device: torch.device):
    model.train()
    measured_steps = args.train_iters if args.overfit_fixed_batch and args.train_iters is not None else args.measure_iters
    warmup_steps = 0 if args.overfit_fixed_batch else args.warmup_iters

    global_step = 0
    for _ in range(warmup_steps):
        global_step += 1
        loss, _ = loss_for(model, inputs, targets, device, args.amp)
        if global_step % args.update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    synchronize(device)
    reset_peak(device)
    started = time.perf_counter()
    optimizer_updates = 0
    layers_used = 0
    first_loss = None
    last_loss = None
    for _ in range(measured_steps):
        global_step += 1
        loss, used = loss_for(model, inputs, targets, device, args.amp)
        layers_used += used
        loss_value = float(loss.detach().cpu())
        first_loss = loss_value if first_loss is None else first_loss
        last_loss = loss_value
        if global_step % args.update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "train_elapsed_sec": elapsed,
        "measured_steps": measured_steps,
        "layers_used": layers_used / measured_steps if measured_steps else 0,
        "optimizer_updates": optimizer_updates,
        "forward_loss_calls": measured_steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decreased": bool(last_loss is not None and first_loss is not None and last_loss < first_loss),
        "peak_cuda_memory_bytes": peak_memory(device),
    }


def rate(tokens: int, steps: int, elapsed: float) -> float:
    return tokens * steps / elapsed if elapsed > 0 else 0.0


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device == "cpu" and args.amp != "off":
        raise SystemExit("AMP modes require CUDA; use --amp off on CPU")
    if args.layers != 1000 or args.chunk_size != 1000:
        raise SystemExit("Experiment 003B training audit requires --layers 1000 --chunk-size 1000")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    inputs, targets = static_batch(args, device)
    model = DynamicDnaSsmLM(
        DynamicDnaSsmConfig(max_layers=args.layers, chunk_size=args.chunk_size, halt_threshold=1.1)
    ).to(device)
    optimizer, optimizer_used, optimizer_fallback, fallback_used = make_optimizer(model, args, device)

    triton_supports_backward = False
    triton_kernel_used = False
    training_path_implementation = "pytorch_autograd_dynamic_dna_ssm"
    if args.force_triton:
        fallback_used = True
        optimizer_fallback = (
            (optimizer_fallback + "; ") if optimizer_fallback else ""
        ) + "force_triton requested, but current Triton SSM kernel is forward-only and has no backward; training used PyTorch autograd path"

    tokens = args.batch_size * args.seq_len
    reset_peak(device)
    forward_only_elapsed, forward_only_layers = time_forward_only(model, inputs, args, device)
    forward_loss_elapsed, forward_loss_layers, _, _ = time_forward_loss(model, inputs, targets, args, device)
    train = time_train(model, optimizer, inputs, targets, args, device)

    result = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_available": TRITON_AVAILABLE,
        "triton_kernel_used": triton_kernel_used,
        "fallback_used": fallback_used,
        "force_triton": args.force_triton,
        "triton_supports_backward": triton_supports_backward,
        "training_path_implementation": training_path_implementation,
        "amp": args.amp,
        "optimizer_requested": args.optimizer,
        "optimizer_used": optimizer_used,
        "optimizer_fallback": optimizer_fallback,
        "update_every": args.update_every,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "layers": args.layers,
        "chunk_size": args.chunk_size,
        "d_model": 256,
        "vocab_size": 256,
        "parameter_count": model.trainable_parameter_count(),
        "true_fixed_1000_layer_execution": train["layers_used"] == 1000,
        "dynamic_halt_used": False,
        "tokens_per_step": tokens,
        "forward_only_input_tok_s": rate(tokens, args.measure_iters, forward_only_elapsed),
        "forward_loss_input_tok_s": rate(tokens, args.measure_iters, forward_loss_elapsed),
        "train_input_tok_s": rate(tokens, train["measured_steps"], train["train_elapsed_sec"]),
        "forward_only_layer_token_updates_s": rate(tokens * args.layers, args.measure_iters, forward_only_elapsed),
        "forward_loss_layer_token_updates_s": rate(tokens * args.layers, args.measure_iters, forward_loss_elapsed),
        "layer_token_updates_s": rate(tokens * args.layers, train["measured_steps"], train["train_elapsed_sec"]),
        "optimizer_updates": train["optimizer_updates"],
        "forward_loss_calls": train["forward_loss_calls"],
        "measured_steps": train["measured_steps"],
        "train_elapsed_sec": train["train_elapsed_sec"],
        "first_loss": train["first_loss"],
        "last_loss": train["last_loss"],
        "loss_decreased": train["loss_decreased"],
        "peak_cuda_memory_bytes": train["peak_cuda_memory_bytes"],
        "reached_100m_train_input_tok_s": rate(tokens, train["measured_steps"], train["train_elapsed_sec"]) >= 100_000_000,
        "reached_100m_layer_token_updates_s": rate(tokens * args.layers, train["measured_steps"], train["train_elapsed_sec"]) >= 100_000_000,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
