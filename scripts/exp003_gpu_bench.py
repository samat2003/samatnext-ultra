#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 003 GPU speed benchmark")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    parser.add_argument("--compile", default="off", choices=["off", "reduce-overhead", "max-autotune"])
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "fused-adamw", "sgd"])
    parser.add_argument("--update-every", type=int, default=1, choices=[1, 4, 8])
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--no-grad-only", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@contextmanager
def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        yield
        return

    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def dtype_label(device: torch.device, amp: str) -> str:
    if device.type != "cuda" or amp == "off":
        return "fp32"
    return amp


def make_model(args: argparse.Namespace, device: torch.device) -> DynamicDnaSsmLM:
    config = DynamicDnaSsmConfig(
        max_layers=args.max_layers,
        chunk_size=args.chunk_size,
        halt_threshold=1.1,
    )
    model = DynamicDnaSsmLM(config).to(device)
    return model


def maybe_compile(model: DynamicDnaSsmLM, args: argparse.Namespace) -> tuple[torch.nn.Module, str | None]:
    if args.compile == "off":
        return model, None
    try:
        return torch.compile(model, mode=args.compile), None
    except Exception as exc:  # pragma: no cover - depends on local torch backend
        return model, f"{type(exc).__name__}: {exc}"


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> tuple[torch.optim.Optimizer, str, str | None]:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-2), "sgd", None
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", None

    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True), "fused-adamw", None
        except Exception as exc:
            fallback = f"fused-adamw unavailable, fell back to adamw: {type(exc).__name__}: {exc}"
            return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", fallback

    return torch.optim.AdamW(model.parameters(), lr=1e-3), "adamw", "fused-adamw requires CUDA, fell back to adamw"


def synthetic_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
    return inputs, targets


def loss_for_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))


def timed_loop(
    *,
    device: torch.device,
    warmup_iters: int,
    measure_iters: int,
    step_fn: Callable[[], int],
) -> tuple[float, int]:
    for _ in range(warmup_iters):
        step_fn()
    synchronize(device)
    started = time.perf_counter()
    total_layers = 0
    for _ in range(measure_iters):
        total_layers += step_fn()
    synchronize(device)
    return time.perf_counter() - started, total_layers


def tokens_per_second(tokens: int, iters: int, elapsed: float) -> float:
    return float(tokens * iters / elapsed) if elapsed > 0 else 0.0


def layer_token_updates_per_second(tokens: int, layers: int, elapsed: float) -> float:
    return float(tokens * layers / elapsed) if elapsed > 0 else 0.0


def run_forward_only(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.eval()

    def step() -> int:
        context = torch.no_grad() if args.no_grad_only or args.forward_only else nullcontext()
        with context:
            with autocast_context(device, args.amp):
                output = model(inputs, return_metadata=False)
        return int(output.layers_used)

    reset_peak_memory(device)
    elapsed, total_layers = timed_loop(
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        step_fn=step,
    )
    tokens = args.batch_size * args.seq_len
    layers_per_iter = total_layers / args.measure_iters if args.measure_iters else 0.0
    return {
        "forward_only_elapsed_sec": elapsed,
        "forward_only_tok_s": tokens_per_second(tokens, args.measure_iters, elapsed),
        "forward_only_layer_token_updates_s": layer_token_updates_per_second(tokens, total_layers, elapsed),
        "layers_used": layers_per_iter,
        "peak_cuda_memory_bytes": peak_memory(device),
    }


def run_forward_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.eval()

    def step() -> int:
        with torch.no_grad():
            with autocast_context(device, args.amp):
                output = model(inputs, return_metadata=False)
                loss_for_logits(output.logits, targets)
        return int(output.layers_used)

    reset_peak_memory(device)
    elapsed, total_layers = timed_loop(
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        step_fn=step,
    )
    tokens = args.batch_size * args.seq_len
    return {
        "forward_loss_elapsed_sec": elapsed,
        "forward_loss_tok_s": tokens_per_second(tokens, args.measure_iters, elapsed),
        "forward_loss_layer_token_updates_s": layer_token_updates_per_second(tokens, total_layers, elapsed),
    }


def run_train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.train()
    step_index = 0
    optimizer_updates = 0

    def step() -> int:
        nonlocal step_index, optimizer_updates
        step_index += 1
        with autocast_context(device, args.amp):
            output = model(inputs, return_metadata=False)
            loss = loss_for_logits(output.logits, targets)
        if step_index % args.update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
        return int(output.layers_used)

    reset_peak_memory(device)
    elapsed, total_layers = timed_loop(
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        step_fn=step,
    )
    tokens = args.batch_size * args.seq_len
    return {
        "train_elapsed_sec": elapsed,
        "train_step_tok_s": tokens_per_second(tokens, args.measure_iters, elapsed),
        "train_layer_token_updates_s": layer_token_updates_per_second(tokens, total_layers, elapsed),
        "optimizer_updates": optimizer_updates,
        "forward_loss_calls": args.warmup_iters + args.measure_iters,
        "measured_forward_loss_calls": args.measure_iters,
    }


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0 or args.measure_iters <= 0:
        raise SystemExit("--warmup-iters must be >= 0 and --measure-iters must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device == "cpu" and args.amp != "off":
        raise SystemExit("AMP modes are CUDA-only in this benchmark; use --amp off on CPU")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    inputs, targets = synthetic_batch(args, device)
    model = make_model(args, device)
    model, compile_error = maybe_compile(model, args)
    optimizer, optimizer_used, optimizer_fallback = make_optimizer(model, args, device)

    results: dict[str, object] = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": dtype_label(device, args.amp),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_iter": args.batch_size * args.seq_len,
        "max_layers": args.max_layers,
        "chunk_size": args.chunk_size,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "compile_mode": args.compile,
        "compile_error": compile_error,
        "optimizer_requested": args.optimizer,
        "optimizer_used": optimizer_used,
        "optimizer_fallback": optimizer_fallback,
        "update_every": args.update_every,
        "forward_only_requested": args.forward_only,
        "no_grad_only_requested": args.no_grad_only,
        "parameter_count": model.trainable_parameter_count() if hasattr(model, "trainable_parameter_count") else None,
        "reached_100m_tok_s": False,
    }

    forward_only = run_forward_only(model, inputs, args, device)
    results.update(forward_only)

    if args.forward_only or args.no_grad_only:
        results["forward_loss_tok_s"] = None
        results["train_step_tok_s"] = None
        results["optimizer_updates"] = 0
    else:
        results.update(run_forward_loss(model, inputs, targets, args, device))
        results.update(run_train_steps(model, optimizer, inputs, targets, args, device))

    tok_s_candidates = [
        value for key, value in results.items() if key.endswith("_tok_s") and isinstance(value, float)
    ]
    results["reached_100m_tok_s"] = any(value >= 100_000_000 for value in tok_s_candidates)
    if results.get("peak_cuda_memory_bytes") is None:
        results["peak_cuda_memory_bytes"] = peak_memory(device)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
