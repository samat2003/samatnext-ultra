#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""EXP004B: Fast32 UE Training Speed Ladder with Cached Triton Forward Path.

Goal:
Reach or approach 1M+ train tokens/sec using the cached Triton forward path
on non-update steps, while keeping update steps as true autograd training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Generator

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM

FROZEN_DIR = ROOT / "checkpoints" / "fast32_frozen"
FROZEN_CONFIG = FROZEN_DIR / "config.json"
FROZEN_CHECKSUMS = FROZEN_DIR / "SHA256SUMS.txt"
REQUIRED_PARAM_COUNT = 216_320


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


@contextmanager
def autocast_ctx(device: torch.device, amp: str) -> Generator:
    if device.type != "cuda" or amp == "off":
        yield
        return
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_frozen_config() -> DynamicDnaSsmConfig:
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    kwargs = {f.name: payload[f.name] for f in fields(DynamicDnaSsmConfig)}
    return DynamicDnaSsmConfig(**kwargs)


def make_model(device: torch.device) -> DynamicDnaSsmLM:
    config = load_frozen_config()
    model = DynamicDnaSsmLM(config).to(device)
    n = model.trainable_parameter_count()
    if n != REQUIRED_PARAM_COUNT:
        raise RuntimeError(
            f"parameter count changed: {n} != {REQUIRED_PARAM_COUNT}. "
            "Architecture must remain frozen."
        )
    return model


def frozen_dir_unchanged() -> bool:
    if not FROZEN_CHECKSUMS.exists():
        return False
    lines = FROZEN_CHECKSUMS.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        sha, name = line.split("  ", 1)
        path = FROZEN_DIR / name.strip()
        if not path.exists():
            return False
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha:
            return False
    return True


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(
    model: DynamicDnaSsmLM,
    args: argparse.Namespace,
    device: torch.device,
):
    lr = 1e-3
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-2), "sgd", None
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr), "adamw", None
    if args.optimizer == "fused-adamw":
        if device.type == "cuda":
            try:
                opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
                return opt, "fused-adamw", None
            except Exception as exc:
                note = f"fused-adamw unavailable, fell back to adamw: {exc}"
                return torch.optim.AdamW(model.parameters(), lr=lr), "adamw", note
        note = "fused-adamw requires CUDA, fell back to adamw"
        return torch.optim.AdamW(model.parameters(), lr=lr), "adamw", note
    note = f"unknown optimizer {args.optimizer!r}, used adamw"
    return torch.optim.AdamW(model.parameters(), lr=lr), "adamw", note


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def static_batch(args: argparse.Namespace, device: torch.device):
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    x = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device, generator=g)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def bytes_encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def load_tiny_shakespeare(dataset_name: str) -> torch.Tensor:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    candidates = list(dict.fromkeys([dataset_name, "karpathy/tiny_shakespeare", "tiny_shakespeare"]))
    for name in candidates:
        try:
            ds = load_dataset(name, trust_remote_code=True)
            pieces: list[str] = []
            for split in ds.values():
                cols = getattr(split, "column_names", [])
                col = "text" if "text" in cols else cols[0]
                for row in split:
                    v = row[col]
                    pieces.append(v if isinstance(v, str) else str(v))
            return bytes_encode("\n".join(pieces))
        except Exception:
            continue
    raise RuntimeError(f"Could not load tiny_shakespeare; tried {candidates}")


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
):
    if data.numel() <= seq_len + 1:
        raise ValueError("data too small for sampling")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return x.long(), y.long()


def mode_to_update_every(mode: str) -> int:
    mapping = {
        "standard": 1,
        "ue1": 1,
        "ue4": 4,
        "ue8": 8,
        "ue16": 16,
        "ue32": 32,
        "ue64": 64,
        "ue128": 128,
    }
    if mode not in mapping:
        raise ValueError(f"unknown mode {mode!r}")
    return mapping[mode]


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class ComponentProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.data_fetch_sec = 0.0
        self.data_fetch_count = 0
        
        self.no_grad_fwd_sec = 0.0
        self.no_grad_fwd_count = 0
        
        self.grad_fwd_sec = 0.0
        self.grad_fwd_count = 0
        
        self.backward_sec = 0.0
        self.backward_count = 0
        
        self.optimizer_sec = 0.0
        self.optimizer_count = 0
        
        self.cache_refresh_sec = 0.0
        self.cache_refresh_count = 0

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def record_data_fetch(self, duration: float):
        self.data_fetch_sec += duration
        self.data_fetch_count += 1

    def record_no_grad_fwd(self, duration: float):
        self.no_grad_fwd_sec += duration
        self.no_grad_fwd_count += 1

    def record_grad_fwd(self, duration: float):
        self.grad_fwd_sec += duration
        self.grad_fwd_count += 1

    def record_backward(self, duration: float):
        self.backward_sec += duration
        self.backward_count += 1

    def record_optimizer(self, duration: float):
        self.optimizer_sec += duration
        self.optimizer_count += 1

    def record_cache_refresh(self, duration: float):
        self.cache_refresh_sec += duration
        self.cache_refresh_count += 1

    def averages(self) -> dict[str, float]:
        return {
            "avg_data_fetch_sec": self.data_fetch_sec / max(self.data_fetch_count, 1),
            "avg_no_grad_fwd_sec": self.no_grad_fwd_sec / max(self.no_grad_fwd_count, 1),
            "avg_grad_fwd_sec": self.grad_fwd_sec / max(self.grad_fwd_count, 1),
            "avg_backward_sec": self.backward_sec / max(self.backward_count, 1),
            "avg_optimizer_sec": self.optimizer_sec / max(self.optimizer_count, 1),
            "avg_cache_refresh_sec": self.cache_refresh_sec / max(self.cache_refresh_count, 1),
        }


# ---------------------------------------------------------------------------
# Cached Triton forward/loss path (V0 no-grad)
# ---------------------------------------------------------------------------

def cached_triton_forward_loss(
    model: DynamicDnaSsmLM,
    cached_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    amp: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    device = input_ids.device
    cfg = model.config

    if cached_params is None:
        a, b, c, g = model.generate_chunk(0, cfg.max_layers, device)
        a_sig = torch.sigmoid(a).contiguous()
        b_c = b.contiguous()
        c_c = c.contiguous()
        g_silu = F.silu(g).contiguous()
        cached_params = (a_sig, b_c, c_c, g_silu)

    a_sig, b_c, c_c, g_silu = cached_params

    with autocast_ctx(device, amp):
        x = model.token_embed(input_ids)
        from samatnext_dna_ssm.triton_ssm import fixed_ssm_triton
        x = fixed_ssm_triton(
            x,
            a_sig,
            b_c,
            c_c,
            g_silu,
            residual_scale=cfg.residual_scale,
        )
        x = model._output_normalize(x, cfg.output_norm_eps)
        logits = F.linear(x, model.token_embed.weight)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    return loss, cached_params


def verify_cached_triton_correctness(
    model: DynamicDnaSsmLM,
    x: torch.Tensor,
    y: torch.Tensor,
    amp: str,
):
    """Verify that cached_triton_loss produces logits and loss matching py_loop."""
    device = x.device
    model.eval()
    
    with torch.no_grad():
        # Standard py_loop forward pass
        with autocast_ctx(device, amp):
            out_py = model(x, return_metadata=False)
            loss_py = F.cross_entropy(out_py.logits.reshape(-1, out_py.logits.shape[-1]), y.reshape(-1))

        # Triton forward pass
        loss_tri, _ = cached_triton_forward_loss(model, None, x, y, amp)

    model.train()
    
    diff_loss = (loss_py - loss_tri).abs().item()
    print(f"[Correctness Check] py_loop loss: {loss_py.item():.6f}, cached_triton loss: {loss_tri.item():.6f}, diff: {diff_loss:.6e}")
    
    if diff_loss > 1e-3:
        raise RuntimeError(
            f"Correctness validation failed: loss difference ({diff_loss:.2e}) "
            "exceeds tolerance."
        )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def run_ue_train_loop(
    model: DynamicDnaSsmLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    get_batch,
    device: torch.device,
    args: argparse.Namespace,
    update_every: int,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[dict[str, Any], ComponentProfiler]:
    """Execute the main training loop with profiling."""
    model.train()
    global_step = 0
    
    cached_params = None
    amp = args.amp

    # --- WARMUP PASS (Not timed or profiled) ---
    for _ in range(warmup_steps):
        x, y = get_batch()
        global_step += 1
        is_update = (global_step % update_every == 0)
        
        if is_update:
            if args.update_impl == "fused_backward":
                raise NotImplementedError("fused_backward update-impl is a placeholder and not implemented.")
            
            with autocast_ctx(device, amp):
                out = model(x, return_metadata=False)
                loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            # Regenerate the DNA cache after optimizer update
            with torch.no_grad():
                a, b, c, g = model.generate_chunk(0, model.config.max_layers, device)
                cached_a_sig = torch.sigmoid(a).contiguous()
                cached_b = b.contiguous()
                cached_c = c.contiguous()
                cached_g_silu = F.silu(g).contiguous()
                cached_params = (cached_a_sig, cached_b, cached_c, cached_g_silu)
        else:
            with torch.no_grad():
                if args.forward_impl == "cached_triton_loss":
                    loss, cached_params = cached_triton_forward_loss(
                        model, cached_params, x, y, amp
                    )
                else:
                    with autocast_ctx(device, amp):
                        out = model(x, return_metadata=False)
                        _ = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))

    # Sync and reset before timed loop
    synchronize(device)
    reset_peak(device)
    
    profiler = ComponentProfiler(device)
    
    first_loss: float | None = None
    final_loss: float | None = None
    measured_optimizer_updates = 0
    measured_forward_loss_calls = 0

    t_start_loop = time.perf_counter()

    for _ in range(measured_steps):
        # 1. Data Fetch
        t_comp = time.perf_counter()
        x, y = get_batch()
        profiler.sync()
        profiler.record_data_fetch(time.perf_counter() - t_comp)
        
        global_step += 1
        is_update = (global_step % update_every == 0)
        measured_forward_loss_calls += 1

        if is_update:
            if args.update_impl == "fused_backward":
                raise NotImplementedError("fused_backward update-impl is a placeholder and not implemented.")
            
            # Grad Forward/Loss
            t_comp = time.perf_counter()
            with autocast_ctx(device, amp):
                out = model(x, return_metadata=False)
                loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            profiler.sync()
            profiler.record_grad_fwd(time.perf_counter() - t_comp)
            
            # Backward Pass
            t_comp = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            profiler.sync()
            profiler.record_backward(time.perf_counter() - t_comp)
            
            # Optimizer step
            t_comp = time.perf_counter()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            profiler.sync()
            profiler.record_optimizer(time.perf_counter() - t_comp)
            
            # Cache refresh (always done immediately after optimizer.step())
            t_comp = time.perf_counter()
            with torch.no_grad():
                a, b, c, g = model.generate_chunk(0, model.config.max_layers, device)
                cached_a_sig = torch.sigmoid(a).contiguous()
                cached_b = b.contiguous()
                cached_c = c.contiguous()
                cached_g_silu = F.silu(g).contiguous()
                cached_params = (cached_a_sig, cached_b, cached_c, cached_g_silu)
            profiler.sync()
            profiler.record_cache_refresh(time.perf_counter() - t_comp)
            
            measured_optimizer_updates += 1
        else:
            # Non-update step: no_grad Forward/Loss
            t_comp = time.perf_counter()
            with torch.no_grad():
                if args.forward_impl == "cached_triton_loss":
                    loss, cached_params = cached_triton_forward_loss(
                        model, cached_params, x, y, amp
                    )
                else:
                    with autocast_ctx(device, amp):
                        out = model(x, return_metadata=False)
                        loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            profiler.sync()
            profiler.record_no_grad_fwd(time.perf_counter() - t_comp)
            
        lv = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = lv
        final_loss = lv

    synchronize(device)
    total_elapsed = time.perf_counter() - t_start_loop
    peak_mem = peak_memory(device)

    # Compile results
    metrics = {
        "train_elapsed_sec": total_elapsed,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "measured_forward_loss_calls": measured_forward_loss_calls,
        "measured_optimizer_updates": measured_optimizer_updates,
        "peak_cuda_memory_bytes": peak_mem,
    }
    return metrics, profiler


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_once(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    
    # 1. Make model
    model = make_model(device)
    
    # 2. Get data
    if args.data == "synthetic":
        x, y = static_batch(args, device)
        def get_batch():
            return x, y
    else:
        tokens = load_tiny_shakespeare(args.dataset_name)
        split = int(tokens.numel() * 0.9)
        train_data = tokens[:split].contiguous()
        val_data = tokens[split:].contiguous()
        
        g = torch.Generator().manual_seed(args.seed)
        if args.overfit_one_batch:
            fixed_x, fixed_y = sample_batch(train_data, args.batch_size, args.seq_len, device, g)
            def get_batch():
                return fixed_x, fixed_y
        else:
            def get_batch():
                return sample_batch(train_data, args.batch_size, args.seq_len, device, g)

    # 3. Correctness check at start (if using cuda)
    if device.type == "cuda" and args.forward_impl == "cached_triton_loss":
        check_x, check_y = get_batch()
        verify_cached_triton_correctness(model, check_x, check_y, args.amp)

    # 4. Prepare optimizer and scaler
    optimizer, opt_label, opt_fallback = make_optimizer(model, args, device)
    use_scaler = (device.type == "cuda" and args.amp == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    
    update_every = mode_to_update_every(args.mode)
    
    # 5. Run loop
    metrics, profiler = run_ue_train_loop(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        get_batch=get_batch,
        device=device,
        args=args,
        update_every=update_every,
        warmup_steps=args.warmup_steps,
        measured_steps=args.steps,
    )
    
    # Calculate measured stats
    tokens_per_step = args.batch_size * args.seq_len
    train_input_tok_s = (tokens_per_step * metrics["measured_forward_loss_calls"]) / metrics["train_elapsed_sec"]
    
    # 6. Upper-bound Estimations
    avgs = profiler.averages()
    avg_data = avgs["avg_data_fetch_sec"]
    avg_no_grad = avgs["avg_no_grad_fwd_sec"]
    avg_grad = avgs["avg_grad_fwd_sec"]
    avg_back = avgs["avg_backward_sec"]
    avg_opt = avgs["avg_optimizer_sec"]
    avg_refresh = avgs["avg_cache_refresh_sec"]
    
    non_update_time = avg_data + avg_no_grad
    update_time = avg_data + avg_grad + avg_back + avg_opt + avg_refresh
    
    estimations = {}
    for m in [1, 4, 8, 16, 32, 64, 128]:
        cycle_time = non_update_time * (m - 1) + update_time
        est_tok_s = (tokens_per_step * m) / cycle_time if cycle_time > 0 else 0.0
        estimations[f"estimated_tok_s_ue{m}"] = est_tok_s

    # Val loss eval (if applicable)
    best_val_loss = None
    if args.data == "tiny_shakespeare" and not args.overfit_one_batch:
        model.eval()
        val_losses = []
        g_val = torch.Generator().manual_seed(args.seed + 999)
        with torch.no_grad():
            for _ in range(8):
                vx, vy = sample_batch(val_data, args.batch_size, args.seq_len, device, g_val)
                if args.forward_impl == "cached_triton_loss":
                    loss_v, _ = cached_triton_forward_loss(model, None, vx, vy, args.amp)
                else:
                    with autocast_ctx(device, args.amp):
                        out_v = model(vx, return_metadata=False)
                        loss_v = F.cross_entropy(out_v.logits.reshape(-1, 256), vy.reshape(-1))
                val_losses.append(loss_v.item())
        best_val_loss = sum(val_losses) / len(val_losses)
        model.train()

    # Required output structure
    output = {
        "experiment": "EXP004B",
        "mode": args.mode,
        "update_every": update_every,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "amp": args.amp,
        "optimizer": opt_label,
        "optimizer_requested": args.optimizer,
        "optimizer_fallback": opt_fallback,
        "forward_impl": args.forward_impl,
        "update_impl": args.update_impl,
        "parameter_count": model.trainable_parameter_count(),
        
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "measured_forward_loss_calls": metrics["measured_forward_loss_calls"],
        "measured_optimizer_updates": metrics["measured_optimizer_updates"],
        "total_forward_loss_calls": metrics["measured_forward_loss_calls"] + args.warmup_steps,
        "total_optimizer_updates": metrics["measured_optimizer_updates"] + (args.warmup_steps // update_every),
        
        # Primary speed metrics
        "train_elapsed_sec": metrics["train_elapsed_sec"],
        "train_input_tok_s": train_input_tok_s,
        "forward_loss_tok_s": train_input_tok_s, # same under scheduled style
        "update_step_tok_s": (tokens_per_step * metrics["measured_optimizer_updates"]) / metrics["train_elapsed_sec"],
        
        # Loss checks
        "first_loss": metrics["first_loss"],
        "final_loss": metrics["final_loss"],
        "loss_decreased": bool(metrics["first_loss"] is not None and metrics["final_loss"] is not None and metrics["final_loss"] < metrics["first_loss"]),
        "best_val_loss": best_val_loss,
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        
        # Integrity checks
        "architecture_changed": False,
        "frozen_artifacts_modified": not frozen_dir_unchanged(),
        
        # Profiler breakdown
        "profile_data_fetch_avg_sec": avg_data,
        "profile_no_grad_fwd_avg_sec": avg_no_grad,
        "profile_grad_fwd_avg_sec": avg_grad,
        "profile_backward_avg_sec": avg_back,
        "profile_optimizer_avg_sec": avg_opt,
        "profile_cache_refresh_avg_sec": avg_refresh,
        
        # Upper bounds
        **estimations,
    }
    
    if args.profile_components:
        print("\n=== Component Timing Breakdown (seconds) ===")
        print(f"Data Fetch:     {avg_data:.6f}")
        print(f"No-Grad Fwd:    {avg_no_grad:.6f}")
        print(f"Grad Fwd:       {avg_grad:.6f}")
        print(f"Backward Pass:  {avg_back:.6f}")
        print(f"Optimizer Step: {avg_opt:.6f}")
        print(f"Cache Refresh:  {avg_refresh:.6f}")
        print("=== Upper Bound Estimations (tok/s) ===")
        for m in [1, 4, 8, 16, 32, 64, 128]:
            print(f"UE{m:3d}:         {estimations[f'estimated_tok_s_ue{m}']:.0f} tok/s")
        print("============================================\n")

    return output


# ---------------------------------------------------------------------------
# CLI Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP004B Fast32 UE Training Speed Ladder")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--data",
        default="synthetic",
        choices=["synthetic", "tiny_shakespeare"],
    )
    p.add_argument("--dataset-name", default="karpathy/tiny_shakespeare")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="fp16")
    p.add_argument(
        "--mode",
        default="ue4",
        choices=["standard", "ue1", "ue4", "ue8", "ue16", "ue32", "ue64", "ue128"],
    )
    p.add_argument(
        "--optimizer",
        default="fused-adamw",
        choices=["adamw", "fused-adamw", "sgd"],
    )
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument(
        "--forward-impl",
        default="py_loop",
        choices=["py_loop", "cached_triton_loss"],
    )
    p.add_argument(
        "--update-impl",
        default="py_autograd",
        choices=["py_autograd", "fused_backward"],
    )
    p.add_argument("--profile-components", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu" and args.amp != "off":
        raise SystemExit("AMP requires CUDA; use --amp off on CPU")
    result = run_once(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
