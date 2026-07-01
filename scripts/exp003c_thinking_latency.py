#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE, fixed_ssm_reference, fixed_ssm_triton


@dataclass(frozen=True)
class LatencyStats:
    mean_us: float
    std_us: float
    min_us: float
    p50_us: float
    p90_us: float
    p95_us: float
    p99_us: float
    p999_us: float
    max_us: float
    cv: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 003C thinking latency benchmark")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=10_000)
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--block-d", type=int, default=32)
    parser.add_argument("--block-seq", type=int, default=None)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def latency_stats(values_us: list[float]) -> LatencyStats:
    ordered = sorted(values_us)
    mean = statistics.fmean(ordered) if ordered else 0.0
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return LatencyStats(
        mean_us=mean,
        std_us=std,
        min_us=ordered[0] if ordered else 0.0,
        p50_us=percentile(ordered, 0.50),
        p90_us=percentile(ordered, 0.90),
        p95_us=percentile(ordered, 0.95),
        p99_us=percentile(ordered, 0.99),
        p999_us=percentile(ordered, 0.999),
        max_us=ordered[-1] if ordered else 0.0,
        cv=(std / mean) if mean > 0 else 0.0,
    )


def dtype_for(amp: str) -> torch.dtype:
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return torch.float32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def make_model_and_tensors(args: argparse.Namespace, device: torch.device):
    torch.manual_seed(args.seed)
    dtype = dtype_for(args.amp)
    model = DynamicDnaSsmLM(
        DynamicDnaSsmConfig(max_layers=args.layers, chunk_size=args.layers, halt_threshold=1.1)
    ).to(device)
    model.eval()
    tokens = torch.randint(0, 256, (args.batch_size, args.seq_len), device=device)
    x = model.token_embed(tokens).to(dtype)
    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, args.layers, device)
    a_sig = torch.sigmoid(a).to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    g_silu = F.silu(g).to(dtype)
    return model, tokens, x, a_sig, b, c, g_silu


def run_once_kernel(args: argparse.Namespace, x, a_sig, b, c, g_silu, use_triton: bool):
    if use_triton:
        return fixed_ssm_triton(
            x,
            a_sig,
            b,
            c,
            g_silu,
            block_seq=args.block_seq or args.seq_len,
            block_d=args.block_d,
        )
    return fixed_ssm_reference(x, a_sig, b, c, g_silu)


def run_once_end_to_end(args: argparse.Namespace, model, tokens, use_triton: bool):
    dtype = dtype_for(args.amp)
    x = model.token_embed(tokens).to(dtype)
    a, b, c, g = model.generate_chunk(0, args.layers, tokens.device)
    a_sig = torch.sigmoid(a).to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    g_silu = F.silu(g).to(dtype)
    if use_triton:
        x = fixed_ssm_triton(
            x,
            a_sig,
            b,
            c,
            g_silu,
            block_seq=args.block_seq or args.seq_len,
            block_d=args.block_d,
        )
    else:
        x = fixed_ssm_reference(x, a_sig, b, c, g_silu)
    x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + model.config.output_norm_eps)
    return F.linear(x.float(), model.token_embed.weight.float())


def time_cuda_events(device: torch.device, warmup_iters: int, measure_iters: int, fn: Callable[[], torch.Tensor]):
    for _ in range(warmup_iters):
        fn()
    sync(device)
    timings_us: list[float] = []
    host_start = time.perf_counter()
    for _ in range(measure_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000.0)
    host_elapsed = time.perf_counter() - host_start
    return latency_stats(timings_us), host_elapsed


def stats_payload(prefix: str, stats: LatencyStats, tokens: int, layers: int):
    input_tok_s = tokens * 1_000_000.0 / stats.mean_us if stats.mean_us > 0 else 0.0
    layer_tok_s = tokens * layers * 1_000_000.0 / stats.mean_us if stats.mean_us > 0 else 0.0
    return {
        f"{prefix}_mean_us": stats.mean_us,
        f"{prefix}_std_us": stats.std_us,
        f"{prefix}_min_us": stats.min_us,
        f"{prefix}_p50_us": stats.p50_us,
        f"{prefix}_p90_us": stats.p90_us,
        f"{prefix}_p95_us": stats.p95_us,
        f"{prefix}_p99_us": stats.p99_us,
        f"{prefix}_p999_us": stats.p999_us,
        f"{prefix}_max_us": stats.max_us,
        f"{prefix}_cv": stats.cv,
        f"{prefix}_input_tok_s": input_tok_s,
        f"{prefix}_thinking_layer_token_updates_s": layer_tok_s,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device != "cuda":
        raise SystemExit("Experiment 003C requires CUDA event timing")
    if args.force_triton and not TRITON_AVAILABLE:
        raise SystemExit("Triton forced but unavailable")
    if args.force_triton and args.force_fallback:
        raise SystemExit("--force-triton and --force-fallback are mutually exclusive")
    if args.d_model != 256:
        raise SystemExit("Experiment 003C target requires --d-model 256")
    if args.block_seq is None:
        args.block_seq = args.seq_len
    if args.seq_len > args.block_seq:
        raise SystemExit("--seq-len must be <= --block-seq")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    model, tokens, x, a_sig, b, c, g_silu = make_model_and_tensors(args, device)
    use_triton = TRITON_AVAILABLE and not args.force_fallback
    if args.force_triton:
        use_triton = True
    fallback_used = not use_triton
    tokens_per_iter = args.batch_size * args.seq_len

    with torch.no_grad():
        kernel_stats, kernel_host_elapsed = time_cuda_events(
            device,
            args.warmup_iters,
            args.measure_iters,
            lambda: run_once_kernel(args, x, a_sig, b, c, g_silu, use_triton),
        )
        if args.skip_end_to_end:
            end_stats = LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            end_host_elapsed = 0.0
        else:
            end_stats, end_host_elapsed = time_cuda_events(
                device,
                args.warmup_iters,
                args.measure_iters,
                lambda: run_once_end_to_end(args, model, tokens, use_triton),
            )

    payload = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_available": TRITON_AVAILABLE,
        "triton_kernel_used": use_triton,
        "fallback_used": fallback_used,
        "true_fixed_depth_execution": True,
        "layers": args.layers,
        "d_model": args.d_model,
        "vocab_size": 256,
        "parameter_count": model.trainable_parameter_count(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "amp": args.amp,
        "block_seq": args.block_seq,
        "block_d": args.block_d,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "tokens_per_iter": tokens_per_iter,
        "kernel_only_host_elapsed_sec": kernel_host_elapsed,
        "end_to_end_host_elapsed_sec": end_host_elapsed,
        "peak_cuda_memory_bytes": peak_memory(device),
    }
    payload.update(stats_payload("kernel_only", kernel_stats, tokens_per_iter, args.layers))
    payload.update(stats_payload("end_to_end", end_stats, tokens_per_iter, args.layers))
    payload["kernel_only_reached_100m_thinking_layer_token_updates_s"] = (
        payload["kernel_only_thinking_layer_token_updates_s"] >= 100_000_000
    )
    payload["end_to_end_reached_100m_thinking_layer_token_updates_s"] = (
        payload["end_to_end_thinking_layer_token_updates_s"] >= 100_000_000
    )
    payload["kernel_only_p99_under_50us"] = payload["kernel_only_p99_us"] < 50.0
    payload["kernel_only_p999_under_100us"] = payload["kernel_only_p999_us"] < 100.0
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
