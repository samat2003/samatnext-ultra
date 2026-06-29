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
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    stateful_ssm_token_reference,
    stateful_ssm_token_triton_,
)


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
    parser = argparse.ArgumentParser(description="Experiment 003D stateful single-token thinking latency")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=10_000)
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--reset-state-each-iter", action="store_true")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--mode", choices=["kernel-only", "end-to-end"], default="kernel-only")
    parser.add_argument("--block-d", type=int, default=256)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--profile", action="store_true")
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


def make_tensors(args: argparse.Namespace, device: torch.device):
    torch.manual_seed(args.seed)
    dtype = dtype_for(args.amp)
    model = DynamicDnaSsmLM(
        DynamicDnaSsmConfig(max_layers=args.layers, chunk_size=args.layers, halt_threshold=1.1)
    ).to(device)
    model.eval()
    token = torch.randint(0, 256, (1, 1), device=device)
    x = model.token_embed(token).reshape(args.d_model).to(dtype).contiguous()
    h = torch.zeros(args.layers, args.d_model, device=device, dtype=dtype).contiguous()
    h_initial = h.clone()
    out = torch.empty_like(x)
    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, args.layers, device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    return model, token, x, h, h_initial, out, a_sig, b, c, g_silu


def run_kernel_once(args: argparse.Namespace, x, h, h_initial, out, a_sig, b, c, g_silu, use_triton: bool):
    if args.reset_state_each_iter:
        h.copy_(h_initial)
    if use_triton:
        return stateful_ssm_token_triton_(
            x,
            h,
            a_sig,
            b,
            c,
            g_silu,
            out,
            block_d=args.block_d,
        )
    ref_out, ref_h = stateful_ssm_token_reference(x, h, a_sig, b, c, g_silu)
    out.copy_(ref_out)
    h.copy_(ref_h)
    return out


def run_end_to_end_once(args: argparse.Namespace, model, token, h, h_initial, out, use_triton: bool):
    if args.reset_state_each_iter:
        h.copy_(h_initial)
    dtype = dtype_for(args.amp)
    x = model.token_embed(token).reshape(args.d_model).to(dtype).contiguous()
    a, b, c, g = model.generate_chunk(0, args.layers, token.device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    if use_triton:
        x = stateful_ssm_token_triton_(x, h, a_sig, b, c, g_silu, out, block_d=args.block_d)
    else:
        ref_out, ref_h = stateful_ssm_token_reference(x, h, a_sig, b, c, g_silu)
        out.copy_(ref_out)
        h.copy_(ref_h)
        x = out
    x = x / torch.sqrt(torch.mean(x.float() * x.float()) + model.config.output_norm_eps)
    return F.linear(x.float().unsqueeze(0), model.token_embed.weight.float())


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
    return latency_stats(timings_us), time.perf_counter() - host_start


def time_cuda_graph(device: torch.device, warmup_iters: int, measure_iters: int, fn: Callable[[], torch.Tensor]):
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("torch.cuda.CUDAGraph is unavailable")
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream):
        for _ in range(max(3, warmup_iters)):
            fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup_iters):
        graph.replay()
    sync(device)

    timings_us: list[float] = []
    host_start = time.perf_counter()
    for _ in range(measure_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000.0)
    return latency_stats(timings_us), time.perf_counter() - host_start


def stats_payload(stats: LatencyStats, layers: int) -> dict[str, float]:
    input_tok_s = 1_000_000.0 / stats.mean_us if stats.mean_us > 0 else 0.0
    layer_tok_s = layers * 1_000_000.0 / stats.mean_us if stats.mean_us > 0 else 0.0
    return {
        "mean_us": stats.mean_us,
        "std_us": stats.std_us,
        "cv": stats.cv,
        "min_us": stats.min_us,
        "p50_us": stats.p50_us,
        "p90_us": stats.p90_us,
        "p95_us": stats.p95_us,
        "p99_us": stats.p99_us,
        "p999_us": stats.p999_us,
        "max_us": stats.max_us,
        "input_tok_s": input_tok_s,
        "thinking_layer_token_updates_s": layer_tok_s,
    }


def compare_reference(args: argparse.Namespace, x, h_initial, a_sig, b, c, g_silu, out) -> dict[str, float]:
    ref_out, ref_h = stateful_ssm_token_reference(x, h_initial, a_sig, b, c, g_silu)
    actual_h = h_initial.clone()
    stateful_ssm_token_triton_(x, actual_h, a_sig, b, c, g_silu, out, block_d=args.block_d)
    sync(x.device)
    x_diff = (out.float() - ref_out.float()).abs()
    h_diff = (actual_h.float() - ref_h.float()).abs()
    return {
        "max_abs_error": float(max(x_diff.max().item(), h_diff.max().item())),
        "mean_abs_error": float((x_diff.mean() + h_diff.mean()).item() / 2.0),
    }


def run_once(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device != "cuda":
        raise SystemExit("Experiment 003D requires CUDA event timing")
    if args.force_triton and not TRITON_AVAILABLE:
        raise SystemExit("Triton forced but unavailable")
    if args.d_model != 256:
        raise SystemExit("Experiment 003D target requires --d-model 256")
    if args.d_model % args.block_d != 0:
        raise SystemExit("--d-model must be divisible by --block-d")
    if args.use_cuda_graph and args.mode != "kernel-only":
        raise SystemExit("--use-cuda-graph is only supported for --mode kernel-only")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    model, token, x, h, h_initial, out, a_sig, b, c, g_silu = make_tensors(args, device)
    use_triton = TRITON_AVAILABLE
    if args.force_triton:
        use_triton = True
    fallback_used = not use_triton

    if args.mode == "kernel-only":
        fn = lambda: run_kernel_once(args, x, h, h_initial, out, a_sig, b, c, g_silu, use_triton)
    else:
        fn = lambda: run_end_to_end_once(args, model, token, h, h_initial, out, use_triton)

    with torch.no_grad():
        if args.use_cuda_graph:
            stats, host_elapsed = time_cuda_graph(device, args.warmup_iters, args.measure_iters, fn)
        else:
            stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)

    payload = {
        "implementation": "stateful_single_token_triton" if use_triton else "stateful_single_token_reference",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_available": TRITON_AVAILABLE,
        "triton_kernel_used": use_triton,
        "fallback_used": fallback_used,
        "layers": args.layers,
        "d_model": args.d_model,
        "vocab_size": 256,
        "amp": args.amp,
        "block_d": args.block_d,
        "mode": args.mode,
        "cuda_graph_used": args.use_cuda_graph,
        "reset_state_each_iter": args.reset_state_each_iter,
        "state_reset_included_in_timed_path": args.reset_state_each_iter,
        "kernel_launches_per_step": 1 if use_triton else 0,
        "python_dispatch_in_timed_path": not args.use_cuda_graph,
        "allocations_in_kernel_only_timed_path": False if args.mode == "kernel-only" and use_triton else None,
        "dna_generation_in_timed_path": args.mode == "end-to-end",
        "ab_c_g_preprocessing_in_timed_path": args.mode == "end-to-end",
        "embedding_logits_projection_in_timed_path": args.mode == "end-to-end",
        "global_memory_state_write": "one write per layer/channel per step",
        "global_memory_state_read": "one read per layer/channel per step",
        "reads_ab_c_g_from_global_memory_each_layer": True,
        "true_fixed_depth_execution": True,
        "parameter_count": model.trainable_parameter_count(),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "host_elapsed_sec": host_elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_graph_error": None,
        "profile_requested": args.profile,
    }
    payload.update(stats_payload(stats, args.layers))
    payload["target_100m_ltu_reached"] = payload["thinking_layer_token_updates_s"] >= 100_000_000
    payload["p99_under_50us"] = payload["p99_us"] < 50.0
    payload["p999_under_100us"] = payload["p999_us"] < 100.0
    if args.compare_reference and use_triton:
        payload.update(compare_reference(args, x, h_initial, a_sig, b, c, g_silu, out))
    return payload


def main() -> None:
    args = parse_args()
    if args.autotune:
        results = []
        for block_d in (16, 32, 64, 128, 256):
            tuned_args = argparse.Namespace(**vars(args))
            tuned_args.block_d = block_d
            results.append(run_once(tuned_args))
        print(json.dumps({"autotune_results": results}, indent=2, sort_keys=True))
    else:
        print(json.dumps(run_once(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
