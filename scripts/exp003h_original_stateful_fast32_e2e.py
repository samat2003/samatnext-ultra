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
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE, rms_project_triton_, stateful_ssm_token_triton_


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
    parser = argparse.ArgumentParser(description="Experiment 003H original stateful Fast32 fused end-to-end")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=10_000)
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--output-mode", choices=["full_logits", "top1_only"], default="full_logits")
    parser.add_argument("--compare-reference", action="store_true")
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


def make_tensors(args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    dtype = dtype_for(args.amp)
    model = DynamicDnaSsmLM(
        DynamicDnaSsmConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            max_layers=args.layers,
            chunk_size=args.layers,
            halt_threshold=1.1,
        )
    ).to(device)
    model.eval()
    token = torch.randint(0, args.vocab_size, (1,), device=device, dtype=torch.long)
    weight = model.token_embed.weight.detach().to(dtype).contiguous()
    embed_row = torch.empty(1, args.d_model, device=device, dtype=dtype)
    hidden = torch.empty(args.d_model, device=device, dtype=dtype)
    logits = torch.empty(args.vocab_size, device=device, dtype=torch.float32)
    top1 = torch.empty((), device=device, dtype=torch.int64)
    stateful_h = torch.zeros(args.layers, args.d_model, device=device, dtype=dtype).contiguous()
    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, args.layers, device)
    return {
        "model": model,
        "token": token,
        "weight": weight,
        "embed_row": embed_row,
        "hidden": hidden,
        "logits": logits,
        "top1": top1,
        "stateful_h": stateful_h,
        "a_sig": torch.sigmoid(a).to(dtype).contiguous(),
        "b": b.to(dtype).contiguous(),
        "c": c.to(dtype).contiguous(),
        "g_silu": F.silu(g).to(dtype).contiguous(),
    }


def run_e2e(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    torch.index_select(tensors["weight"], 0, tensors["token"], out=tensors["embed_row"])
    x = tensors["embed_row"][0]
    stateful_ssm_token_triton_(
        x,
        tensors["stateful_h"],
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
        tensors["hidden"],
    )
    rms_project_triton_(
        tensors["hidden"],
        tensors["weight"],
        tensors["logits"],
        eps=tensors["model"].config.output_norm_eps,
    )
    if args.output_mode == "top1_only":
        tensors["top1"].copy_(torch.argmax(tensors["logits"]))
        return tensors["top1"]
    return tensors["logits"]


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
    stream = torch.cuda.Stream(device=device)
    sync(device)
    with torch.cuda.stream(stream):
        for _ in range(max(3, warmup_iters)):
            fn()
    sync(device)
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


def stats_payload(stats: LatencyStats) -> dict[str, float]:
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
        "input_tok_s": 1_000_000.0 / stats.mean_us if stats.mean_us > 0 else 0.0,
    }


def compare_reference(tensors: dict) -> dict[str, float]:
    x = tensors["weight"][tensors["token"][0]]
    h = tensors["stateful_h"].clone()
    out = x.clone()
    for layer in range(tensors["a_sig"].shape[0]):
        h_l = tensors["a_sig"][layer].to(out.dtype) * h[layer].to(out.dtype) + tensors["b"][layer].to(out.dtype) * out
        y_l = tensors["c"][layer].to(out.dtype) * h_l
        out = out + tensors["model"].config.residual_scale * tensors["g_silu"][layer].to(out.dtype) * y_l
        h[layer] = h_l
    out_float = out.float()
    out_norm = out_float / torch.sqrt(torch.mean(out_float * out_float) + tensors["model"].config.output_norm_eps)
    expected = F.linear(out_norm.unsqueeze(0), tensors["weight"].float()).squeeze(0)
    actual = tensors["logits"].float()
    diff = (actual - expected).abs()
    return {
        "max_abs_error": float(diff.max().detach().cpu()),
        "mean_abs_error": float(diff.mean().detach().cpu()),
        "top1_matches_reference": bool(torch.argmax(actual).item() == torch.argmax(expected).item()),
    }


def run_once(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device != "cuda":
        raise SystemExit("Experiment 003H requires CUDA timing")
    if args.force_triton and not TRITON_AVAILABLE:
        raise SystemExit("Triton forced but unavailable")
    if args.layers != 32 or args.d_model != 256 or args.vocab_size != 256:
        raise SystemExit("Experiment 003H requires layers=32, d_model=256, vocab_size=256")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    tensors = make_tensors(args, device)
    fn = lambda: run_e2e(args, tensors)
    cuda_graph_used = False
    graph_capture_failure_reason = None
    with torch.no_grad():
        if args.use_cuda_graph:
            try:
                stats, host_elapsed = time_cuda_graph(device, args.warmup_iters, args.measure_iters, fn)
                cuda_graph_used = True
            except Exception as exc:
                graph_capture_failure_reason = str(exc)
                stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)
        else:
            stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)

    payload = {
        "experiment": "EXP003H",
        "variant": "original_stateful_32_fused_e2e",
        "layers": args.layers,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "amp": args.amp,
        "output_mode": args.output_mode,
        "cuda_graph_requested": args.use_cuda_graph,
        "cuda_graph_used": cuda_graph_used,
        "graph_capture_failure_reason": graph_capture_failure_reason,
        "parameter_count": tensors["model"].trainable_parameter_count(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "triton_kernel_used": True,
        "fallback_used": False,
        "full_logits_produced": args.output_mode == "full_logits",
        "kernel_launches_per_step": 3 if args.output_mode == "full_logits" else 4,
        "host_elapsed_sec": host_elapsed,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "target_mean_10_15us_met": False,
        "target_p99_under_50us_met": False,
        "target_p999_under_100us_met": False,
    }
    payload.update(stats_payload(stats))
    payload["target_mean_10_15us_met"] = 10.0 <= payload["mean_us"] <= 15.0
    payload["target_p99_under_50us_met"] = payload["p99_us"] < 50.0
    payload["target_p999_under_100us_met"] = payload["p999_us"] < 100.0
    if args.compare_reference:
        payload.update(compare_reference(tensors))
    return payload


def main() -> None:
    args = parse_args()
    print(json.dumps(run_once(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
