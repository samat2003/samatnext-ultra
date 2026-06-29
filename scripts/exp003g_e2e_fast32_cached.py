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
    fused_precomposed_logits_triton_,
    fused_precomposed_top1_triton_,
    precompose_stateless_master_coeff,
    precomposed_stateless_triton_,
    rms_project_triton_,
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
    parser = argparse.ArgumentParser(description="Experiment 003G cached Fast32 end-to-end inference")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "original_stateful_32_e2e",
            "original_stateful_32_cached_params",
            "precomposed_stateless_32_e2e",
            "precomposed_stateless_32_cached_master",
            "precomposed_stateless_32_fused_e2e",
        ],
    )
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=10_000)
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--audit-allocations", action="store_true")
    parser.add_argument("--projection", choices=["pytorch", "triton", "fused"], default="pytorch")
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
    top1_score = torch.empty((), device=device, dtype=torch.float32)
    stateful_h = torch.zeros(args.layers, args.d_model, device=device, dtype=dtype).contiguous()
    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, args.layers, device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).to(dtype).contiguous()
    master_coeff = precompose_stateless_master_coeff(coeff).contiguous()
    return {
        "model": model,
        "token": token,
        "weight": weight,
        "embed_row": embed_row,
        "hidden": hidden,
        "logits": logits,
        "top1": top1,
        "top1_score": top1_score,
        "stateful_h": stateful_h,
        "a_sig": a_sig,
        "b": b,
        "c": c,
        "g_silu": g_silu,
        "coeff": coeff,
        "master_coeff": master_coeff,
    }


def embedding_lookup(tensors: dict) -> torch.Tensor:
    torch.index_select(tensors["weight"], 0, tensors["token"], out=tensors["embed_row"])
    return tensors["embed_row"][0]


def pytorch_project(x: torch.Tensor, tensors: dict, eps: float) -> torch.Tensor:
    x_float = x.float()
    x_norm = x_float / torch.sqrt(torch.mean(x_float * x_float) + eps)
    logits = F.linear(x_norm.unsqueeze(0), tensors["weight"].float()).squeeze(0)
    if tensors.get("output_mode") == "top1_only":
        return torch.argmax(logits)
    return logits


def triton_project(x: torch.Tensor, tensors: dict, eps: float, output_mode: str) -> torch.Tensor:
    rms_project_triton_(x, tensors["weight"], tensors["logits"], eps=eps)
    if output_mode == "top1_only":
        tensors["top1"].copy_(torch.argmax(tensors["logits"]))
        return tensors["top1"]
    return tensors["logits"]


def run_uncached_precomposed(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    model = tensors["model"]
    dtype = dtype_for(args.amp)
    x = model.token_embed(tensors["token"].reshape(1, 1)).reshape(args.d_model).to(dtype).contiguous()
    a, b, c, g = model.generate_chunk(0, args.layers, tensors["token"].device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).to(dtype).contiguous()
    master = precompose_stateless_master_coeff(coeff).contiguous()
    precomposed_stateless_triton_(x, master, tensors["hidden"])
    return pytorch_project(tensors["hidden"], tensors, model.config.output_norm_eps)


def run_uncached_original(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    model = tensors["model"]
    dtype = dtype_for(args.amp)
    x = model.token_embed(tensors["token"].reshape(1, 1)).reshape(args.d_model).to(dtype).contiguous()
    a, b, c, g = model.generate_chunk(0, args.layers, tensors["token"].device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    stateful_ssm_token_triton_(x, tensors["stateful_h"], a_sig, b, c, g_silu, tensors["hidden"])
    return pytorch_project(tensors["hidden"], tensors, model.config.output_norm_eps)


def run_cached_original(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    x = embedding_lookup(tensors)
    stateful_ssm_token_triton_(
        x,
        tensors["stateful_h"],
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
        tensors["hidden"],
    )
    if args.projection == "pytorch":
        return pytorch_project(tensors["hidden"], tensors, tensors["model"].config.output_norm_eps)
    return triton_project(tensors["hidden"], tensors, tensors["model"].config.output_norm_eps, args.output_mode)


def run_cached_precomposed(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    x = embedding_lookup(tensors)
    precomposed_stateless_triton_(x, tensors["master_coeff"], tensors["hidden"])
    if args.projection == "pytorch":
        return pytorch_project(tensors["hidden"], tensors, tensors["model"].config.output_norm_eps)
    return triton_project(tensors["hidden"], tensors, tensors["model"].config.output_norm_eps, args.output_mode)


def run_fused_precomposed(args: argparse.Namespace, tensors: dict) -> torch.Tensor:
    if args.output_mode == "top1_only":
        fused_precomposed_top1_triton_(
            tensors["token"],
            tensors["weight"],
            tensors["master_coeff"],
            tensors["top1"],
            tensors["top1_score"],
            eps=tensors["model"].config.output_norm_eps,
        )
        return tensors["top1"]
    fused_precomposed_logits_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["logits"],
        eps=tensors["model"].config.output_norm_eps,
    )
    return tensors["logits"]


def total_fn(args: argparse.Namespace, tensors: dict) -> Callable[[], torch.Tensor]:
    tensors["output_mode"] = args.output_mode
    if args.variant == "precomposed_stateless_32_e2e":
        return lambda: run_uncached_precomposed(args, tensors)
    if args.variant == "original_stateful_32_e2e":
        return lambda: run_uncached_original(args, tensors)
    if args.variant == "original_stateful_32_cached_params":
        return lambda: run_cached_original(args, tensors)
    if args.variant == "precomposed_stateless_32_cached_master":
        return lambda: run_cached_precomposed(args, tensors)
    if args.variant == "precomposed_stateless_32_fused_e2e":
        return lambda: run_fused_precomposed(args, tensors)
    raise RuntimeError(f"unknown variant {args.variant}")


def estimated_kernel_launches(args: argparse.Namespace) -> int:
    if args.variant == "precomposed_stateless_32_fused_e2e":
        return 1
    if args.variant.endswith("_e2e"):
        return 10 if args.projection == "pytorch" else 7
    if args.projection == "pytorch":
        return 6
    return 3


def has_timed_allocations(args: argparse.Namespace) -> bool:
    if args.variant.endswith("_e2e") and args.variant != "precomposed_stateless_32_fused_e2e":
        return True
    if args.projection == "pytorch":
        return True
    return False


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


def profile_component_timings(args: argparse.Namespace, tensors: dict) -> dict[str, dict]:
    if not args.profile_components:
        return {}
    device = torch.device(args.device)
    iters = min(args.measure_iters, 1000)
    eps = tensors["model"].config.output_norm_eps
    components: list[tuple[str, Callable[[], torch.Tensor], bool, bool, str, int]] = []

    components.append(("token_input_preparation", lambda: tensors["token"], False, True, "cpu/gpu-resident", 0))
    components.append(("embedding_lookup", lambda: embedding_lookup(tensors), False, True, "gpu", 1))
    if args.variant.endswith("_e2e") and args.variant != "precomposed_stateless_32_fused_e2e":
        components.append(("dna_generation", lambda: tensors["model"].generate_chunk(0, args.layers, tensors["token"].device)[0], True, False, "gpu", 3))
        components.append(
            (
                "activation_preprocessing",
                lambda: torch.sigmoid(tensors["a_sig"].float()) + F.silu(tensors["g_silu"].float()),
                True,
                False,
                "gpu",
                2,
            )
        )
        components.append(("master_coeff_precomposition", lambda: precompose_stateless_master_coeff(tensors["coeff"]), True, False, "gpu", 1))
    else:
        components.append(("dna_generation_cached", lambda: tensors["a_sig"], False, True, "gpu", 0))
        components.append(("activation_preprocessing_cached", lambda: tensors["coeff"], False, True, "gpu", 0))
        components.append(("master_coeff_cached", lambda: tensors["master_coeff"], False, True, "gpu", 0))

    if args.variant == "precomposed_stateless_32_fused_e2e":
        components.append(
            (
                "fused_embedding_thinking_rms_projection",
                lambda: run_fused_precomposed(args, tensors),
                False,
                True,
                "gpu",
                1,
            )
        )
    else:
        if "original_stateful" in args.variant:
            components.append(
                (
                    "thinking_kernel",
                    lambda: stateful_ssm_token_triton_(
                        embedding_lookup(tensors),
                        tensors["stateful_h"],
                        tensors["a_sig"],
                        tensors["b"],
                        tensors["c"],
                        tensors["g_silu"],
                        tensors["hidden"],
                    ),
                    False,
                    True,
                    "gpu",
                    1,
                )
            )
        else:
            components.append(
                (
                    "thinking_kernel",
                    lambda: precomposed_stateless_triton_(embedding_lookup(tensors), tensors["master_coeff"], tensors["hidden"]),
                    False,
                    True,
                    "gpu",
                    1,
                )
            )
        components.append(
            (
                "rms_normalization",
                lambda: tensors["hidden"].float()
                / torch.sqrt(torch.mean(tensors["hidden"].float() * tensors["hidden"].float()) + eps),
                True,
                False,
                "gpu",
                3,
            )
        )
        if args.projection == "pytorch":
            components.append(("tied_projection_logits", lambda: pytorch_project(tensors["hidden"], tensors, eps), True, False, "gpu", 1))
        else:
            components.append(("tied_projection_logits", lambda: triton_project(tensors["hidden"], tensors, eps, "full_logits"), False, True, "gpu", 1))
        if args.output_mode == "top1_only":
            components.append(("argmax_top1", lambda: torch.argmax(tensors["logits"]), True, False, "gpu", 1))

    profiled = {}
    for name, fn, allocates, graph_ok, device_kind, launches in components:
        stats, _ = time_cuda_events(device, args.warmup_iters, iters, fn)
        profiled[name] = {
            **stats_payload(stats),
            "allocates": allocates,
            "cuda_graph_compatible": graph_ok,
            "device": device_kind,
            "kernel_launches": launches,
            "profile_measure_iters": iters,
        }
    return profiled


def compare_reference(args: argparse.Namespace, tensors: dict) -> dict[str, float | bool]:
    if args.variant != "precomposed_stateless_32_fused_e2e":
        return {}
    fused_precomposed_logits_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["logits"],
        eps=tensors["model"].config.output_norm_eps,
    )
    x = tensors["weight"][tensors["token"][0]] * tensors["master_coeff"]
    x = x.float() / torch.sqrt(torch.mean(x.float() * x.float()) + tensors["model"].config.output_norm_eps)
    expected = F.linear(x.unsqueeze(0), tensors["weight"].float()).squeeze(0)
    sync(tensors["token"].device)
    diff = (tensors["logits"].float() - expected.float()).abs()
    fused_precomposed_top1_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["top1"],
        tensors["top1_score"],
        eps=tensors["model"].config.output_norm_eps,
    )
    sync(tensors["token"].device)
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "top1_matches_full_logits": bool(int(tensors["top1"].item()) == int(torch.argmax(expected).item())),
    }


def run_once(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device != "cuda":
        raise SystemExit("Experiment 003G requires CUDA timing")
    if args.force_triton and not TRITON_AVAILABLE:
        raise SystemExit("Triton forced but unavailable")
    if args.layers != 32 or args.d_model != 256 or args.vocab_size != 256:
        raise SystemExit("Experiment 003G target requires layers=32, d_model=256, vocab_size=256")
    if args.projection != "fused" and args.variant == "precomposed_stateless_32_fused_e2e":
        raise SystemExit("precomposed_stateless_32_fused_e2e requires --projection fused")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    tensors = make_tensors(args, device)
    fn = total_fn(args, tensors)
    cuda_graph_requested = args.use_cuda_graph
    graph_capture_failure_reason = None
    cuda_graph_used = False
    with torch.no_grad():
        if cuda_graph_requested:
            try:
                stats, host_elapsed = time_cuda_graph(device, args.warmup_iters, args.measure_iters, fn)
                cuda_graph_used = True
            except Exception as exc:  # capture can fail for allocation-heavy paths
                graph_capture_failure_reason = str(exc)
                stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)
        else:
            stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)

    component_breakdown = profile_component_timings(args, tensors)
    total_mean = stats.mean_us
    for component in component_breakdown.values():
        component["percent_of_total_mean"] = (
            100.0 * component["mean_us"] / total_mean if total_mean > 0 else 0.0
        )

    allocation_notes = []
    if args.variant.endswith("_e2e") and args.variant != "precomposed_stateless_32_fused_e2e":
        allocation_notes.append("uncached DNA generation and preprocessing allocate tensors per token")
    if args.projection == "pytorch":
        allocation_notes.append("PyTorch projection/RMS path allocates intermediate tensors")
    if args.variant == "precomposed_stateless_32_fused_e2e":
        allocation_notes.append("fused path uses cached weight/master and writes only logits or top1")

    payload = {
        "experiment": "EXP003G",
        "variant": args.variant,
        "layers": args.layers,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "amp": args.amp,
        "projection": args.projection,
        "output_mode": args.output_mode,
        "cuda_graph_requested": cuda_graph_requested,
        "cuda_graph_used": cuda_graph_used,
        "graph_capture_failure_reason": graph_capture_failure_reason,
        "parameter_count": tensors["model"].trainable_parameter_count(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "triton_kernel_used": TRITON_AVAILABLE,
        "fallback_used": False,
        "kernel_launches_per_step": estimated_kernel_launches(args),
        "allocations_in_timed_path": has_timed_allocations(args),
        "allocation_notes": "; ".join(allocation_notes),
        "component_breakdown": component_breakdown,
        "host_elapsed_sec": host_elapsed,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
    }
    payload.update(stats_payload(stats))
    if args.compare_reference:
        payload.update(compare_reference(args, tensors))
    return payload


def main() -> None:
    args = parse_args()
    print(json.dumps(run_once(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
