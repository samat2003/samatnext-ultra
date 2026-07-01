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
    compressed_state_reference,
    compressed_state_triton_,
    empty_triton_,
    fixed_ssm_triton,
    shared_state_d_reference,
    shared_state_d_triton_,
    stateful_ssm_token_triton_,
    stateless_x_only_reference,
    stateless_x_only_triton_,
    vector_only_triton_,
)


COMPRESSED_RANK = {
    "compressed_state_r1": 1,
    "compressed_state_r4": 4,
    "compressed_state_r8": 8,
    "compressed_state_r16": 16,
    "compressed_state_r32": 32,
}


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
    parser = argparse.ArgumentParser(description="Experiment 003E stateless/compressed thinking latency")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "empty_python",
            "event_noop",
            "original_003c",
            "stateful_003d",
            "stateless_x_only",
            "shared_state_d",
            "compressed_state_r1",
            "compressed_state_r4",
            "compressed_state_r8",
            "compressed_state_r16",
            "compressed_state_r32",
            "empty_triton",
            "vector_only",
        ],
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--measure-iters", type=int, default=10_000)
    parser.add_argument("--force-triton", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--mode", choices=["kernel-only", "end-to-end"], default="kernel-only")
    parser.add_argument("--block-d", type=int, default=256)
    parser.add_argument(
        "--layout",
        choices=["separate_layers_d", "packed_layers_d_param", "combined_coeff"],
        default="separate_layers_d",
    )
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--profile", action="store_true")
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


def make_model_and_tensors(args: argparse.Namespace, device: torch.device):
    torch.manual_seed(args.seed)
    dtype = dtype_for(args.amp)
    model = DynamicDnaSsmLM(
        DynamicDnaSsmConfig(max_layers=args.layers, chunk_size=args.layers, halt_threshold=1.1)
    ).to(device)
    model.eval()
    token = torch.randint(0, 256, (1, 1), device=device)
    x = model.token_embed(token).reshape(args.d_model).to(dtype).contiguous()
    x3 = x.reshape(1, 1, args.d_model).contiguous()
    out = torch.empty_like(x)
    out3 = torch.empty_like(x3)
    stateful_h = torch.zeros(args.layers, args.d_model, device=device, dtype=dtype).contiguous()
    shared_h = torch.zeros(args.d_model, device=device, dtype=dtype).contiguous()
    rank = COMPRESSED_RANK.get(args.variant, 1)
    compressed_h = torch.zeros(rank, args.d_model, device=device, dtype=dtype).contiguous()
    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, args.layers, device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    packed = torch.stack([a_sig, b, c, g_silu], dim=-1).contiguous()
    coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).to(dtype).contiguous()
    empty_out = torch.zeros(1, device=device, dtype=dtype)
    return {
        "model": model,
        "token": token,
        "x": x,
        "x3": x3,
        "out": out,
        "out3": out3,
        "stateful_h": stateful_h,
        "shared_h": shared_h,
        "compressed_h": compressed_h,
        "a_sig": a_sig,
        "b": b,
        "c": c,
        "g_silu": g_silu,
        "packed": packed,
        "coeff": coeff,
        "empty_out": empty_out,
    }


def effective_block_d(args: argparse.Namespace) -> tuple[int, str | None]:
    if args.variant in COMPRESSED_RANK:
        rank = COMPRESSED_RANK[args.variant]
        capped = min(args.block_d, max(1, 1024 // rank))
        if capped != args.block_d:
            return capped, f"compressed rank {rank} capped block_d from {args.block_d} to {capped}"
    return args.block_d, None


def run_kernel_once(args: argparse.Namespace, tensors: dict, block_d: int, use_triton: bool):
    if args.variant == "empty_python":
        return tensors["empty_out"]
    if args.variant == "event_noop":
        return tensors["empty_out"]
    if not use_triton:
        if args.variant == "stateless_x_only":
            tensors["out"].copy_(
                stateless_x_only_reference(
                    tensors["x"], tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"]
                )
            )
            return tensors["out"]
        if args.variant == "shared_state_d":
            out, state = shared_state_d_reference(
                tensors["x"], tensors["shared_h"], tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"]
            )
            tensors["out"].copy_(out)
            tensors["shared_h"].copy_(state)
            return tensors["out"]
        if args.variant in COMPRESSED_RANK:
            out, state = compressed_state_reference(
                tensors["x"],
                tensors["compressed_h"],
                tensors["a_sig"],
                tensors["b"],
                tensors["c"],
                tensors["g_silu"],
            )
            tensors["out"].copy_(out)
            tensors["compressed_h"].copy_(state)
            return tensors["out"]
        raise RuntimeError(f"fallback is not implemented for variant {args.variant}")

    if args.variant == "empty_triton":
        return empty_triton_(tensors["empty_out"])
    if args.variant == "vector_only":
        return vector_only_triton_(tensors["x"], tensors["out"], block_d=block_d)
    if args.variant == "original_003c":
        out = fixed_ssm_triton(
            tensors["x3"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            block_seq=1,
            block_d=block_d,
        )
        tensors["out3"].copy_(out)
        return tensors["out3"]
    if args.variant == "stateful_003d":
        return stateful_ssm_token_triton_(
            tensors["x"],
            tensors["stateful_h"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            tensors["out"],
            block_d=block_d,
        )
    if args.variant == "stateless_x_only":
        packed = tensors["packed"] if args.layout == "packed_layers_d_param" else None
        coeff = tensors["coeff"] if args.layout == "combined_coeff" else None
        return stateless_x_only_triton_(
            tensors["x"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            tensors["out"],
            block_d=block_d,
            packed=packed,
            coeff=coeff,
        )
    if args.variant == "shared_state_d":
        return shared_state_d_triton_(
            tensors["x"],
            tensors["shared_h"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            tensors["out"],
            block_d=block_d,
        )
    if args.variant in COMPRESSED_RANK:
        return compressed_state_triton_(
            tensors["x"],
            tensors["compressed_h"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            tensors["out"],
            block_d=block_d,
        )
    raise RuntimeError(f"unknown variant {args.variant}")


def run_end_to_end_once(args: argparse.Namespace, tensors: dict, block_d: int, use_triton: bool):
    model = tensors["model"]
    dtype = dtype_for(args.amp)
    x = model.token_embed(tensors["token"]).reshape(args.d_model).to(dtype).contiguous()
    a, b, c, g = model.generate_chunk(0, args.layers, tensors["token"].device)
    a_sig = torch.sigmoid(a).to(dtype).contiguous()
    b = b.to(dtype).contiguous()
    c = c.to(dtype).contiguous()
    g_silu = F.silu(g).to(dtype).contiguous()
    if not use_triton:
        raise RuntimeError("end-to-end fallback is not implemented")
    if args.variant == "stateless_x_only":
        packed = torch.stack([a_sig, b, c, g_silu], dim=-1).contiguous() if args.layout == "packed_layers_d_param" else None
        coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).to(dtype).contiguous() if args.layout == "combined_coeff" else None
        x = stateless_x_only_triton_(
            x,
            a_sig,
            b,
            c,
            g_silu,
            tensors["out"],
            block_d=block_d,
            packed=packed,
            coeff=coeff,
        )
    elif args.variant == "shared_state_d":
        x = shared_state_d_triton_(x, tensors["shared_h"], a_sig, b, c, g_silu, tensors["out"], block_d=block_d)
    elif args.variant in COMPRESSED_RANK:
        x = compressed_state_triton_(
            x, tensors["compressed_h"], a_sig, b, c, g_silu, tensors["out"], block_d=block_d
        )
    else:
        raise RuntimeError(f"end-to-end mode is only implemented for 003E variants, got {args.variant}")
    x = x / torch.sqrt(torch.mean(x.float() * x.float()) + model.config.output_norm_eps)
    return F.linear(x.float().unsqueeze(0), model.token_embed.weight.float())


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_python_loop(warmup_iters: int, measure_iters: int, fn: Callable[[], torch.Tensor]) -> tuple[LatencyStats, float]:
    for _ in range(warmup_iters):
        fn()
    values = []
    host_start = time.perf_counter()
    for _ in range(measure_iters):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1_000_000.0)
    return latency_stats(values), time.perf_counter() - host_start


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


def compare_reference(args: argparse.Namespace, tensors: dict, block_d: int) -> dict[str, float]:
    if args.variant == "stateless_x_only":
        expected = stateless_x_only_reference(
            tensors["x"], tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"]
        )
        actual = torch.empty_like(tensors["x"])
        stateless_x_only_triton_(
            tensors["x"],
            tensors["a_sig"],
            tensors["b"],
            tensors["c"],
            tensors["g_silu"],
            actual,
            block_d=block_d,
            packed=tensors["packed"] if args.layout == "packed_layers_d_param" else None,
            coeff=tensors["coeff"] if args.layout == "combined_coeff" else None,
        )
        sync(tensors["x"].device)
        diff = (actual.float() - expected.float()).abs()
        return {"max_abs_error": float(diff.max().item()), "mean_abs_error": float(diff.mean().item())}
    if args.variant == "shared_state_d":
        expected, expected_h = shared_state_d_reference(
            tensors["x"], tensors["shared_h"], tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"]
        )
        actual_h = tensors["shared_h"].clone()
        actual = torch.empty_like(tensors["x"])
        shared_state_d_triton_(
            tensors["x"], actual_h, tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"], actual, block_d=block_d
        )
        sync(tensors["x"].device)
        diff = torch.cat([(actual.float() - expected.float()).abs(), (actual_h.float() - expected_h.float()).abs()])
        return {"max_abs_error": float(diff.max().item()), "mean_abs_error": float(diff.mean().item())}
    if args.variant in COMPRESSED_RANK:
        expected, expected_h = compressed_state_reference(
            tensors["x"], tensors["compressed_h"], tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"]
        )
        actual_h = tensors["compressed_h"].clone()
        actual = torch.empty_like(tensors["x"])
        compressed_state_triton_(
            tensors["x"], actual_h, tensors["a_sig"], tensors["b"], tensors["c"], tensors["g_silu"], actual, block_d=block_d
        )
        sync(tensors["x"].device)
        diff = torch.cat([(actual.float() - expected.float()).abs().flatten(), (actual_h.float() - expected_h.float()).abs().flatten()])
        return {"max_abs_error": float(diff.max().item()), "mean_abs_error": float(diff.mean().item())}
    return {}


def run_once(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device != "cuda":
        raise SystemExit("Experiment 003E requires CUDA timing on this path")
    if args.force_triton and not TRITON_AVAILABLE:
        raise SystemExit("Triton forced but unavailable")
    if args.d_model != 256:
        raise SystemExit("Experiment 003E target requires --d-model 256")
    cuda_graph_requested = args.use_cuda_graph
    cuda_graph_used = args.use_cuda_graph and args.mode == "kernel-only"
    cuda_graph_error = None
    if cuda_graph_requested and not cuda_graph_used:
        cuda_graph_error = "CUDA Graph replay disabled for end-to-end mode because DNA generation allocates tensors"

    block_d, block_note = effective_block_d(args)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    tensors = make_model_and_tensors(args, device)
    use_triton = TRITON_AVAILABLE
    if args.force_triton:
        use_triton = True
    fallback_used = not use_triton

    if args.mode == "kernel-only":
        fn = lambda: run_kernel_once(args, tensors, block_d, use_triton)
    else:
        fn = lambda: run_end_to_end_once(args, tensors, block_d, use_triton)

    with torch.no_grad():
        if args.variant == "empty_python":
            stats, host_elapsed = time_python_loop(args.warmup_iters, args.measure_iters, fn)
        elif cuda_graph_used:
            stats, host_elapsed = time_cuda_graph(device, args.warmup_iters, args.measure_iters, fn)
        else:
            stats, host_elapsed = time_cuda_events(device, args.warmup_iters, args.measure_iters, fn)

    notes = []
    if block_note:
        notes.append(block_note)
    if args.variant == "stateless_x_only":
        notes.append("architectural variant: x = x + residual_scale * silu(G_i) * C_i * (A_sig_i + B_i) * x")
    if args.variant == "shared_state_d":
        notes.append("architectural variant: one persistent h[d] shared across all layers")
    if args.variant in COMPRESSED_RANK:
        notes.append(f"architectural variant: compressed state shape [{COMPRESSED_RANK[args.variant]}, d_model]")
    if stats.mean_us <= 1.0 or stats.p50_us <= 1.0:
        notes.append("near or below 1us: CUDA event timing resolution/noise must be considered")

    payload = {
        "experiment": "EXP003E",
        "variant": args.variant,
        "implementation": "triton" if use_triton else "reference",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "layers": args.layers,
        "d_model": args.d_model,
        "amp": args.amp,
        "block_d": block_d,
        "requested_block_d": args.block_d,
        "layout": args.layout,
        "mode": args.mode,
        "cuda_graph_requested": cuda_graph_requested,
        "cuda_graph_used": cuda_graph_used,
        "cuda_graph_error": cuda_graph_error,
        "kernel_launches_per_step": 0 if args.variant == "empty_python" else (1 if use_triton else 0),
        "python_dispatch_in_timed_path": args.variant == "empty_python" or not cuda_graph_used,
        "allocations_in_kernel_only_timed_path": False if args.mode == "kernel-only" and use_triton else None,
        "dna_generation_in_timed_path": args.mode == "end-to-end",
        "embedding_logits_projection_in_timed_path": args.mode == "end-to-end",
        "triton_kernel_used": use_triton and args.variant != "empty_python",
        "fallback_used": fallback_used,
        "true_fixed_depth_execution": args.variant in {"original_003c", "stateful_003d", "stateless_x_only", "shared_state_d", *COMPRESSED_RANK.keys()},
        "parameter_count": tensors["model"].trainable_parameter_count(),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "host_elapsed_sec": host_elapsed,
        "profile_requested": args.profile,
        "notes": "; ".join(notes),
    }
    payload.update(stats_payload(stats, args.layers))
    payload["target_1us_mean_reached"] = payload["mean_us"] <= 1.0
    payload["target_p99_5us_reached"] = payload["p99_us"] <= 5.0
    payload["target_100m_ltu_reached"] = payload["thinking_layer_token_updates_s"] >= 100_000_000
    payload["p99_under_50us"] = payload["p99_us"] < 50.0
    payload["p999_under_100us"] = payload["p999_us"] < 100.0
    if args.compare_reference:
        payload.update(compare_reference(args, tensors, block_d))
    return payload


def main() -> None:
    args = parse_args()
    if args.autotune:
        results = []
        for block_d in (16, 32, 64, 128, 256):
            tuned = argparse.Namespace(**vars(args))
            tuned.block_d = block_d
            results.append(run_once(tuned))
        print(json.dumps({"autotune_results": results}, indent=2, sort_keys=True))
    else:
        print(json.dumps(run_once(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
