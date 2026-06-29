#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    compare_tensors,
    fixed_ssm_reference,
    fixed_ssm_triton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 003B Triton fixed SSM benchmark")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1000)
    parser.add_argument("--amp", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--block-seq", type=int, default=None)
    parser.add_argument("--block-d", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
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


def timed(device: torch.device, warmup_iters: int, measure_iters: int, fn):
    for _ in range(warmup_iters):
        fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(measure_iters):
        fn()
    synchronize(device)
    return time.perf_counter() - started


def make_inputs(args: argparse.Namespace, device: torch.device):
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float32
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device, dtype=dtype) * 0.02

    if args.d_model == 256:
        model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=args.layers, chunk_size=args.layers)).to(device)
        with torch.no_grad():
            a, b, c, g = model.generate_chunk(0, args.layers, device)
    else:
        a = torch.randn(args.layers, args.d_model, device=device, dtype=torch.float32) * 0.01
        b = torch.randn(args.layers, args.d_model, device=device, dtype=torch.float32) * 0.01
        c = torch.randn(args.layers, args.d_model, device=device, dtype=torch.float32) * 0.01
        g = torch.randn(args.layers, args.d_model, device=device, dtype=torch.float32) * 0.01
        model = None

    a_sig = torch.sigmoid(a).to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    g_silu = F.silu(g).to(dtype)
    return x, a_sig, b, c, g_silu, model


def rate(tokens: int, layers: int, iters: int, elapsed: float):
    input_tok_s = tokens * iters / elapsed if elapsed > 0 else 0.0
    layer_tok_s = tokens * layers * iters / elapsed if elapsed > 0 else 0.0
    return input_tok_s, layer_tok_s


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device == "cpu":
        raise SystemExit("Experiment 003B Triton benchmark requires CUDA")
    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is not available")
    if args.block_seq is None:
        args.block_seq = args.seq_len
    if args.seq_len > args.block_seq:
        raise SystemExit("--seq-len must be <= --block-seq for the V0 Triton kernel")

    device = torch.device(args.device)
    x, a_sig, b, c, g_silu, model = make_inputs(args, device)
    tokens = args.batch_size * args.seq_len
    parameter_count = model.trainable_parameter_count() if model is not None else None

    reference = fixed_ssm_reference(x, a_sig, b, c, g_silu)
    triton_out = fixed_ssm_triton(
        x,
        a_sig,
        b,
        c,
        g_silu,
        block_seq=args.block_seq,
        block_d=args.block_d,
    )
    err = compare_tensors(triton_out, reference)

    reset_peak(device)
    pytorch_elapsed = timed(
        device,
        args.warmup_iters,
        args.measure_iters,
        lambda: fixed_ssm_reference(x, a_sig, b, c, g_silu),
    )
    pytorch_peak = peak_memory(device)
    pytorch_tok_s, pytorch_layer_tok_s = rate(tokens, args.layers, args.measure_iters, pytorch_elapsed)

    reset_peak(device)
    triton_elapsed = timed(
        device,
        args.warmup_iters,
        args.measure_iters,
        lambda: fixed_ssm_triton(
            x,
            a_sig,
            b,
            c,
            g_silu,
            block_seq=args.block_seq,
            block_d=args.block_d,
        ),
    )
    triton_peak = peak_memory(device)
    triton_tok_s, triton_layer_tok_s = rate(tokens, args.layers, args.measure_iters, triton_elapsed)

    result = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_available": TRITON_AVAILABLE,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "layers": args.layers,
        "amp": args.amp,
        "block_seq": args.block_seq,
        "block_d": args.block_d,
        "parameter_count": parameter_count,
        "true_fixed_layer_execution": args.layers == 1000,
        "pytorch_reference_elapsed_sec": pytorch_elapsed,
        "triton_elapsed_sec": triton_elapsed,
        "pytorch_reference_input_tok_s": pytorch_tok_s,
        "triton_input_tok_s": triton_tok_s,
        "pytorch_reference_layer_token_updates_s": pytorch_layer_tok_s,
        "triton_layer_token_updates_s": triton_layer_tok_s,
        "triton_speedup": triton_tok_s / pytorch_tok_s if pytorch_tok_s > 0 else None,
        "max_abs_error": err.max_abs_error,
        "mean_abs_error": err.mean_abs_error,
        "pytorch_peak_cuda_memory_bytes": pytorch_peak,
        "triton_peak_cuda_memory_bytes": triton_peak,
        "reached_100m_input_tok_s": triton_tok_s >= 100_000_000,
        "reached_100m_layer_token_updates_s": triton_layer_tok_s >= 100_000_000,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
