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


BYTE_CORPUS = (
    "SamatNext Ultra Experiment 002 trains on UTF-8 bytes. "
    "The causal DNA SSM predicts the next byte from compact repeated text. "
    "Numerical stability matters more than benchmark score in this smoke run. "
    "Chunked generated vectors stay ephemeral; future bytes must not leak backward.\n"
).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 002 smoke runner")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--no-grad-only", action="store_true")
    parser.add_argument("--halt-threshold", type=float, default=1.1)
    parser.add_argument("--min-chunks", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_model(args: argparse.Namespace, device: torch.device) -> DynamicDnaSsmLM:
    config = DynamicDnaSsmConfig(
        max_layers=args.max_layers,
        chunk_size=args.chunk_size,
        halt_threshold=args.halt_threshold,
        min_chunks=args.min_chunks,
    )
    return DynamicDnaSsmLM(config).to(device)


def random_batch(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=256,
        size=(args.batch_size, args.seq_len),
        device=device,
    )


def byte_corpus_batch(args: argparse.Namespace, device: torch.device, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    corpus = torch.tensor(list(BYTE_CORPUS), dtype=torch.long, device=device)
    if corpus.numel() < args.seq_len + 1:
        repeats = (args.seq_len + 1 + corpus.numel() - 1) // corpus.numel()
        corpus = corpus.repeat(repeats)

    max_start = corpus.numel() - args.seq_len - 1
    starts = [(step * args.batch_size + item) % (max_start + 1) for item in range(args.batch_size)]
    windows = [corpus[start : start + args.seq_len + 1] for start in starts]
    batch = torch.stack(windows, dim=0)
    return batch[:, :-1], batch[:, 1:]


def measure_peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def run_causality(model: DynamicDnaSsmLM, args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    cutoff = max(0, args.seq_len // 2 - 1)
    tokens = random_batch(args, device)
    mutated = tokens.clone()
    if cutoff + 1 < args.seq_len:
        mutated[:, cutoff + 1 :] = (mutated[:, cutoff + 1 :] + 17) % 256

    with torch.no_grad():
        out_a = model(tokens, halt_threshold=args.halt_threshold, max_chunks=args.max_chunks)
        out_b = model(mutated, halt_threshold=args.halt_threshold, max_chunks=args.max_chunks)
    return float((out_a.logits[:, : cutoff + 1] - out_b.logits[:, : cutoff + 1]).abs().max().cpu())


def run_no_grad(model: DynamicDnaSsmLM, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model.eval()
    tokens = random_batch(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(tokens, halt_threshold=args.halt_threshold, max_chunks=args.max_chunks)
    elapsed = time.perf_counter() - started
    return {
        "mode": "no_grad",
        "elapsed_sec": elapsed,
        "layers_used": output.layers_used,
        "chunks_used": output.chunks_used,
        "halt_score": output.halt_score,
        "halted": output.halted,
        "hidden_rms": output.hidden_rms,
        "logits_rms": output.logits_rms,
        "peak_vram_bytes": measure_peak(device),
    }


def run_training(model: DynamicDnaSsmLM, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    optimizer_updates = 0
    last_output = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for step in range(1, args.train_steps + 1):
        tokens, targets = byte_corpus_batch(args, device, step)
        output = model(tokens, halt_threshold=args.halt_threshold, max_chunks=args.max_chunks)
        last_output = output
        loss = F.cross_entropy(output.logits.reshape(-1, 256), targets.reshape(-1))
        losses.append(float(loss.detach().cpu()))

        if step % args.update_every == 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1

    elapsed = time.perf_counter() - started
    return {
        "mode": "train",
        "elapsed_sec": elapsed,
        "train_steps": args.train_steps,
        "update_every": args.update_every,
        "optimizer_updates": optimizer_updates,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "layers_used_last": last_output.layers_used if last_output is not None else None,
        "chunks_used_last": last_output.chunks_used if last_output is not None else None,
        "halt_score_last": last_output.halt_score if last_output is not None else None,
        "halted_last": last_output.halted if last_output is not None else None,
        "hidden_rms_last": last_output.hidden_rms if last_output is not None else None,
        "logits_rms_last": last_output.logits_rms if last_output is not None else None,
        "peak_vram_bytes": measure_peak(device),
    }


def main() -> None:
    args = parse_args()
    if args.update_every <= 0:
        raise SystemExit("--update-every must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = make_model(args, device)
    param_count = model.trainable_parameter_count()

    results: dict[str, object] = {
        "parameter_count": param_count,
        "under_1m_parameters": param_count < 1_000_000,
        "device": args.device,
        "max_layers": args.max_layers,
        "chunk_size": args.chunk_size,
        "max_chunks": args.max_chunks,
        "causality_max_abs_diff": run_causality(model, args, device),
    }
    if args.no_grad_only:
        results.update(run_no_grad(model, args, device))
    else:
        results.update(run_training(model, args, device))

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
