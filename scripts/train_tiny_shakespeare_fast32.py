#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM

FROZEN_CONFIG = ROOT / "checkpoints" / "fast32_frozen" / "config.json"
FROZEN_CHECKSUMS = ROOT / "checkpoints" / "fast32_frozen" / "SHA256SUMS.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 004 Fast32 Tiny Shakespeare health check")
    parser.add_argument("--dataset-name", default="tiny_shakespeare")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--amp", choices=["fp16", "bf16", "off"], default="fp16")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="checkpoints/tiny_shakespeare_fast32/small_run")
    return parser.parse_args()


def import_datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Hugging Face datasets is required. Install it with `python -m pip install datasets`."
        ) from exc
    return load_dataset


def bytes_encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def bytes_decode(tokens: torch.Tensor | list[int]) -> str:
    values = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
    return bytes(int(x) % 256 for x in values).decode("utf-8", errors="replace")


def dataset_text(dataset: Any) -> str:
    pieces: list[str] = []
    for split in dataset.values():
        columns = getattr(split, "column_names", [])
        text_col = "text" if "text" in columns else columns[0]
        for row in split:
            value = row[text_col]
            if isinstance(value, str):
                pieces.append(value)
            elif isinstance(value, list):
                pieces.extend(str(item) for item in value)
            else:
                pieces.append(str(value))
    text = "\n".join(pieces)
    if not text:
        raise RuntimeError("loaded dataset did not contain any text")
    return text


def load_hf_tiny_shakespeare(name: str) -> tuple[str, str, list[str]]:
    load_dataset = import_datasets()
    tried = []
    names = [name]
    if name != "tiny_shakespeare":
        names.append("tiny_shakespeare")
    if "karpathy/tiny_shakespeare" not in names:
        names.append("karpathy/tiny_shakespeare")
    last_error: Exception | None = None
    for candidate in names:
        tried.append(candidate)
        try:
            dataset = load_dataset(candidate, trust_remote_code=True)
            return dataset_text(dataset), candidate, tried
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to load Hugging Face Tiny Shakespeare. tried={tried}. last_error={last_error}")


def split_tokens(tokens: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() < 4096:
        raise RuntimeError(f"dataset is too small for health check: {tokens.numel()} bytes")
    split = int(tokens.numel() * 0.9)
    return tokens[:split].contiguous(), tokens[split:].contiguous()


def sample_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= seq_len + 1:
        raise ValueError("data must be longer than seq_len + 1")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return x.long(), y.long()


def load_frozen_config() -> DynamicDnaSsmConfig:
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    kwargs = {field.name: payload[field.name] for field in fields(DynamicDnaSsmConfig)}
    return DynamicDnaSsmConfig(**kwargs)


def make_model(device: torch.device) -> DynamicDnaSsmLM:
    config = load_frozen_config()
    model = DynamicDnaSsmLM(config).to(device)
    if model.trainable_parameter_count() != 216_320:
        raise RuntimeError(f"parameter count changed: {model.trainable_parameter_count()}")
    return model


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def eval_loss(model: DynamicDnaSsmLM, data: torch.Tensor, args: argparse.Namespace, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 999)
    losses = []
    for _ in range(8):
        x, y = sample_batch(data, batch_size=args.batch_size, seq_len=args.seq_len, device=device, generator=generator)
        with autocast_context(device, args.amp):
            logits = model(x, return_metadata=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    *,
    model: DynamicDnaSsmLM,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataset_name: str,
    step: int,
    best_val_loss: float | None,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": json.loads(FROZEN_CONFIG.read_text(encoding="utf-8")),
            "args": vars(args),
            "dataset_name": dataset_name,
            "step": step,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_checksums_before = FROZEN_CHECKSUMS.read_text(encoding="utf-8")
    text, dataset_name, tried = load_hf_tiny_shakespeare(args.dataset_name)
    tokens = bytes_encode(text)
    train_data, val_data = split_tokens(tokens, args.seed)

    model = make_model(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    generator = torch.Generator().manual_seed(args.seed)
    fixed_batch = sample_batch(train_data, batch_size=args.batch_size, seq_len=args.seq_len, device=device, generator=generator)

    train_curve: list[dict[str, float]] = []
    val_curve: list[dict[str, float]] = []
    first_loss: float | None = None
    final_loss = math.nan
    best_val_loss: float | None = None
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    total_tokens = 0
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        if args.overfit_one_batch:
            x, y = fixed_batch
        else:
            x, y = sample_batch(train_data, batch_size=args.batch_size, seq_len=args.seq_len, device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            logits = model(x, return_metadata=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        final_loss = float(loss.detach().cpu())
        total_tokens += args.batch_size * args.seq_len

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            row = {"step": step, "loss": final_loss, "tokens_sec": total_tokens / elapsed if elapsed > 0 else 0.0}
            train_curve.append(row)
            if not args.overfit_one_batch:
                val_loss = eval_loss(model, val_data, args, device)
                val_row = {"step": step, "loss": val_loss}
                val_curve.append(val_row)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        dataset_name=dataset_name,
                        step=step,
                        best_val_loss=best_val_loss,
                        metrics={"train_curve": train_curve, "val_curve": val_curve},
                    )
            print(
                json.dumps({"step": step, "train_loss": final_loss, "best_val_loss": best_val_loss, "tokens_sec": row["tokens_sec"]}),
                flush=True,
            )

    elapsed = time.perf_counter() - start
    metrics = {
        "dataset_name": dataset_name,
        "dataset_names_tried": tried,
        "parameter_count": model.trainable_parameter_count(),
        "architecture": "original_stateful_32",
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_decreased": bool(first_loss is not None and final_loss < first_loss),
        "train_curve": train_curve,
        "val_curve": val_curve,
        "best_val_loss": best_val_loss,
        "tokens_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "checkpoint_path": str(best_path if best_path.exists() else last_path),
        "last_checkpoint_path": str(last_path),
        "frozen_checksums_unchanged": FROZEN_CHECKSUMS.read_text(encoding="utf-8") == frozen_checksums_before,
    }
    save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        args=args,
        dataset_name=dataset_name,
        step=args.steps,
        best_val_loss=best_val_loss,
        metrics=metrics,
    )
    if args.overfit_one_batch:
        save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            args=args,
            dataset_name=dataset_name,
            step=args.steps,
            best_val_loss=None,
            metrics=metrics,
        )
        metrics["checkpoint_path"] = str(best_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
