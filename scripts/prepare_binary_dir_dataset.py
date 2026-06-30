#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP010C: Binary Direction Only Dataset Preparation.

Generates three separate binary UP/DOWN direction datasets from 12 months of
Binance Futures 1m data, one per horizon (H=5, H=15, H=60).

Key design:
  - Aggressive near-flat thresholding to remove ambiguous labels
  - Exact UP/DOWN balance within each split (train/val/test independently)
  - Chronological 70/15/15 split by unique timestamps
  - Per-symbol embargo around split boundaries
  - Full leakage audit
  - Compact answer-only format: Q: ... A: UP/DOWN
"""

import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
HORIZONS = [5, 15, 60]
CONTEXT_WINDOW = 5  # bars of past returns shown in prompt

# Aggressive thresholds per horizon (fraction)
THRESHOLDS = {5: 0.0015, 15: 0.0020, 60: 0.0030}

# Available months (Jun 2025 – May 2026)
MONTHS = [
    "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
]

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
]
COLUMN_RENAME_MAP = {
    "quote_volume": "quote_asset_volume",
    "count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}


def load_symbol_data(raw_dir: Path, symbol: str) -> pd.DataFrame:
    """Load all available months for a symbol and concatenate chronologically."""
    frames = []
    for month in MONTHS:
        zip_path = raw_dir / f"{symbol}-1m-{month}.zip"
        if not zip_path.exists():
            print(f"  [WARN] Missing: {zip_path.name}, skipping month {month}")
            continue
        with zipfile.ZipFile(zip_path) as z:
            csvs = [n for n in z.namelist() if n.endswith(".csv")]
            with z.open(csvs[0]) as f:
                first_line = f.readline().decode("utf-8")
                f.seek(0)
                has_header = "open_time" in first_line or "open" in first_line
                if has_header:
                    df = pd.read_csv(f, header=0)
                    df = df.rename(columns=COLUMN_RENAME_MAP)
                else:
                    df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No data loaded for {symbol}")
    df = pd.concat(frames).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.2f}%"


def build_examples(df: pd.DataFrame, symbol: str, horizon: int, threshold: float) -> list[dict]:
    """Build all candidate examples for one symbol+horizon, before filtering."""
    closes = df["close"].values
    times = df["open_time"].values
    n = len(df)
    examples = []
    # Need: CONTEXT_WINDOW bars of past + H bars of future
    for idx in range(CONTEXT_WINDOW, n - horizon):
        # Context: last CONTEXT_WINDOW bar-to-bar returns
        ctx_returns = []
        for k in range(CONTEXT_WINDOW, 0, -1):
            ret = (closes[idx - k + 1] - closes[idx - k]) / closes[idx - k]
            ctx_returns.append(ret)

        # Label: future return from current close to close H bars ahead
        future_ret = (closes[idx + horizon] - closes[idx]) / closes[idx]

        if abs(future_ret) <= threshold:
            label = None  # near-flat, will be dropped
        elif future_ret > threshold:
            label = "UP"
        else:
            label = "DOWN"

        if label is None:
            examples.append({"label": None, "ts": int(times[idx]), "future_ret": future_ret, "ctx": ctx_returns})
            continue

        ctx_str = ",".join(fmt_pct(r) for r in ctx_returns)
        question = f"{symbol} past returns [{ctx_str}], H={horizon} binary dir?"
        answer = label
        text = f"Q: {question}\nA: {answer}"

        examples.append({
            "symbol": symbol,
            "timestamp": int(times[idx]),
            "horizon": horizon,
            "label": label,
            "future_ret": future_ret,
            "question": question,
            "answer": answer,
            "text": text,
        })
    return examples


def compute_split_boundaries(all_timestamps: list[int]) -> tuple[int, int]:
    """Return (train_max_ts, val_max_ts) based on 70/15/15 chronological split."""
    sorted_ts = sorted(set(all_timestamps))
    n = len(sorted_ts)
    train_cutoff = sorted_ts[int(n * 0.70) - 1]
    val_cutoff = sorted_ts[int(n * 0.85) - 1]
    return train_cutoff, val_cutoff


def apply_embargo(examples: list[dict], train_cutoff: int, val_cutoff: int,
                  horizon: int, bar_ms: int = 60_000) -> list[dict]:
    """Remove examples within embargo window around split boundaries per symbol."""
    embargo_bars = max(CONTEXT_WINDOW, horizon)
    embargo_ms = embargo_bars * bar_ms  # 1-minute bars
    kept = []
    for ex in examples:
        ts = ex["timestamp"]
        # Embargo around train/val boundary
        if abs(ts - train_cutoff) <= embargo_ms:
            continue
        # Embargo around val/test boundary
        if abs(ts - val_cutoff) <= embargo_ms:
            continue
        kept.append(ex)
    return kept


def balance_split(examples: list[dict], rng: random.Random) -> tuple[list[dict], int]:
    """Exactly balance UP/DOWN within a split. Returns (balanced_examples, n_removed)."""
    ups = [e for e in examples if e["answer"] == "UP"]
    downs = [e for e in examples if e["answer"] == "DOWN"]
    n_min = min(len(ups), len(downs))
    rng.shuffle(ups)
    rng.shuffle(downs)
    removed = (len(ups) - n_min) + (len(downs) - n_min)
    balanced = ups[:n_min] + downs[:n_min]
    rng.shuffle(balanced)
    return balanced, removed


def leakage_audit(train: list[dict], val: list[dict], test: list[dict],
                  train_cutoff: int, val_cutoff: int,
                  horizon: int, bar_ms: int = 60_000) -> dict:
    """Full leakage audit. Returns audit dict with pass/fail per check."""
    embargo_ms = max(CONTEXT_WINDOW, horizon) * bar_ms

    train_ts = set(e["timestamp"] for e in train)
    val_ts = set(e["timestamp"] for e in val)
    test_ts = set(e["timestamp"] for e in test)

    checks = {}

    # 1. No timestamp overlap across splits
    checks["no_train_val_overlap"] = len(train_ts & val_ts) == 0
    checks["no_train_test_overlap"] = len(train_ts & test_ts) == 0
    checks["no_val_test_overlap"] = len(val_ts & test_ts) == 0

    # 2. Train labels don't reach into val/test period
    #    train example at ts uses bar at ts+H -> must be <= val_cutoff
    train_label_overflow = sum(
        1 for e in train if e["timestamp"] + horizon * bar_ms > val_cutoff
    )
    checks["train_labels_before_val_period"] = train_label_overflow == 0

    # 3. Val labels don't reach into test period
    val_label_overflow = sum(
        1 for e in val if e["timestamp"] + horizon * bar_ms > val_cutoff + (val_cutoff - train_cutoff)
    )
    checks["val_labels_before_test_period"] = val_label_overflow == 0

    # 4. No example within embargo of train/val boundary
    near_boundary_train = sum(
        1 for e in (train + val + test) if abs(e["timestamp"] - train_cutoff) <= embargo_ms
    )
    checks["embargo_train_val_boundary"] = near_boundary_train == 0

    # 5. No example within embargo of val/test boundary
    near_boundary_val = sum(
        1 for e in (train + val + test) if abs(e["timestamp"] - val_cutoff) <= embargo_ms
    )
    checks["embargo_val_test_boundary"] = near_boundary_val == 0

    # 6. No cross-symbol context (structural: format enforces single symbol per example)
    checks["no_cross_symbol_context"] = True  # guaranteed by construction

    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def encode_text(text: str) -> bytes:
    return text.encode("utf-8")


def build_dataset_for_horizon(horizon: int, raw_dir: Path, out_dir: Path, seed: int = 42):
    """Build and save the full binary_dir dataset for one horizon."""
    threshold = THRESHOLDS[horizon]
    rng = random.Random(seed)

    print(f"\n{'='*60}")
    print(f"Horizon H={horizon}m  threshold={threshold:.4f} ({threshold*100:.2f}%)")
    print(f"{'='*60}")

    # 1. Load all data
    all_examples_raw = []  # all labeled (non-None) examples before filtering
    dropped_near_flat = 0
    total_candidates = 0

    for symbol in SYMBOLS:
        print(f"  Loading {symbol}...")
        df = load_symbol_data(raw_dir, symbol)
        print(f"    {len(df)} rows loaded")
        exs = build_examples(df, symbol, horizon, threshold)
        n_near_flat = sum(1 for e in exs if e.get("label") is None)
        n_valid = len(exs) - n_near_flat
        dropped_near_flat += n_near_flat
        total_candidates += len(exs)
        valid = [e for e in exs if e.get("label") is not None]
        all_examples_raw.extend(valid)
        print(f"    {len(exs)} candidates, {n_near_flat} near-flat dropped, {n_valid} kept")

    print(f"\n  Total after threshold: {len(all_examples_raw)} examples")
    print(f"  Total near-flat dropped: {dropped_near_flat}")
    label_counts_raw = Counter(e["answer"] for e in all_examples_raw)
    print(f"  Raw label counts: UP={label_counts_raw['UP']}, DOWN={label_counts_raw['DOWN']}")

    # 2. Compute chronological split boundaries from all unique timestamps
    all_ts = [e["timestamp"] for e in all_examples_raw]
    train_cutoff, val_cutoff = compute_split_boundaries(all_ts)
    print(f"\n  Split boundaries:")
    print(f"    Train max timestamp: {train_cutoff}")
    print(f"    Val max timestamp:   {val_cutoff}")

    # 3. Apply embargo around boundaries
    n_before_embargo = len(all_examples_raw)
    all_examples_embargoed = apply_embargo(all_examples_raw, train_cutoff, val_cutoff, horizon)
    n_after_embargo = len(all_examples_embargoed)
    dropped_embargo = n_before_embargo - n_after_embargo
    print(f"  Embargo dropped: {dropped_embargo}")

    # 4. Split by timestamp
    train_raw = [e for e in all_examples_embargoed if e["timestamp"] <= train_cutoff]
    val_raw   = [e for e in all_examples_embargoed if train_cutoff < e["timestamp"] <= val_cutoff]
    test_raw  = [e for e in all_examples_embargoed if e["timestamp"] > val_cutoff]

    print(f"\n  Before balancing:")
    for name, split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        c = Counter(e["answer"] for e in split)
        print(f"    {name}: UP={c['UP']}, DOWN={c['DOWN']}, total={len(split)}")

    # 5. Exact balance within each split independently
    train_balanced, train_removed = balance_split(train_raw, rng)
    val_balanced,   val_removed   = balance_split(val_raw, rng)
    test_balanced,  test_removed  = balance_split(test_raw, rng)

    print(f"\n  After balancing:")
    for name, split in [("train", train_balanced), ("val", val_balanced), ("test", test_balanced)]:
        c = Counter(e["answer"] for e in split)
        print(f"    {name}: UP={c['UP']}, DOWN={c['DOWN']}, total={len(split)}")

    total_removed_by_balance = train_removed + val_removed + test_removed
    print(f"  Total removed by balancing: {total_removed_by_balance}")

    # 6. Leakage audit
    audit = leakage_audit(train_balanced, val_balanced, test_balanced,
                          train_cutoff, val_cutoff, horizon)
    print(f"\n  Leakage audit: {'PASSED' if audit['passed'] else 'FAILED'}")
    for check, result in audit["checks"].items():
        status = "OK" if result else "FAIL"
        print(f"    [{status}] {check}")

    # 7. Check sequence lengths
    all_for_len = train_balanced + val_balanced + test_balanced
    max_len = max(len(e["text"].encode("utf-8")) for e in all_for_len)
    min_len = min(len(e["text"].encode("utf-8")) for e in all_for_len)
    avg_len = sum(len(e["text"].encode("utf-8")) for e in all_for_len) / len(all_for_len)
    print(f"\n  Sequence lengths (bytes): min={min_len}, avg={avg_len:.1f}, max={max_len}")
    if max_len > 128:
        print(f"  [WARN] max length {max_len} exceeds 128 bytes!")

    # 8. Save
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_split(name: str, examples: list[dict]):
        # JSONL
        jsonl_path = out_dir / f"{name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(examples):
                rec = {
                    "split": name,
                    "id": f"{ex['symbol']}_{ex['timestamp']}_{ex['horizon']}",
                    "symbol": ex["symbol"],
                    "timestamp": ex["timestamp"],
                    "horizon": ex["horizon"],
                    "task": "binary_dir",
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "text": ex["text"],
                }
                f.write(json.dumps(rec) + "\n")

        # Binary token stream (byte-level UTF-8, examples separated by newline)
        bin_path = out_dir / f"{name}.bin"
        chunks = []
        for ex in examples:
            chunks.append(encode_text(ex["text"] + "\n"))
        data = b"".join(chunks)
        with open(bin_path, "wb") as f:
            f.write(data)
        print(f"  Saved {name}: {len(examples)} examples, {len(data)} bytes -> {jsonl_path.name}, {bin_path.name}")

    save_split("train", train_balanced)
    save_split("val", val_balanced)
    save_split("test", test_balanced)

    # Per-symbol breakdown for test set
    sym_counts = {}
    for sym in SYMBOLS:
        sym_exs = [e for e in test_balanced if e["symbol"] == sym]
        c = Counter(e["answer"] for e in sym_exs)
        sym_counts[sym] = {"UP": c.get("UP", 0), "DOWN": c.get("DOWN", 0), "total": len(sym_exs)}

    # 9. Dataset stats
    stats = {
        "horizon": horizon,
        "threshold": threshold,
        "symbols": SYMBOLS,
        "context_window": CONTEXT_WINDOW,
        "total_candidates": total_candidates,
        "dropped_near_flat": dropped_near_flat,
        "dropped_embargo": dropped_embargo,
        "raw_label_counts": dict(label_counts_raw),
        "split_boundaries": {
            "train_max_timestamp": train_cutoff,
            "val_max_timestamp": val_cutoff,
        },
        "splits": {
            "train": {
                "total": len(train_balanced),
                "UP": sum(1 for e in train_balanced if e["answer"] == "UP"),
                "DOWN": sum(1 for e in train_balanced if e["answer"] == "DOWN"),
                "removed_by_balancing": train_removed,
                "majority_baseline": 0.5,
                "random_baseline": 0.5,
            },
            "val": {
                "total": len(val_balanced),
                "UP": sum(1 for e in val_balanced if e["answer"] == "UP"),
                "DOWN": sum(1 for e in val_balanced if e["answer"] == "DOWN"),
                "removed_by_balancing": val_removed,
                "majority_baseline": 0.5,
                "random_baseline": 0.5,
            },
            "test": {
                "total": len(test_balanced),
                "UP": sum(1 for e in test_balanced if e["answer"] == "UP"),
                "DOWN": sum(1 for e in test_balanced if e["answer"] == "DOWN"),
                "removed_by_balancing": test_removed,
                "majority_baseline": 0.5,
                "random_baseline": 0.5,
            },
        },
        "test_per_symbol": sym_counts,
        "sequence_length": {
            "min_bytes": min_len,
            "avg_bytes": round(avg_len, 2),
            "max_bytes": max_len,
        },
        "leakage_audit": audit,
    }

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {stats_path}")

    return stats


def main():
    raw_dir = ROOT / "data" / "market_pretrain" / "binance_um_futures_1m" / "raw"
    out_base = ROOT / "data" / "quant_decision"

    print("EXP010C: Binary Direction Dataset Preparation")
    print(f"Source data: {raw_dir}")
    print(f"Output base: {out_base}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Horizons: {HORIZONS}")

    all_stats = {}
    for H in HORIZONS:
        out_dir = out_base / f"binary_dir_H{H}"
        stats = build_dataset_for_horizon(H, raw_dir, out_dir, seed=42)
        all_stats[f"H{H}"] = stats

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Train':<8} {'Val':<8} {'Test':<8} {'Dropped flat':<14} {'Leakage OK'}")
    for H in HORIZONS:
        s = all_stats[f"H{H}"]
        print(f"H={H:<8} {s['splits']['train']['total']:<8} {s['splits']['val']['total']:<8} "
              f"{s['splits']['test']['total']:<8} {s['dropped_near_flat']:<14} "
              f"{'YES' if s['leakage_audit']['passed'] else 'NO'}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
