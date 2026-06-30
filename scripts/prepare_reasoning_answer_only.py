#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP008B: Supervised Fine-Tuning Answer-Only Dataset preparation script."""

import argparse
import datetime
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore"
]

COLUMN_RENAME_MAP = {
    "quote_volume": "quote_asset_volume",
    "count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_boolean(rng: random.Random) -> tuple[str, str]:
    # Short boolean expressions: TRUE AND NOT FALSE -> TRUE
    a = rng.choice(["TRUE", "FALSE"])
    b = rng.choice(["TRUE", "FALSE"])
    op = rng.choice(["AND", "OR"])
    not_b = "TRUE" if b == "FALSE" else "FALSE"
    
    q = f"{a} {op} NOT {b}"
    
    # Calculate answer
    a_bool = (a == "TRUE")
    not_b_bool = (not_b == "TRUE")
    ans_bool = (a_bool and not_b_bool) if op == "AND" else (a_bool or not_b_bool)
    ans = "TRUE" if ans_bool else "FALSE"
    
    return q, ans

def gen_arithmetic_compare(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    q = f"{a} vs {b}"
    if a > b:
        ans = "GT"
    elif a < b:
        ans = "LT"
    else:
        ans = "EQ"
    return q, ans

def gen_arithmetic_add_sub(rng: random.Random) -> tuple[str, str]:
    op = rng.choice(["+", "-"])
    if op == "+":
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        q = f"{a} + {b}"
        ans = str(a + b)
    else:
        a = rng.randint(50, 999)
        b = rng.randint(10, a - 1)
        q = f"{a} - {b}"
        ans = str(a - b)
    return q, ans

def gen_market_direction(rng: random.Random, market_data: dict[str, pd.DataFrame]) -> tuple[str, str]:
    symbols = list(market_data.keys())
    symbol = rng.choice(symbols)
    df = market_data[symbol]
    idx = rng.randint(10, len(df) - 10)
    row = df.iloc[idx]
    prev_row = df.iloc[idx - 5]
    
    q = f"{symbol} direction {int(prev_row['open_time'])} to {int(row['open_time'])}"
    close_now = float(row["close"])
    close_prev = float(prev_row["close"])
    
    if close_now > close_prev:
        ans = "UP"
    elif close_now < close_prev:
        ans = "DOWN"
    else:
        ans = "FLAT"
    return q, ans

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_market_data(raw_dir: Path, symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    data = {}
    month = "2025-06"
    for symbol in symbols:
        zip_path = raw_dir / f"{symbol}-{interval}-{month}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing market zip: {zip_path}")
            
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            matching = [n for n in names if n.endswith(".csv")]
            with z.open(matching[0]) as f:
                first_line = f.readline().decode("utf-8")
                f.seek(0)
                has_header = "open_time" in first_line or "open" in first_line
                if has_header:
                    df = pd.read_csv(f, header=0)
                    df = df.rename(columns=COLUMN_RENAME_MAP)
                else:
                    df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                data[symbol] = df
    return data

# ---------------------------------------------------------------------------
# Core preparation logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Answer-Only SFT datasets")
    parser.add_argument("--out-dir", default="data/reasoning_finetune/fast32_reasoning_v2")
    parser.add_argument("--market-data-dir", default="data/market_pretrain/binance_um_futures_1m")
    parser.add_argument("--num-train", type=int, default=50000)
    parser.add_argument("--num-val", type=int, default=5000)
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_market_dir = Path(args.market_data_dir) / "raw"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    print("Loading market data...")
    market_data = load_market_data(raw_market_dir, symbols, "1m")
    
    rng = random.Random(args.seed)
    
    datasets = {
        "boolean_only": lambda r: gen_boolean(r),
        "arithmetic_compare_only": lambda r: gen_arithmetic_compare(r),
        "arithmetic_small_add_sub_only": lambda r: gen_arithmetic_add_sub(r),
        "market_direction_only": lambda r: gen_market_direction(r, market_data)
    }
    
    # Track metadata
    all_metadata = {}
    
    for task_name, gen_func in datasets.items():
        print(f"\nGenerating task dataset: {task_name}...")
        task_dir = out_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        def gen_split(count: int, prefix: str) -> list[dict]:
            examples = []
            for i in range(count):
                q, ans = gen_func(rng)
                text = f"Q: {q}\nA: {ans}"
                examples.append({
                    "id": f"{prefix}_{i:06d}",
                    "category": task_name,
                    "question": q,
                    "answer": ans,
                    "text": text
                })
            return examples
            
        train_ex = gen_split(args.num_train, "train")
        val_ex = gen_split(args.num_val, "val")
        test_ex = gen_split(args.num_test, "test")
        
        # Validate splits
        def validate(exs: list[dict]):
            seen_ids = set()
            for r in exs:
                assert all(k in r for k in ["id", "category", "question", "answer", "text"])
                assert r["answer"].strip()
                assert len(r["text"].encode("utf-8")) <= 128
                assert r["id"] not in seen_ids
                seen_ids.add(r["id"])
                
        validate(train_ex)
        validate(val_ex)
        validate(test_ex)
        
        # Save JSONLs
        def save_jsonl(exs, p):
            with open(p, "w", encoding="utf-8") as f:
                for ex in exs:
                    f.write(json.dumps(ex) + "\n")
                    
        save_jsonl(train_ex, task_dir / "train.jsonl")
        save_jsonl(val_ex, task_dir / "val.jsonl")
        save_jsonl(test_ex, task_dir / "test.jsonl")
        
        # Packaging binaries
        train_toks = []
        val_toks = []
        test_toks = []
        for ex in train_ex:
            train_toks.extend(ex["text"].encode("utf-8"))
            train_toks.append(2)  # EOS marker
        for ex in val_ex:
            val_toks.extend(ex["text"].encode("utf-8"))
            val_toks.append(2)
        for ex in test_ex:
            test_toks.extend(ex["text"].encode("utf-8"))
            test_toks.append(2)
            
        np.array(train_toks, dtype=np.uint8).tofile(task_dir / "train.bin")
        np.array(val_toks, dtype=np.uint8).tofile(task_dir / "val.bin")
        np.array(test_toks, dtype=np.uint8).tofile(task_dir / "test.bin")
        
        # Save tiny smoke splits
        save_jsonl(train_ex[:1000], task_dir / "smoke_train.jsonl")
        save_jsonl(val_ex[:100], task_dir / "smoke_val.jsonl")
        save_jsonl(test_ex[:100], task_dir / "smoke_test.jsonl")
        
        # Record stats
        lens = [len(ex["text"].encode("utf-8")) for ex in train_ex]
        all_metadata[task_name] = {
            "total_examples": len(train_ex) + len(val_ex) + len(test_ex),
            "train_token_count": len(train_toks),
            "val_token_count": len(val_toks),
            "max_len": int(np.max(lens)),
            "avg_len": float(np.mean(lens))
        }
        
    # Generate mixed dataset (mixed_answer_only)
    print("\nGenerating task dataset: mixed_answer_only...")
    mixed_dir = out_dir / "mixed_answer_only"
    mixed_dir.mkdir(exist_ok=True)
    
    def gen_mixed_split(count: int, prefix: str) -> list[dict]:
        examples = []
        cats = list(datasets.keys())
        for i in range(count):
            cat = rng.choice(cats)
            q, ans = datasets[cat](rng)
            text = f"Q: {q}\nA: {ans}"
            examples.append({
                "id": f"{prefix}_{i:06d}",
                "category": cat,
                "question": q,
                "answer": ans,
                "text": text
            })
        return examples
        
    train_ex = gen_mixed_split(args.num_train, "train")
    val_ex = gen_mixed_split(args.num_val, "val")
    test_ex = gen_mixed_split(args.num_test, "test")
    
    save_jsonl(train_ex, mixed_dir / "train.jsonl")
    save_jsonl(val_ex, mixed_dir / "val.jsonl")
    save_jsonl(test_ex, mixed_dir / "test.jsonl")
    
    train_toks = []
    val_toks = []
    test_toks = []
    for ex in train_ex:
        train_toks.extend(ex["text"].encode("utf-8"))
        train_toks.append(2)
    for ex in val_ex:
        val_toks.extend(ex["text"].encode("utf-8"))
        val_toks.append(2)
    for ex in test_ex:
        test_toks.extend(ex["text"].encode("utf-8"))
        test_toks.append(2)
        
    np.array(train_toks, dtype=np.uint8).tofile(mixed_dir / "train.bin")
    np.array(val_toks, dtype=np.uint8).tofile(mixed_dir / "val.bin")
    np.array(test_toks, dtype=np.uint8).tofile(mixed_dir / "test.bin")
    
    save_jsonl(train_ex[:1000], mixed_dir / "smoke_train.jsonl")
    save_jsonl(val_ex[:100], mixed_dir / "smoke_val.jsonl")
    save_jsonl(test_ex[:100], mixed_dir / "smoke_test.jsonl")
    
    lens = [len(ex["text"].encode("utf-8")) for ex in train_ex]
    all_metadata["mixed_answer_only"] = {
        "total_examples": len(train_ex) + len(val_ex) + len(test_ex),
        "train_token_count": len(train_toks),
        "val_token_count": len(val_toks),
        "max_len": int(np.max(lens)),
        "avg_len": float(np.mean(lens))
    }
    
    # Save combined metadata.json
    metadata_payload = {
        "dataset_name": "fast32_reasoning_v2",
        "creation_date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "vocab_size": 256,
        "byte_level_encoding": True,
        "tasks": all_metadata,
        "confirmation_no_trading_labels": True,
        "confirmation_no_profit_labels": True,
        "confirmation_no_architecture_changes": True,
        "training_ready": True
    }
    
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
        
    # Generate sample_preview.json
    preview = {}
    for task_name in list(datasets.keys()) + ["mixed_answer_only"]:
        with open(out_dir / task_name / "train.jsonl", "r", encoding="utf-8") as f:
            line = f.readline()
            preview[task_name] = json.loads(line)
            
    with open(out_dir / "sample_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)
        
    # Write DATASET_CARD.md
    card = """# Dataset Card: Fast32 Answer-Only Reasoning SFT (v2)

## Overview
- Supervised fine-tuning dataset formatted as:
  ```
  Q: <short task>
  A: <answer>
  ```
- **No reasoning trace, no formatting overhead.**
- **Sequence limit:** strictly <= 128 bytes.

## Tasks included:
1. `boolean_only`
2. `arithmetic_compare_only`
3. `arithmetic_small_add_sub_only`
4. `market_direction_only`
5. `mixed_answer_only`
"""
    (out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    
    print("\nDataset v2 preparation completed successfully!")

if __name__ == "__main__":
    main()
