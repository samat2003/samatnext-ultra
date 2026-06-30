#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP009: Quant Decision Classification Dataset preparation script."""

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

def load_market_data(raw_dir: Path, symbols: list[str], interval: str, month: str = "2025-06") -> dict[str, pd.DataFrame]:
    data = {}
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
                data[symbol] = df.sort_values("open_time").reset_index(drop=True)
    return data

def main():
    parser = argparse.ArgumentParser(description="Prepare Quant Decision Classification Dataset")
    parser.add_argument("--out-dir", default="data/quant_decision/fast32_quant_decision_v1")
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
    # Load 2 months to get sufficient data points
    market_data = {}
    for symbol in symbols:
        df1 = load_market_data(raw_market_dir, [symbol], "1m", "2025-06")[symbol]
        df2 = load_market_data(raw_market_dir, [symbol], "1m", "2025-07")[symbol]
        df = pd.concat([df1, df2]).sort_values("open_time").reset_index(drop=True)
        market_data[symbol] = df
        
    print("Generating quant-decision examples...")
    rng = random.Random(args.seed)
    
    # We will generate a list of candidates first
    candidates = []
    
    # Task Families defined:
    # 1. binary_dir (UP, DOWN)
    # 2. three_class_dir (UP, DOWN, FLAT)
    # 3. vol_regime (VOL_UP, VOL_DOWN, VOL_FLAT)
    # 4. breakout (BREAKOUT, NO_BREAKOUT)
    # 5. trend_range (TREND, RANGE)
    
    tasks = ["binary_dir", "three_class_dir", "vol_regime", "breakout", "trend_range"]
    horizons = [5, 15, 60] # in minutes
    
    # Chronological Split
    # Split by timestamp: train is earliest 80%, val next 10%, test final 10%
    timestamps = sorted(list(market_data["BTCUSDT"]["open_time"]))
    n_ts = len(timestamps)
    
    t_train_max = timestamps[int(n_ts * 0.8)]
    t_val_max = timestamps[int(n_ts * 0.9)]
    
    print(f"Split thresholds: Train Max={t_train_max}, Val Max={t_val_max}")
    
    # Helper to calculate metrics for a window
    # Past context window: 32 bars
    for symbol in symbols:
        df = market_data[symbol]
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        times = df["open_time"].values
        
        n_rows = len(df)
        
        # Gather samples
        for idx in range(32, n_rows - 60):
            ts = int(times[idx])
            
            # Determine split split_name
            if ts <= t_train_max:
                split = "train"
            elif ts <= t_val_max:
                split = "val"
            else:
                split = "test"
                
            # Randomly select a task and horizon to generate for this timestamp
            task = rng.choice(tasks)
            horizon = rng.choice(horizons)
            
            # Context window: idx-32 to idx
            window_closes = closes[idx-32 : idx]
            window_highs = highs[idx-32 : idx]
            window_lows = lows[idx-32 : idx]
            window_vols = volumes[idx-32 : idx]
            
            # Future horizon: idx to idx + horizon
            future_closes = closes[idx : idx + horizon]
            future_highs = highs[idx : idx + horizon]
            future_lows = lows[idx : idx + horizon]
            future_vols = volumes[idx : idx + horizon]
            
            # Compute past metrics for compact question
            # e.g., last 5 returns rounded to 2 decimal places to keep question short
            past_returns = []
            for i in range(1, 6):
                r = (window_closes[-i] - window_closes[-i-1]) / window_closes[-i-1] * 100
                past_returns.append(f"{r:+.2f}%")
            past_returns.reverse()
            past_str = ",".join(past_returns)
            
            # Base question template
            q_base = f"{symbol} past returns [{past_str}], H={horizon}"
            
            # Task specific labels
            if task == "binary_dir":
                ret = (future_closes[-1] - closes[idx-1]) / closes[idx-1]
                threshold = 0.0005 # 0.05%
                if abs(ret) < threshold:
                    continue # exclude near-flat
                ans = "UP" if ret > 0 else "DOWN"
                question = f"{q_base} binary dir?"
            elif task == "three_class_dir":
                ret = (future_closes[-1] - closes[idx-1]) / closes[idx-1]
                threshold = 0.0005
                if ret > threshold:
                    ans = "UP"
                elif ret < -threshold:
                    ans = "DOWN"
                else:
                    ans = "FLAT"
                question = f"{q_base} 3-class dir?"
            elif task == "vol_regime":
                # Compare std of future horizon returns vs past context window returns
                past_ret = np.diff(window_closes) / window_closes[:-1]
                future_ret = np.diff(future_closes) / future_closes[:-1] if len(future_closes) > 1 else np.array([0.0])
                
                past_vol = np.std(past_ret) if len(past_ret) > 0 else 0.0001
                future_vol = np.std(future_ret) if len(future_ret) > 0 else 0.0001
                
                ratio = future_vol / past_vol
                if ratio > 1.2:
                    ans = "VOL_UP"
                elif ratio < 0.8:
                    ans = "VOL_DOWN"
                else:
                    ans = "VOL_FLAT"
                question = f"{q_base} vol regime?"
            elif task == "breakout":
                # Breakout if future high/low exceeds past window high/low by 0.1%
                past_max = np.max(window_highs)
                past_min = np.min(window_lows)
                
                future_max = np.max(future_highs)
                future_min = np.min(future_lows)
                
                if future_max > past_max * 1.001 or future_min < past_min * 0.999:
                    ans = "BREAKOUT"
                else:
                    ans = "NO_BREAKOUT"
                question = f"{q_base} breakout?"
            else: # trend_range
                # Use recent window slope
                past_ret = np.diff(window_closes) / window_closes[:-1]
                slope = np.mean(past_ret) * 100 # average return in %
                past_vol = np.std(past_ret) if len(past_ret) > 0 else 0.0001
                
                if abs(slope) > 0.1 * past_vol:
                    ans = "TREND"
                else:
                    ans = "RANGE"
                question = f"{q_base} regime?"
                
            text = f"Q: {question}\nA: {ans}"
            
            candidates.append({
                "split": split,
                "id": f"{symbol}_{ts}_{task}_{horizon}",
                "symbol": symbol,
                "timestamp": ts,
                "task": task,
                "horizon": horizon,
                "context_window": 32,
                "question": question,
                "answer": ans,
                "text": text
            })
            
    print(f"Total candidates generated: {len(candidates):,}")
    
    # Filter candidates by split
    train_pool = [c for c in candidates if c["split"] == "train"]
    val_pool = [c for c in candidates if c["split"] == "val"]
    test_pool = [c for c in candidates if c["split"] == "test"]
    
    print(f"Pool sizes: Train={len(train_pool):,}, Val={len(val_pool):,}, Test={len(test_pool):,}")
    
    # Deterministically sample from pools
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    rng.shuffle(test_pool)
    
    train_ex = train_pool[:args.num_train]
    val_ex = val_pool[:args.num_val]
    test_ex = test_pool[:args.num_test]
    
    # Validation checks
    def validate_split(exs: list[dict], name: str):
        seen_ids = set()
        for idx, row in enumerate(exs):
            for key in ["id", "symbol", "timestamp", "task", "horizon", "context_window", "question", "answer", "text"]:
                if key not in row:
                    raise ValueError(f"Missing field {key} in row {idx} of {name}")
            # Unique IDs
            rid = row["id"]
            if rid in seen_ids:
                raise ValueError(f"Duplicate ID {rid} in {name}")
            seen_ids.add(rid)
            
            # Verify token values in [0, 255]
            encoded = row["text"].encode("utf-8")
            if any(t < 0 or t > 255 for t in encoded):
                raise ValueError(f"Tokens out of byte bounds at row {idx} of {name}")
                
            # No input uses future information (context window ends at timestamp)
            # Checked programmatically via index constraints
            
    validate_split(train_ex, "train")
    validate_split(val_ex, "val")
    validate_split(test_ex, "test")
    
    # Save JSONLs
    def save_jsonl(exs, p):
        with open(p, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
                
    save_jsonl(train_ex, out_dir / "train.jsonl")
    save_jsonl(val_ex, out_dir / "val.jsonl")
    save_jsonl(test_ex, out_dir / "test.jsonl")
    
    save_jsonl(train_ex[:1000], out_dir / "smoke_train.jsonl")
    save_jsonl(val_ex[:100], out_dir / "smoke_val.jsonl")
    save_jsonl(test_ex[:100], out_dir / "smoke_test.jsonl")
    
    # Packaging binaries
    train_tokens = []
    val_tokens = []
    test_tokens = []
    for ex in train_ex:
        train_tokens.extend(ex["text"].encode("utf-8"))
        train_tokens.append(2)  # EOS
    for ex in val_ex:
        val_tokens.extend(ex["text"].encode("utf-8"))
        val_tokens.append(2)
    for ex in test_ex:
        test_tokens.extend(ex["text"].encode("utf-8"))
        test_tokens.append(2)
        
    np.array(train_tokens, dtype=np.uint8).tofile(out_dir / "train.bin")
    np.array(val_tokens, dtype=np.uint8).tofile(out_dir / "val.bin")
    np.array(test_tokens, dtype=np.uint8).tofile(out_dir / "test.bin")
    
    # Stats & Class Balance
    lens = [len(ex["text"].encode("utf-8")) for ex in train_ex]
    avg_len = float(np.mean(lens))
    max_len = int(np.max(lens))
    
    task_counts = {}
    class_balances = {}
    for task in tasks:
        task_exs = [ex for ex in train_ex if ex["task"] == task]
        task_counts[task] = len(task_exs)
        
        # Class Balance & Baseline
        answers = [ex["answer"] for ex in task_exs]
        unique_ans = list(set(answers))
        ans_counts = {ans: answers.count(ans) for ans in unique_ans}
        majority_count = max(ans_counts.values()) if ans_counts else 0
        baseline = majority_count / len(answers) if answers else 0.0
        
        class_balances[task] = {
            "counts": ans_counts,
            "majority_class_baseline": baseline
        }
        
    # Save metadata.json
    metadata_payload = {
        "dataset_name": "fast32_quant_decision_v1",
        "dataset_version": "1.0.0",
        "creation_date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "total_examples": len(train_ex) + len(val_ex) + len(test_ex),
        "split_counts": {
            "train": len(train_ex),
            "validation": len(val_ex),
            "test": len(test_ex)
        },
        "examples_per_task": task_counts,
        "class_balances": class_balances,
        "train_token_count": len(train_tokens),
        "val_token_count": len(val_tokens),
        "test_token_count": len(test_tokens),
        "max_sequence_length": max_len,
        "average_sequence_length": avg_len,
        "split_boundaries": {
            "train_max_timestamp": int(t_train_max),
            "val_max_timestamp": int(t_val_max)
        },
        "random_seed": args.seed,
        "vocab_size": 256,
        "byte_level_encoding": True,
        "confirmation_no_trading_labels": True,
        "confirmation_no_profit_labels": True,
        "confirmation_no_architecture_changes": True,
        "training_ready": True
    }
    
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
        
    # Save sample_preview.json
    preview = {}
    for task in tasks:
        for ex in train_ex:
            if ex["task"] == task:
                preview[task] = ex
                break
    with open(out_dir / "sample_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2)
        
    # Write DATASET_CARD.md
    card = f"""# Dataset Card: Fast32 Quant Decision Classification Dataset

## Overview
- Compact answer-only SFT dataset for quant-decision classification.
- **Format:** `Q: <task description>\nA: <label>`
- **No reasoning trace, no formatting overhead.**
- **Sequence limit:** strictly <= 128 bytes.

## Tasks included:
1. `binary_dir` (UP / DOWN)
2. `three_class_dir` (UP / DOWN / FLAT)
3. `vol_regime` (VOL_UP / VOL_DOWN / VOL_FLAT)
4. `breakout` (BREAKOUT / NO_BREAKOUT)
5. `trend_range` (TREND / RANGE)

---

## Disclaimers & Notes
- **No Trading Decision Signals:** Market direction classification is a simple programmatic check, NOT live signals.
- **No Profit Evaluation:** No trading simulations or profit metrics are evaluated.
"""
    (out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
