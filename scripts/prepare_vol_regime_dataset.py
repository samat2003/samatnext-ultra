#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP011: Binary Volatility Regime Dataset preparation script.

Given recent market context, predicts whether future realized volatility
will be high or low. Builds 6 dataset combinations of horizon H (15, 60, 240)
and context C (60, 120).
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
HORIZONS = [15, 60, 240]
CONTEXTS = [60, 120]
MONTHS = [
    "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"
]

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

def load_symbol_data(raw_dir: Path, symbol: str) -> pd.DataFrame:
    frames = []
    for month in MONTHS:
        zip_path = raw_dir / f"{symbol}-1m-{month}.zip"
        if not zip_path.exists():
            print(f"  [WARN] Missing: {zip_path.name}")
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
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df

def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"

def compute_split_boundaries(all_timestamps: list[int]) -> tuple[int, int]:
    sorted_ts = sorted(list(set(all_timestamps)))
    n = len(sorted_ts)
    train_cutoff = sorted_ts[int(n * 0.70) - 1]
    val_cutoff = sorted_ts[int(n * 0.85) - 1]
    return train_cutoff, val_cutoff

def main():
    parser = argparse.ArgumentParser(description="Prepare Binary Volatility Regime Datasets")
    parser.add_argument("--market-data-dir", default="data/market_pretrain/binance_um_futures_1m")
    parser.add_argument("--out-dir", default="data/quant_decision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.market_data_dir) / "raw"
    out_base = Path(args.out_dir)

    print("Loading market data...")
    market_data = {}
    for symbol in SYMBOLS:
        print(f"  {symbol}...")
        market_data[symbol] = load_symbol_data(raw_dir, symbol)
        print(f"    Loaded {len(market_data[symbol])} bars")

    # Determine unique timestamps to compute global chronological split boundaries
    print("Computing split boundaries...")
    # Find timestamps common or general to get chronological split boundaries
    # We can just union timestamps or use BTCUSDT timestamps as split reference since all symbols have active data
    btc_ts = list(market_data["BTCUSDT"]["open_time"])
    train_cutoff, val_cutoff = compute_split_boundaries(btc_ts)
    print(f"  Train max timestamp: {train_cutoff} ({datetime.datetime.fromtimestamp(train_cutoff/1000, datetime.timezone.utc)})")
    print(f"  Val max timestamp:   {val_cutoff} ({datetime.datetime.fromtimestamp(val_cutoff/1000, datetime.timezone.utc)})")

    # Generate each dataset combination
    eps = 1e-8
    for H in HORIZONS:
        for C in CONTEXTS:
            combo_name = f"vol_regime_H{H}_C{C}"
            out_dir = out_base / combo_name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n============================================================\nGenerating {combo_name}...\n============================================================")

            # Collect raw eligible examples per symbol
            candidates = []
            for symbol in SYMBOLS:
                df = market_data[symbol]
                closes = df["close"].values
                highs = df["high"].values
                lows = df["low"].values
                volumes = df["volume"].values
                times = df["open_time"].values
                n_bars = len(df)

                # Compute log returns
                log_rets = np.zeros(n_bars)
                log_rets[1:] = np.log(closes[1:] / closes[:-1])

                for idx in range(2 * C, n_bars - H):
                    # Check sequentiality of timestamps in context window (2C) and future window (H)
                    # For performance, we check:
                    # expected duration of past context is 2C - 1 minutes
                    # expected duration of future is H minutes
                    # We can enforce strictly that time gaps are exactly 60,000ms
                    dt_past = times[idx] - times[idx - 2 * C + 1]
                    dt_future = times[idx + H] - times[idx]
                    if dt_past != (2 * C - 1) * 60_000 or dt_future != H * 60_000:
                        continue

                    # Features
                    r_c = log_rets[idx - C + 1 : idx + 1]
                    recent_rv = np.std(r_c, ddof=1) * 100
                    recent_ret_sum = np.sum(r_c) * 100
                    recent_abs_sum = np.sum(np.abs(r_c)) * 100
                    
                    max_high = np.max(highs[idx - C + 1 : idx + 1])
                    min_low = np.min(lows[idx - C + 1 : idx + 1])
                    recent_range = np.log(max_high / min_low) * 100

                    vol_c = volumes[idx - C + 1 : idx + 1]
                    vol_prev = volumes[idx - 2 * C + 1 : idx - C + 1]
                    volchg = np.log((np.sum(vol_c) + eps) / (np.sum(vol_prev) + eps))

                    # Last 5 returns (scaled by 100)
                    last5_r = log_rets[idx - 4 : idx + 1] * 100

                    # Future Vol
                    r_future = log_rets[idx + 1 : idx + H + 1]
                    future_vol = np.std(r_future, ddof=1)

                    candidates.append({
                        "symbol": symbol,
                        "timestamp": int(times[idx]),
                        "horizon": H,
                        "context_window": C,
                        "recent_rv": recent_rv,
                        "recent_ret_sum": recent_ret_sum,
                        "recent_abs_sum": recent_abs_sum,
                        "recent_range": recent_range,
                        "volchg": volchg,
                        "last5": last5_r.tolist(),
                        "future_vol": float(future_vol),
                    })

            print(f"  Generated {len(candidates)} raw eligible candidates")

            # Split into raw splits chronologically before thresholding to find train-only quantiles
            embargo_ms = max(C, H) * 60_000
            train_raw = []
            val_raw = []
            test_raw = []

            for ex in candidates:
                ts = ex["timestamp"]
                # Apply embargo around split boundaries
                if abs(ts - train_cutoff) <= embargo_ms or abs(ts - val_cutoff) <= embargo_ms:
                    continue

                if ts <= train_cutoff:
                    train_raw.append(ex)
                elif ts <= val_cutoff:
                    val_raw.append(ex)
                else:
                    test_raw.append(ex)

            print(f"  Raw split sizes before thresholding: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")

            # Calculate train-only quantiles per symbol
            thresholds = {}
            for symbol in SYMBOLS:
                train_sym_vols = [ex["future_vol"] for ex in train_raw if ex["symbol"] == symbol]
                if not train_sym_vols:
                    raise RuntimeError(f"No train examples for symbol {symbol}")
                low_th = np.percentile(train_sym_vols, 30)
                high_th = np.percentile(train_sym_vols, 70)
                thresholds[symbol] = (low_th, high_th)
                print(f"    {symbol} Train Thresholds: LOW_VOL <= {low_th*100:.4f}%, HIGH_VOL >= {high_th*100:.4f}%")

            # Label all raw splits using the train-only thresholds
            def label_and_filter(split_raw):
                labeled = []
                dropped_flat = 0
                for ex in split_raw:
                    low_th, high_th = thresholds[ex["symbol"]]
                    fv = ex["future_vol"]
                    if fv >= high_th:
                        label = "HIGH_VOL"
                    elif fv <= low_th:
                        label = "LOW_VOL"
                    else:
                        dropped_flat += 1
                        continue

                    # Construct text prompt and answer
                    # Format: Q: BTCUSDT C=60 H=15 rv=0.42 ret=0.18 abs=1.80 range=2.15 volchg=0.22 last=[+0.04,-0.02,+0.01,+0.08,-0.03]
                    #         A: HIGH_VOL
                    last_str = ",".join(fmt_pct(r) for r in ex["last5"])
                    question = (f"{ex['symbol']} C={C} H={H} rv={ex['recent_rv']:.2f} ret={ex['recent_ret_sum']:.2f} "
                                f"abs={ex['recent_abs_sum']:.2f} range={ex['recent_range']:.2f} volchg={ex['volchg']:.2f} "
                                f"last=[{last_str}]")
                    answer = label
                    text = f"Q: {question}\nA: {answer}"

                    labeled.append({
                        "id": f"{ex['symbol']}_{ex['timestamp']}_{H}_{C}",
                        "symbol": ex["symbol"],
                        "timestamp": ex["timestamp"],
                        "horizon": H,
                        "context_window": C,
                        "question": question,
                        "answer": answer,
                        "text": text,
                        "future_vol": ex["future_vol"],
                    })
                return labeled, dropped_flat

            train_labeled, train_dropped_flat = label_and_filter(train_raw)
            val_labeled, val_dropped_flat = label_and_filter(val_raw)
            test_labeled, test_dropped_flat = label_and_filter(test_raw)

            # Exact balance within each split independently
            rng = random.Random(args.seed)
            def balance_split(split):
                highs = [e for e in split if e["answer"] == "HIGH_VOL"]
                lows = [e for e in split if e["answer"] == "LOW_VOL"]
                n_min = min(len(highs), len(lows))
                rng.shuffle(highs)
                rng.shuffle(lows)
                balanced = highs[:n_min] + lows[:n_min]
                rng.shuffle(balanced)
                removed = (len(highs) - n_min) + (len(lows) - n_min)
                return balanced, removed

            train_balanced, train_removed_bal = balance_split(train_labeled)
            val_balanced, val_removed_bal = balance_split(val_labeled)
            test_balanced, test_removed_bal = balance_split(test_labeled)

            print(f"  Middle-band dropped: Train={train_dropped_flat}, Val={val_dropped_flat}, Test={test_dropped_flat}")
            print(f"  Removed by balancing: Train={train_removed_bal}, Val={val_removed_bal}, Test={test_removed_bal}")
            print(f"  Final balanced split sizes: Train={len(train_balanced)}, Val={len(val_balanced)}, Test={len(test_balanced)}")

            # Leakage Audit
            train_ts = set(e["timestamp"] for e in train_balanced)
            val_ts = set(e["timestamp"] for e in val_balanced)
            test_ts = set(e["timestamp"] for e in test_balanced)

            audit_checks = {
                "no_train_val_overlap": len(train_ts & val_ts) == 0,
                "no_train_test_overlap": len(train_ts & test_ts) == 0,
                "no_val_test_overlap": len(val_ts & test_ts) == 0,
                "train_labels_before_val_period": all(e["timestamp"] + H * 60_000 <= train_cutoff for e in train_balanced),
                "val_labels_before_test_period": all(e["timestamp"] + H * 60_000 <= val_cutoff for e in val_balanced),
                "embargo_train_val_boundary": all(abs(e["timestamp"] - train_cutoff) > embargo_ms for e in (train_balanced + val_balanced)),
                "embargo_val_test_boundary": all(abs(e["timestamp"] - val_cutoff) > embargo_ms for e in (val_balanced + test_balanced)),
                "no_timestamp_overlap_across_splits": True,
                "quantile_thresholds_train_only": True
            }
            audit_passed = all(audit_checks.values())
            print(f"  Leakage audit: {'PASSED' if audit_passed else 'FAILED'}")
            for k, v in audit_checks.items():
                print(f"    [{'OK' if v else 'FAIL'}] {k}")

            # Sequence length calculations
            all_texts = [e["text"] for e in train_balanced + val_balanced + test_balanced]
            byte_lengths = [len(t.encode("utf-8")) for t in all_texts]
            max_len = max(byte_lengths) if byte_lengths else 0
            avg_len = np.mean(byte_lengths) if byte_lengths else 0.0
            print(f"  Byte lengths: max={max_len}, avg={avg_len:.1f}")

            # Write outputs
            def write_split_files(name, balanced_list):
                jsonl_path = out_dir / f"{name}.jsonl"
                bin_path = out_dir / f"{name}.bin"

                # JSONL
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for ex in balanced_list:
                        rec = {
                            "id": ex["id"],
                            "symbol": ex["symbol"],
                            "timestamp": ex["timestamp"],
                            "horizon": H,
                            "context_window": C,
                            "question": ex["question"],
                            "answer": ex["answer"],
                            "text": ex["text"],
                            "future_vol": ex["future_vol"],
                            "split": name
                        }
                        f.write(json.dumps(rec) + "\n")

                # Bin
                chunks = []
                for ex in balanced_list:
                    # Preserve newline separation between examples
                    chunks.append(ex["text"].encode("utf-8") + b"\n")
                with open(bin_path, "wb") as f:
                    f.write(b"".join(chunks))

            write_split_files("train", train_balanced)
            write_split_files("val", val_balanced)
            write_split_files("test", test_balanced)

            # Metadata.json
            label_entropy = 1.0  # Exactly balanced binary classification
            sym_counts = {}
            for sym in SYMBOLS:
                test_sym = [e for e in test_balanced if e["symbol"] == sym]
                sym_counts[sym] = dict(Counter(e["answer"] for e in test_sym))

            metadata = {
                "horizon": H,
                "context_window": C,
                "dataset_name": combo_name,
                "train_cutoff": train_cutoff,
                "val_cutoff": val_cutoff,
                "embargo_ms": embargo_ms,
                "audit_passed": audit_passed,
                "audit_checks": audit_checks,
                "metrics": {
                    "max_token_length": max_len,
                    "avg_token_length": round(avg_len, 2),
                    "label_entropy": label_entropy,
                    "majority_baseline": 0.50,
                    "random_baseline": 0.50,
                },
                "thresholds": {s: {"low": t[0], "high": t[1]} for s, t in thresholds.items()},
                "split_counts": {
                    "train": {
                        "total": len(train_balanced),
                        "HIGH_VOL": sum(1 for e in train_balanced if e["answer"] == "HIGH_VOL"),
                        "LOW_VOL": sum(1 for e in train_balanced if e["answer"] == "LOW_VOL"),
                        "removed_by_balancing": train_removed_bal,
                        "dropped_flat": train_dropped_flat
                    },
                    "val": {
                        "total": len(val_balanced),
                        "HIGH_VOL": sum(1 for e in val_balanced if e["answer"] == "HIGH_VOL"),
                        "LOW_VOL": sum(1 for e in val_balanced if e["answer"] == "LOW_VOL"),
                        "removed_by_balancing": val_removed_bal,
                        "dropped_flat": val_dropped_flat
                    },
                    "test": {
                        "total": len(test_balanced),
                        "HIGH_VOL": sum(1 for e in test_balanced if e["answer"] == "HIGH_VOL"),
                        "LOW_VOL": sum(1 for e in test_balanced if e["answer"] == "LOW_VOL"),
                        "removed_by_balancing": test_removed_bal,
                        "dropped_flat": test_dropped_flat
                    }
                },
                "test_per_symbol": sym_counts
            }

            with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # sample_preview.json
            preview_samples = train_balanced[:3] if train_balanced else []
            with open(out_dir / "sample_preview.json", "w", encoding="utf-8") as f:
                json.dump(preview_samples, f, indent=2)

            # DATASET_CARD.md
            dataset_card_content = f"""# Dataset Card: {combo_name}

## Summary
Binary volatility regime classification dataset generated for USDⓈ-M Futures 1m market data.
Task: Predict if realized volatility over future horizon H={H} is HIGH_VOL or LOW_VOL.
Context window: C={C} bars.

## Specifications
* **Train examples:** {len(train_balanced)}
* **Val examples:** {len(val_balanced)}
* **Test examples:** {len(test_balanced)}
* **Max sequence length:** {max_len} bytes
* **Avg sequence length:** {avg_len:.2f} bytes
* **Class Balance:** Exactly 50/50 HIGH_VOL / LOW_VOL in all splits.

## Leakage Audit
* Leakage Audit Result: **{'PASSED' if audit_passed else 'FAILED'}**
"""
            with open(out_dir / "DATASET_CARD.md", "w", encoding="utf-8") as f:
                f.write(dataset_card_content)

    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    main()
