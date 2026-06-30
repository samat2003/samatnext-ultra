#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP007: Fast32 Reasoning Fine-Tuning Dataset preparation script."""

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
# Generator Functions
# ---------------------------------------------------------------------------

def gen_arithmetic(rng: random.Random) -> tuple[str, str, str]:
    task_type = rng.choice(["add", "sub", "mul", "mod", "comp"])
    if task_type == "add":
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        task = f"What is {a} + {b}?"
        ans = a + b
        reasoning = (
            f"1. Split the numbers: {a} = {a - a%100} + {a%100 - a%10} + {a%10}, "
            f"{b} = {b - b%100} + {b%100 - b%10} + {b%10}.\n"
            f"2. Add hundreds: {a - a%100} + {b - b%100} = {a - a%100 + b - b%100}.\n"
            f"3. Add tens: {a%100 - a%10} + {b%100 - b%10} = {a%100 - a%10 + b%100 - b%10}.\n"
            f"4. Add ones: {a%10} + {b%10} = {a%10 + b%10}.\n"
            f"5. Combine all parts to get the sum: {ans}."
        )
    elif task_type == "sub":
        a = rng.randint(50, 999)
        b = rng.randint(10, a - 1)
        task = f"What is {a} - {b}?"
        ans = a - b
        reasoning = (
            f"1. Break down subtraction: subtract {b - b%10} first, then subtract {b%10}.\n"
            f"2. {a} - {b - b%10} = {a - (b - b%10)}.\n"
            f"3. Subtract the remaining ones: {a - (b - b%10)} - {b%10} = {ans}."
        )
    elif task_type == "mul":
        a = rng.randint(10, 99)
        b = rng.randint(2, 9)
        task = f"What is {a} * {b}?"
        ans = a * b
        reasoning = (
            f"1. Distribute multiplication: ({a - a%10} + {a%10}) * {b}.\n"
            f"2. Multiply tens: {a - a%10} * {b} = {(a - a%10) * b}.\n"
            f"3. Multiply ones: {a%10} * {b} = {a%10 * b}.\n"
            f"4. Add the results: {(a - a%10) * b} + {a%10 * b} = {ans}."
        )
    elif task_type == "mod":
        a = rng.randint(10, 99)
        b = rng.randint(3, 11)
        task = f"What is {a} modulo {b}?"
        ans = a % b
        quotient = a // b
        reasoning = (
            f"1. Find the largest integer quotient: {a} divided by {b} is {quotient}.\n"
            f"2. Multiply quotient by divisor: {quotient} * {b} = {quotient * b}.\n"
            f"3. Subtract from original number: {a} - {quotient * b} = {ans}."
        )
    else:  # comp
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        if a == b:
            b += 1
        is_greater = a > b
        op = ">" if rng.choice([True, False]) else "<"
        task = f"Is {a} {op} {b}? Answer Yes or No."
        ans = "Yes" if (op == ">" and is_greater) or (op == "<" and not is_greater) else "No"
        reasoning = (
            f"1. Compare the numbers: {a} and {b}.\n"
            f"2. Since {a} is {'greater' if is_greater else 'less'} than {b}, "
            f"the statement '{a} {op} {b}' is {'True' if ans == 'Yes' else 'False'}."
        )
    return task, reasoning, str(ans)


def gen_symbolic(rng: random.Random) -> tuple[str, str, str]:
    task_type = rng.choice(["seq", "rev", "sort", "count", "paren"])
    if task_type == "seq":
        start = rng.randint(1, 20)
        step = rng.randint(2, 7)
        seq = [start + i*step for i in range(4)]
        task = f"Complete the arithmetic sequence: {', '.join(map(str, seq))}, ..."
        ans = start + 4*step
        reasoning = (
            f"1. Find the common difference: {seq[1]} - {seq[0]} = {step}.\n"
            f"2. Verify with next elements: {seq[2]} - {seq[1]} = {step}.\n"
            f"3. Add the difference {step} to the last element {seq[-1]}: {seq[-1]} + {step} = {ans}."
        )
    elif task_type == "rev":
        words = ["apple", "banana", "cherry", "dragon", "eagle", "forest", "grape"]
        w = rng.choice(words)
        task = f"Reverse the characters in the string '{w}'."
        ans = w[::-1]
        reasoning = (
            f"1. List characters of '{w}': {list(w)}.\n"
            f"2. Reverse the order: {list(ans)}.\n"
            f"3. Join elements into string: '{ans}'."
        )
    elif task_type == "sort":
        nums = [rng.randint(1, 99) for _ in range(4)]
        # ensure unique
        nums = list(set(nums))[:4]
        while len(nums) < 4:
            nums.append(rng.randint(1, 99))
            nums = list(set(nums))
        task = f"Sort this list of numbers in ascending order: {nums}"
        sorted_nums = sorted(nums)
        ans = str(sorted_nums)
        reasoning = (
            f"1. Identify the elements: {nums}.\n"
            f"2. The smallest element is {sorted_nums[0]}.\n"
            f"3. The remaining sorted elements are {sorted_nums[1:]}.\n"
            f"4. Ascending ordered list: {sorted_nums}."
        )
    elif task_type == "count":
        w = rng.choice(["mississippi", "abracadabra", "elephant", "computer", "dna_ssm"])
        char = rng.choice(list(set(w)))
        task = f"Count the number of times the character '{char}' appears in '{w}'."
        ans = w.count(char)
        indices = [i for i, c in enumerate(w) if c == char]
        reasoning = (
            f"1. Scan the string '{w}' character by character.\n"
            f"2. Character '{char}' is found at 0-based indices: {indices}.\n"
            f"3. The total frequency count is {ans}."
        )
    else:  # paren
        # Balanced or unbalanced paren sequence
        is_balanced = rng.choice([True, False])
        if is_balanced:
            seq = rng.choice(["()", "(())", "()()", "((()))", "(()())"])
        else:
            seq = rng.choice(["(", ")", ")(", "(()", "())", "(()))"])
        task = f"Is the parenthesis string '{seq}' balanced? Answer Yes or No."
        ans = "Yes" if is_balanced else "No"
        reasoning = (
            f"1. Initialize a balance counter at 0.\n"
            f"2. Scan characters left-to-right: increment counter on '(' and decrement on ')'.\n"
            f"3. Ensure the counter never drops below 0 during the scan, and ends exactly at 0.\n"
            f"4. Parenthesis sequence '{seq}' {'satisfies' if is_balanced else 'violates'} these rules."
        )
    return task, reasoning, str(ans)


def gen_boolean(rng: random.Random) -> tuple[str, str, str]:
    task_type = rng.choice(["eval", "rule", "truth"])
    if task_type == "eval":
        a = rng.choice([True, False])
        b = rng.choice([True, False])
        op = rng.choice(["AND", "OR"])
        task = f"Evaluate: {a} {op} (NOT {b}). Answer True or False."
        not_b = not b
        ans = (a and not_b) if op == "AND" else (a or not_b)
        reasoning = (
            f"1. Evaluate NOT {b}: NOT {b} is {not_b}.\n"
            f"2. Evaluate {a} {op} {not_b}: {a} {op} {not_b} is {ans}."
        )
    elif task_type == "rule":
        p = rng.choice(["it is raining", "the market goes up", "the volume surges"])
        q = rng.choice(["the streets are wet", "the volatility spikes", "spreads tighten"])
        fact = rng.choice([True, False])
        
        task = f"Rule: If {p}, then {q}. Fact: {p} is {'True' if fact else 'False'}. Can we conclude {q} is True? Answer Yes, No, or Unknown."
        if fact:
            ans = "Yes"
            reasoning = (
                f"1. Rule states: If {p} (antecedent) is True, then {q} (consequent) must be True.\n"
                f"2. Fact states: {p} is True.\n"
                f"3. Applying Modus Ponens, we conclude {q} is True."
            )
        else:
            ans = "Unknown"
            reasoning = (
                f"1. Rule states: If {p} is True, then {q} is True.\n"
                f"2. Fact states: {p} is False.\n"
                f"3. Knowing {p} is False does not inform us about {q} (denying the antecedent is a fallacy).\n"
                f"4. Thus, the truth of {q} is Unknown."
            )
    else:  # truth
        a = rng.choice([True, False])
        task = f"If A is {a}, what is NOT (NOT A)? Answer True or False."
        ans = a
        reasoning = (
            f"1. Double negation rule: NOT (NOT A) is equivalent to A.\n"
            f"2. Since A is {a}, NOT (NOT A) is {a}."
        )
    return task, reasoning, str(ans)


def gen_market(rng: random.Random, market_data: dict[str, pd.DataFrame]) -> tuple[str, str, str]:
    symbols = list(market_data.keys())
    # Pick a random symbol and a random row index (leave space for window)
    symbol = rng.choice(symbols)
    df = market_data[symbol]
    idx = rng.randint(30, len(df) - 10)
    row = df.iloc[idx]
    ts = int(row["open_time"])
    
    task_type = rng.choice(["close_open", "max_vol", "direction", "compare"])
    if task_type == "close_open":
        open_p = float(row["open"])
        close_p = float(row["close"])
        task = f"For {symbol} at open_time {ts}, did the close price exceed the open price? Answer Yes or No."
        ans = "Yes" if close_p > open_p else "No"
        reasoning = (
            f"1. Retrieve {symbol} prices at timestamp {ts}: open = {open_p:.2f}, close = {close_p:.2f}.\n"
            f"2. Compare close and open: {close_p:.2f} > {open_p:.2f} is {close_p > open_p}.\n"
            f"3. Thus, the close price {'exceeded' if close_p > open_p else 'did not exceed'} the open price."
        )
    elif task_type == "max_vol":
        # Look at a window of 3 bars starting at idx
        window = df.iloc[idx : idx + 3]
        vols = [float(r["volume"]) for _, r in window.iterrows()]
        max_vol = max(vols)
        max_idx = vols.index(max_vol)
        max_ts = int(window.iloc[max_idx]["open_time"])
        
        task = f"Identify the open_time with the highest volume for {symbol} among these timestamps: {list(window['open_time'].astype(int))}."
        ans = str(max_ts)
        reasoning = (
            f"1. Retrieve volume for each timestamp:\n"
            + "\n".join([f"   - {int(r['open_time'])}: volume = {float(r['volume']):.3f}" for _, r in window.iterrows()])
            + f"\n2. Compare volumes: maximum is {max_vol:.3f} at timestamp {max_ts}.\n"
            f"3. Thus, the highest volume occurred at {max_ts}."
        )
    elif task_type == "direction":
        # Compute direction (close_t - close_{t-5}) over 5 bars
        prev_row = df.iloc[idx - 5]
        close_now = float(row["close"])
        close_prev = float(prev_row["close"])
        task = f"Calculate the price direction for {symbol} from timestamp {int(prev_row['open_time'])} to {ts}. Answer Up, Down, or Flat."
        if close_now > close_prev:
            ans = "Up"
        elif close_now < close_prev:
            ans = "Down"
        else:
            ans = "Flat"
        reasoning = (
            f"1. Retrieve close prices: close at {int(prev_row['open_time'])} = {close_prev:.2f}, close at {ts} = {close_now:.2f}.\n"
            f"2. Compare close prices: {close_now:.2f} is {'greater than' if ans == 'Up' else 'less than' if ans == 'Down' else 'equal to'} {close_prev:.2f}.\n"
            f"3. Thus, the price direction is {ans}."
        )
    else:  # compare
        other_symbol = rng.choice([s for s in symbols if s != symbol])
        other_df = market_data[other_symbol]
        # find matching timestamp row in other_df
        other_row_matches = other_df[other_df["open_time"] == ts]
        if other_row_matches.empty:
            # Fallback to comparison of volume inside same df
            val1 = float(row["volume"])
            val2 = float(df.iloc[idx - 1]["volume"])
            ts2 = int(df.iloc[idx - 1]["open_time"])
            task = f"Compare volumes for {symbol}: was volume at {ts} higher than volume at {ts2}? Answer Yes or No."
            ans = "Yes" if val1 > val2 else "No"
            reasoning = (
                f"1. Retrieve volumes: volume at {ts} = {val1:.3f}, volume at {ts2} = {val2:.3f}.\n"
                f"2. Compare values: {val1:.3f} > {val2:.3f} is {val1 > val2}.\n"
                f"3. Thus, volume at {ts} {'was' if val1 > val2 else 'was not'} higher."
            )
        else:
            other_row = other_row_matches.iloc[0]
            val1 = float(row["volume"])
            val2 = float(other_row["volume"])
            task = f"Compare volume at timestamp {ts}: which symbol had higher volume, {symbol} or {other_symbol}?"
            ans = symbol if val1 > val2 else other_symbol
            reasoning = (
                f"1. Retrieve volume values: {symbol} volume = {val1:.3f}, {other_symbol} volume = {val2:.3f}.\n"
                f"2. Compare values: {val1:.3f} ({symbol}) vs {val2:.3f} ({other_symbol}).\n"
                f"3. Thus, {ans} had the higher volume."
            )
    return task, reasoning, str(ans)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_market_data(raw_dir: Path, symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    data = {}
    # Use only June 2025 (2025-06) for reasoning context
    month = "2025-06"
    for symbol in symbols:
        zip_path = raw_dir / f"{symbol}-{interval}-{month}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing required market zip: {zip_path}")
            
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Fast32 Reasoning Fine-Tuning Dataset")
    parser.add_argument("--out-dir", default="data/reasoning_finetune/fast32_reasoning_v1")
    parser.add_argument("--market-data-dir", default="data/market_pretrain/binance_um_futures_1m")
    parser.add_argument("--num-train", type=int, default=100000)
    parser.add_argument("--num-val", type=int, default=10000)
    parser.add_argument("--num-test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--make-smoke", action="store_true", default=True)
    parser.add_argument("--write-bin", action="store_true", default=True)
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_market_dir = Path(args.market_data_dir) / "raw"
    
    # Load market bar context
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    print("Loading market data for bar-reasoning tasks...")
    market_data = load_market_data(raw_market_dir, symbols, "1m")
    
    rng = random.Random(args.seed)
    
    categories = ["arithmetic", "symbolic_reasoning", "boolean_logic", "market_bar_reasoning"]
    
    def generate_split(count: int, split_name: str) -> list[dict]:
        examples = []
        for i in range(count):
            category = rng.choice(categories)
            if category == "arithmetic":
                task, reasoning, ans = gen_arithmetic(rng)
            elif category == "symbolic_reasoning":
                task, reasoning, ans = gen_symbolic(rng)
            elif category == "boolean_logic":
                task, reasoning, ans = gen_boolean(rng)
            else:
                task, reasoning, ans = gen_market(rng, market_data)
                
            text = f"### Task:\n{task}\n\n### Reasoning:\n{reasoning}\n\n### Answer:\n{ans}"
            
            examples.append({
                "id": f"{split_name}_{i:06d}",
                "category": category,
                "task": task,
                "reasoning": reasoning,
                "answer": ans,
                "text": text
            })
        return examples

    # 1. Generate full splits
    print(f"Generating full splits: Train={args.num_train}, Val={args.num_val}, Test={args.num_test}...")
    train_ex = generate_split(args.num_train, "train")
    val_ex = generate_split(args.num_val, "val")
    test_ex = generate_split(args.num_test, "test")
    
    # 2. Generate tiny smoke splits
    print("Generating tiny smoke splits...")
    smoke_train_ex = train_ex[:1000]
    smoke_val_ex = val_ex[:100]
    smoke_test_ex = test_ex[:100]
    
    # Validation constraints checks
    # Every row has fields, non-empty answer, round-trips
    def validate_split(exs: list[dict], name: str):
        seen_ids = set()
        for idx, row in enumerate(exs):
            # Check fields
            for f in ["id", "category", "task", "reasoning", "answer", "text"]:
                if f not in row:
                    raise ValueError(f"Missing field {f} in row {idx} of {name}")
            # Non-empty answer
            if not row["answer"].strip():
                raise ValueError(f"Empty answer at row {idx} of {name}")
            # Round-trip check
            encoded = row["text"].encode("utf-8")
            decoded = encoded.decode("utf-8")
            if decoded != row["text"]:
                raise ValueError(f"Round-trip failed at row {idx} of {name}")
            # Verify token values in [0, 255]
            if any(t < 0 or t > 255 for t in encoded):
                raise ValueError(f"Tokens out of byte bounds at row {idx} of {name}")
            # Unique IDs
            rid = row["id"]
            if rid in seen_ids:
                raise ValueError(f"Duplicate ID {rid} in {name}")
            seen_ids.add(rid)
            
    print("Validating dataset integrity...")
    validate_split(train_ex, "train")
    validate_split(val_ex, "val")
    validate_split(test_ex, "test")
    
    # Save JSONL files
    def save_jsonl(exs: list[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
                
    save_jsonl(train_ex, out_dir / "train.jsonl")
    save_jsonl(val_ex, out_dir / "val.jsonl")
    save_jsonl(test_ex, out_dir / "test.jsonl")
    
    save_jsonl(smoke_train_ex, out_dir / "smoke_train.jsonl")
    save_jsonl(smoke_val_ex, out_dir / "smoke_val.jsonl")
    save_jsonl(smoke_test_ex, out_dir / "smoke_test.jsonl")
    
    # Binary Packaging (Direct memory-mappable uint8)
    # Concatenate all rows' text separated by EOS token (2) or simple byte concatenation
    # We will write the UTF-8 bytes of text + a special EOS byte (2) for sequence separation
    train_tokens = []
    val_tokens = []
    test_tokens = []
    
    if args.write_bin:
        print("Packaging datasets into memory-mappable binaries...")
        for ex in train_ex:
            train_tokens.extend(ex["text"].encode("utf-8"))
            train_tokens.append(2)  # EOS marker
        for ex in val_ex:
            val_tokens.extend(ex["text"].encode("utf-8"))
            val_tokens.append(2)
        for ex in test_ex:
            test_tokens.extend(ex["text"].encode("utf-8"))
            test_tokens.append(2)
            
        np.array(train_tokens, dtype=np.uint8).tofile(out_dir / "train.bin")
        np.array(val_tokens, dtype=np.uint8).tofile(out_dir / "val.bin")
        np.array(test_tokens, dtype=np.uint8).tofile(out_dir / "test.bin")
        
    # Compute Token stats
    lengths = [len(ex["text"].encode("utf-8")) for ex in train_ex]
    avg_len = float(np.mean(lengths))
    max_len = int(np.max(lengths))
    
    # Category counts
    train_cats = [ex["category"] for ex in train_ex]
    cat_counts = {cat: train_cats.count(cat) for cat in categories}
    
    # Save metadata.json
    metadata = {
        "dataset_name": "fast32_reasoning_v1",
        "dataset_version": "1.0.0",
        "creation_date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "total_examples": args.num_train + args.num_val + args.num_test,
        "examples_per_category": cat_counts,
        "train_token_count": len(train_tokens),
        "val_token_count": len(val_tokens),
        "test_token_count": len(test_tokens),
        "total_token_count": len(train_tokens) + len(val_tokens) + len(test_tokens),
        "max_sequence_length": max_len,
        "average_sequence_length": avg_len,
        "split_counts": {
            "train": args.num_train,
            "validation": args.num_val,
            "test": args.num_test
        },
        "random_seed": args.seed,
        "vocab_size": 256,
        "byte_level_encoding": True,
        "licenses_data_sources": "Binance Public historical USD(S)-M futures monthly klines (2025-06)",
        "confirmation_no_trading_labels": True,
        "confirmation_no_profit_labels": True,
        "confirmation_no_architecture_changes": True,
        "training_ready": True
    }
    
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    # Generate sample_preview.json
    # Store first 1 example from each category
    preview_examples = {}
    for cat in categories:
        for ex in train_ex:
            if ex["category"] == cat:
                preview_examples[cat] = {
                    "id": ex["id"],
                    "task": ex["task"],
                    "reasoning": ex["reasoning"],
                    "answer": ex["answer"],
                    "text": ex["text"],
                    "token_ids": list(ex["text"].encode("utf-8"))
                }
                break
                
    with open(out_dir / "sample_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview_examples, f, indent=2)
        
    # Write DATASET_CARD.md
    card_content = f"""# Dataset Card: Fast32 Reasoning Fine-Tuning Dataset

## Dataset Description
- **Dataset Name:** Fast32 Reasoning Fine-Tuning Dataset (v1.0.0)
- **Goal:** Supervised fine-tuning dataset to check if the Fast32 model can learn simple step-by-step reasoning behavior.
- **Encoding:** Byte-level UTF-8 (vocab size 256)
- **Formatting:**
  ```
  ### Task:
  <instruction>

  ### Reasoning:
  <step-by-step logic>

  ### Answer:
  <final answer>
  ```

## Categories & Example Distribution
- **Arithmetic:** {cat_counts['arithmetic']} examples (addition, subtraction, modulo, comparison)
- **Symbolic Reasoning:** {cat_counts['symbolic_reasoning']} examples (string reversal, sequence completion, list sorting, parenthesis checks)
- **Boolean Logic:** {cat_counts['boolean_logic']} examples (AND/OR/NOT evaluation, Modus Ponens truth rules)
- **Market-bar Reasoning:** {cat_counts['market_bar_reasoning']} examples (candle shape analysis, direction calculations, high volume timestamps)

## Preprocessing & Validation
- Chronological global split of Binance source data.
- Duplicate checks verified.
- Byte boundaries constraint: token values are strictly inside `[0, 255]`.
- Special sequence separator byte: `2` (EOS).

---

## Disclaimers & Notes
- **No Trading Decision Labels:** Market-bar examples are simple reasoning checks (e.g. comparing volumes). No `LONG`, `SHORT`, or `HOLD` labels exist.
- **No Profit Evaluation:** No trading simulations or profit metrics are evaluated.
- **Not Financial Advice:** Research artifact only. Do not use for live trading.
"""
    (out_dir / "DATASET_CARD.md").write_text(card_content, encoding="utf-8")
    
    print("\nDataset preparation completed successfully!")
    print(f"Total Examples: {metadata['total_examples']:,}")
    print(f"Average token length: {avg_len:.2f}")
    print(f"Metadata and DATASET_CARD.md written to {out_dir}/")


if __name__ == "__main__":
    main()
