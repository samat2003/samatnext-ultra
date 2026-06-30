#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP010: Leakage and Embargo Audit script for Quant Decision Classification Dataset."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return examples

def main():
    src_dir = ROOT / "data" / "quant_decision" / "fast32_quant_decision_v1"
    dst_dir = ROOT / "data" / "quant_decision" / "fast32_quant_decision_v1_audited"
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print("=================================================================")
    # Load metadata.json to get boundaries
    with open(src_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    t_train_max = meta["split_boundaries"]["train_max_timestamp"]
    t_val_max = meta["split_boundaries"]["val_max_timestamp"]
    
    print(f"Original Split Boundaries:\n  Train Max Timestamp: {t_train_max}\n  Val Max Timestamp:   {t_val_max}")
    
    # Audit Parameters
    context_window = 32 # in minutes
    max_horizon = 60 # in minutes
    embargo_ms = max(context_window, max_horizon) * 60 * 1000 # 60 minutes in ms = 3,600,000 ms
    
    print(f"Audit parameters:\n  context_window: {context_window} mins\n  max_horizon:    {max_horizon} mins\n  embargo:        {max(context_window, max_horizon)} mins ({embargo_ms} ms)")
    
    # Load original datasets
    train_exs = load_jsonl(src_dir / "train.jsonl")
    val_exs = load_jsonl(src_dir / "val.jsonl")
    test_exs = load_jsonl(src_dir / "test.jsonl")
    
    print(f"Original Row Counts:\n  Train: {len(train_exs):,}\n  Val:   {len(val_exs):,}\n  Test:  {len(test_exs):,}")
    
    # Remove trend_range
    train_exs = [x for x in train_exs if x["task"] != "trend_range"]
    val_exs = [x for x in val_exs if x["task"] != "trend_range"]
    test_exs = [x for x in test_exs if x["task"] != "trend_range"]
    
    print(f"After Trend/Range Exclusion:\n  Train: {len(train_exs):,}\n  Val:   {len(val_exs):,}\n  Test:  {len(test_exs):,}")
    
    # Exclude examples within the embargo window before boundaries
    # A training example is excluded if its timestamp falls within [t_train_max - embargo_ms, t_train_max]
    # A validation example is excluded if its timestamp falls within [t_val_max - embargo_ms, t_val_max]
    t_train_embargo_threshold = t_train_max - embargo_ms
    t_val_embargo_threshold = t_val_max - embargo_ms
    
    train_audited = [x for x in train_exs if x["timestamp"] < t_train_embargo_threshold]
    val_audited = [x for x in val_exs if x["timestamp"] < t_val_embargo_threshold]
    test_audited = test_exs # Test has no split boundary after it, so no embargo needed at the end
    
    removed_train = len(train_exs) - len(train_audited)
    removed_val = len(val_exs) - len(val_audited)
    
    print(f"Embargo Removal:\n  Train removed: {removed_train}\n  Val removed:   {removed_val}")
    print(f"Audited Row Counts:\n  Train: {len(train_audited):,}\n  Val:   {len(val_audited):,}\n  Test:  {len(test_audited):,}")
    
    # Chronological Leakage Audit Checks:
    # 1. No train prediction horizon overlap into validation:
    # For every training sample, prediction_end_time = timestamp + horizon_ms
    # This must be strictly <= t_train_max
    for idx, ex in enumerate(train_audited):
        horizon_ms = ex["horizon"] * 60 * 1000
        end_time = ex["timestamp"] + horizon_ms
        if end_time > t_train_max:
            raise ValueError(f"Leakage Check Failed: Train example {ex['id']} reaches into validation period! end_time {end_time} > train_max {t_train_max}")
            
    # 2. No validation prediction horizon overlap into test:
    for idx, ex in enumerate(val_audited):
        horizon_ms = ex["horizon"] * 60 * 1000
        end_time = ex["timestamp"] + horizon_ms
        if end_time > t_val_max:
            raise ValueError(f"Leakage Check Failed: Val example {ex['id']} reaches into test period! end_time {end_time} > val_max {t_val_max}")
            
    print("\nCONFIRMED: Chronological leakage/embargo audit PASSED successfully!")
    print("  - No training prediction window reaches into validation.")
    print("  - No validation prediction window reaches into test.")
    
    # Save audited datasets
    def save_jsonl(exs, p):
        with open(p, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
                
    save_jsonl(train_audited, dst_dir / "train.jsonl")
    save_jsonl(val_audited, dst_dir / "val.jsonl")
    save_jsonl(test_audited, dst_dir / "test.jsonl")
    
    save_jsonl(train_audited[:1000], dst_dir / "smoke_train.jsonl")
    save_jsonl(val_audited[:100], dst_dir / "smoke_val.jsonl")
    save_jsonl(test_audited[:100], dst_dir / "smoke_test.jsonl")
    
    # Package binaries
    train_tokens = []
    val_tokens = []
    test_tokens = []
    for ex in train_audited:
        train_tokens.extend(ex["text"].encode("utf-8"))
        train_tokens.append(2)
    for ex in val_audited:
        val_tokens.extend(ex["text"].encode("utf-8"))
        val_tokens.append(2)
    for ex in test_audited:
        test_tokens.extend(ex["text"].encode("utf-8"))
        test_tokens.append(2)
        
    np.array(train_tokens, dtype=np.uint8).tofile(dst_dir / "train.bin")
    np.array(val_tokens, dtype=np.uint8).tofile(dst_dir / "val.bin")
    np.array(test_tokens, dtype=np.uint8).tofile(dst_dir / "test.bin")
    
    print("\nAudited dataset packaged and written to data/quant_decision/fast32_quant_decision_v1_audited/")

if __name__ == "__main__":
    main()
