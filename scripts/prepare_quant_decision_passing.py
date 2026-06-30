#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP010B: Filter audited SFT dataset for passing direction tasks only."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def filter_and_package():
    src_dir = ROOT / "data" / "quant_decision" / "fast32_quant_decision_v1_audited"
    
    print("Loading audited splits...")
    def load_filtered(p):
        examples = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if ex["task"] in ["binary_dir", "three_class_dir"]:
                    examples.append(ex)
        return examples
        
    train_exs = load_filtered(src_dir / "train.jsonl")
    val_exs = load_filtered(src_dir / "val.jsonl")
    test_exs = load_filtered(src_dir / "test.jsonl")
    
    print(f"Filtered Row Counts (passing direction tasks only):\n  Train: {len(train_exs):,}\n  Val:   {len(val_exs):,}\n  Test:  {len(test_exs):,}")
    
    # Save filtered JSONLs
    def save_jsonl(exs, p):
        with open(p, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
                
    save_jsonl(train_exs, src_dir / "train_passing.jsonl")
    save_jsonl(val_exs, src_dir / "val_passing.jsonl")
    save_jsonl(test_exs, src_dir / "test_passing.jsonl")
    
    # Package binaries
    train_tokens = []
    val_tokens = []
    test_tokens = []
    for ex in train_exs:
        train_tokens.extend(ex["text"].encode("utf-8"))
        train_tokens.append(2)
    for ex in val_exs:
        val_tokens.extend(ex["text"].encode("utf-8"))
        val_tokens.append(2)
    for ex in test_exs:
        test_tokens.extend(ex["text"].encode("utf-8"))
        test_tokens.append(2)
        
    np.array(train_tokens, dtype=np.uint8).tofile(src_dir / "train_passing.bin")
    np.array(val_tokens, dtype=np.uint8).tofile(src_dir / "val_passing.bin")
    np.array(test_tokens, dtype=np.uint8).tofile(src_dir / "test_passing.bin")
    
    print("Packaged passing direction SFT datasets successfully!")

if __name__ == "__main__":
    filter_and_package()
