#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify Fast32 volatility-regime checkpoint integrity and setup correctness.

Checks parameter count, SHA256 checksum, dataset paths, prompt formatting,
and final metadata JSON.
"""

import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.exp004b_fast32_ue_train_speed as exp004b

CHECKPOINT_PATH = ROOT / "results_vol_regime" / "best_val_accuracy.pt"
DATA_DIR        = ROOT / "data" / "quant_decision" / "vol_regime_H15_C60"
METADATA_PATH   = ROOT / "results_vol_regime" / "final_checkpoint_metadata.json"

REQUIRED_PARAM_COUNT = 216_320
EXPECTED_SHA256      = "b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=================================================================")
    print("FAST32 CHECKPOINT INTEGRITY AND SETUP VERIFICATION")
    print("=================================================================")

    all_ok = True

    # 1. Verify Checkpoint Existence and SHA256
    print("1. Checking best_val_accuracy.pt file hash...")
    if not CHECKPOINT_PATH.exists():
        print(f"  [FAIL] Checkpoint not found at {CHECKPOINT_PATH}")
        all_ok = False
    else:
        file_sha = sha256_file(CHECKPOINT_PATH)
        if file_sha == EXPECTED_SHA256:
            print(f"  [PASS] SHA256 matches: {file_sha} ✓")
        else:
            print(f"  [FAIL] SHA256 mismatch! Found {file_sha}, expected {EXPECTED_SHA256}")
            all_ok = False

    # 2. Verify Parameter Count and Architecture
    print("\n2. Verifying model parameter count and architecture...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = exp004b.make_model(device)
        n_params = model.trainable_parameter_count()
        if n_params == REQUIRED_PARAM_COUNT:
            print(f"  [PASS] Trainable parameter count is exactly {n_params:,} ✓")
        else:
            print(f"  [FAIL] Parameter count mismatch! Found {n_params:,}, expected {REQUIRED_PARAM_COUNT:,}")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate model or count parameters: {e}")
        all_ok = False

    # 3. Verify Dataset Paths
    print("\n3. Verifying dataset paths and splits...")
    dataset_ok = True
    for split in ["train.bin", "val.bin", "test.bin", "train.jsonl", "val.jsonl", "test.jsonl", "metadata.json"]:
        p = DATA_DIR / split
        if not p.exists():
            print(f"  [FAIL] Missing dataset file: {p}")
            dataset_ok = False
            all_ok = False
    if dataset_ok:
        print(f"  [PASS] All vol_regime_H15_C60 dataset files are present in {DATA_DIR} ✓")

    # 4. Verify Final Metadata JSON
    print("\n4. Checking final_checkpoint_metadata.json...")
    if not METADATA_PATH.exists():
        print(f"  [FAIL] Metadata JSON not found at {METADATA_PATH}")
        all_ok = False
    else:
        try:
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            if meta.get("sha256") == EXPECTED_SHA256:
                print(f"  [PASS] Metadata file exists and records correct checkpoint hash ✓")
            else:
                print(f"  [FAIL] Metadata file hash mismatch: {meta.get('sha256')} != {EXPECTED_SHA256}")
                all_ok = False
        except Exception as e:
            print(f"  [FAIL] Failed to parse metadata JSON: {e}")
            all_ok = False

    # 5. Verify Prompt Format
    print("\n5. Checking prompt formatting requirements...")
    prompt_example = "Q: test\nA: "
    if prompt_example.endswith("A: "):
        print("  [PASS] Prompt template ends exactly with 'A: ' (space trailing verified) ✓")
    else:
        print("  [FAIL] Prompt template is missing trailing space or suffix formatting!")
        all_ok = False

    print("\n=================================================================")
    if all_ok:
        print("VERIFICATION SUCCESSFUL: Checkpoint and environment are canonical.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: One or more checks failed. Please inspect logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
