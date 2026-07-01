#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Downloads final artifacts (checkpoint, metadata, and dataset) from GitHub Release assets.

Verifies checkpoint hash and extracts the dataset tarball automatically.
"""

import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_URL = "https://github.com/samat2003/samatnext-ultra/releases/download/fast32-vol-regime-final-v1/best_val_accuracy.pt"
METADATA_URL   = "https://github.com/samat2003/samatnext-ultra/releases/download/fast32-vol-regime-final-v1/final_checkpoint_metadata.json"
DATASET_URL    = "https://github.com/samat2003/samatnext-ultra/releases/download/fast32-vol-regime-final-v1/vol_regime_H15_C60.tar.gz"

CHECKPOINT_PATH = ROOT / "results_vol_regime" / "best_val_accuracy.pt"
METADATA_PATH   = ROOT / "results_vol_regime" / "final_checkpoint_metadata.json"
DATASET_TAR     = ROOT / "data" / "quant_decision" / "vol_regime_H15_C60.tar.gz"
DATASET_DIR     = ROOT / "data" / "quant_decision"

EXPECTED_SHA256 = "b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4"


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (read_so_far / total_size) * 100.0)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({read_so_far / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
        else:
            sys.stdout.write(f"\r  Progress: {read_so_far / 1024 / 1024:.1f} MB downloaded")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_hook)
    print("\n  Download complete.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=================================================================")
    print("FAST32 ARTIFACT DOWNLOADER")
    print("=================================================================")

    # 1. Download Checkpoint
    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint already exists at {CHECKPOINT_PATH}. Checking hash...")
        if sha256_file(CHECKPOINT_PATH) == EXPECTED_SHA256:
            print("  Hash matches expected value. Skipping download.")
        else:
            print("  Hash mismatch! Redownloading...")
            download_file(CHECKPOINT_URL, CHECKPOINT_PATH)
    else:
        download_file(CHECKPOINT_URL, CHECKPOINT_PATH)

    # Verify Checkpoint Hash
    print("Verifying checkpoint hash...")
    file_sha = sha256_file(CHECKPOINT_PATH)
    if file_sha == EXPECTED_SHA256:
        print(f"  [PASS] SHA256 matches: {file_sha} ✓")
    else:
        print(f"  [FAIL] SHA256 mismatch! Found {file_sha}, expected {EXPECTED_SHA256}")
        sys.exit(1)

    # 2. Download Metadata
    if not METADATA_PATH.exists():
        download_file(METADATA_URL, METADATA_PATH)
    else:
        print(f"Metadata already exists at {METADATA_PATH}. Skipping download.")

    # 3. Download and Extract Dataset
    target_test_file = DATASET_DIR / "vol_regime_H15_C60" / "test.jsonl"
    if target_test_file.exists():
        print(f"Dataset already extracted at {DATASET_DIR / 'vol_regime_H15_C60'}. Skipping download.")
    else:
        download_file(DATASET_URL, DATASET_TAR)
        print("Extracting dataset tarball...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DATASET_TAR, "r:gz") as tar:
            tar.extractall(path=str(DATASET_DIR))
        print("  Extraction complete.")
        
        # Cleanup tarball
        if DATASET_TAR.exists():
            os.remove(DATASET_TAR)
            print("  Cleaned up temporary tarball.")

    print("\n=================================================================")
    print("ALL ARTIFACTS DOWNLOADED AND VERIFIED.")
    print("=================================================================")


if __name__ == "__main__":
    main()
