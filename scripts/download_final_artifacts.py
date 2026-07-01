#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Downloads final artifacts from GitHub Release assets using the gh CLI.

Supports private or access-controlled repositories by leveraging the local gh CLI session.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "results_vol_regime" / "best_val_accuracy.pt"
METADATA_PATH   = ROOT / "results_vol_regime" / "final_checkpoint_metadata.json"
DATASET_DIR     = ROOT / "data" / "quant_decision"
TARGET_TEST_FILE = DATASET_DIR / "vol_regime_H15_C60" / "test.jsonl"

RELEASE_TAG = "fast32-vol-regime-final-v1"
EXPECTED_SHA256 = "b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract tar contents only if every member remains under destination."""
    dest = destination.resolve()
    for member in tar.getmembers():
        member_name = Path(member.name)
        if member_name.is_absolute() or ".." in member_name.parts:
            raise RuntimeError(f"Unsafe tar member path rejected: {member.name}")
        target = (dest / member.name).resolve()
        if target != dest and not str(target).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Unsafe tar extraction target rejected: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Tar links are not allowed in release assets: {member.name}")
    tar.extractall(path=str(dest))


def main():
    print("=================================================================")
    print("FAST32 ARTIFACT DOWNLOADER (via gh CLI)")
    print("=================================================================")

    # Check if checkpoint already exists and is valid
    if CHECKPOINT_PATH.exists() and TARGET_TEST_FILE.exists():
        print("Verifying existing checkpoint hash...")
        try:
            if sha256_file(CHECKPOINT_PATH) == EXPECTED_SHA256:
                print("  [PASS] Existing checkpoint hash matches ✓")
                print("  All artifacts are already present. Skipping download.")
                print("=================================================================")
                return
        except Exception:
            pass

    # Create directories
    ROOT.joinpath("results_vol_regime").mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print("Downloading release assets via gh CLI to temp folder...")

        # Invoke gh release download
        cmd = [
            "gh", "release", "download", RELEASE_TAG,
            "--dir", str(tmpdir),
            "--clobber"
        ]

        try:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("\n[ERROR] Failed to run 'gh release download'.")
            print("Please ensure the GitHub CLI is installed and has access to the release assets.")
            print(f"Details: {e}")
            sys.exit(1)

        print("Downloads completed. Moving files to destination...")

        # Move checkpoint
        tmp_chk = tmpdir / "best_val_accuracy.pt"
        if not tmp_chk.exists():
            print("  [ERROR] Missing release asset: best_val_accuracy.pt")
            sys.exit(1)
        shutil.move(str(tmp_chk), str(CHECKPOINT_PATH))
        print(f"  Moved checkpoint to {CHECKPOINT_PATH}")

        # Move metadata
        tmp_meta = tmpdir / "final_checkpoint_metadata.json"
        if not tmp_meta.exists():
            print("  [ERROR] Missing release asset: final_checkpoint_metadata.json")
            sys.exit(1)
        shutil.move(str(tmp_meta), str(METADATA_PATH))
        print(f"  Moved metadata to {METADATA_PATH}")

        # Extract dataset tarball safely
        tmp_tar = tmpdir / "vol_regime_H15_C60.tar.gz"
        if not tmp_tar.exists():
            print("  [ERROR] Missing release asset: vol_regime_H15_C60.tar.gz")
            sys.exit(1)
        print("Extracting dataset tarball with path-safety checks...")
        with tarfile.open(tmp_tar, "r:gz") as tar:
            safe_extract_tar(tar, DATASET_DIR)
        print(f"  Extracted dataset to {DATASET_DIR / 'vol_regime_H15_C60'}")

    # Verify Checkpoint Hash
    print("\nVerifying downloaded checkpoint hash...")
    if not CHECKPOINT_PATH.exists():
        print("  [ERROR] Checkpoint was not successfully moved!")
        sys.exit(1)

    file_sha = sha256_file(CHECKPOINT_PATH)
    if file_sha == EXPECTED_SHA256:
        print(f"  [PASS] SHA256 matches: {file_sha} ✓")
    else:
        print(f"  [FAIL] SHA256 mismatch! Found {file_sha}, expected {EXPECTED_SHA256}")
        sys.exit(1)

    if not TARGET_TEST_FILE.exists():
        print(f"  [FAIL] Dataset extraction did not create {TARGET_TEST_FILE}")
        sys.exit(1)

    print("\n=================================================================")
    print("ALL ARTIFACTS SUCCESSFULLY DOWNLOADED AND EXTRACTED.")
    print("=================================================================")


if __name__ == "__main__":
    main()
