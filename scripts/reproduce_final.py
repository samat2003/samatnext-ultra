#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runs verify, accuracy reproduction, benchmark, and saves the local JSON report.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.exp004b_fast32_ue_train_speed as exp004b

REPORT_OUT = ROOT / "results_vol_regime" / "reproduce_final_report.json"


def run_cmd(args: list[str]) -> bool:
    r = subprocess.run(args, cwd=str(ROOT))
    return r.returncode == 0


def main():
    print("=================================================================")
    print("RUNNING FINAL REPRODUCTION PIPELINE (REMOVED)")
    print("=================================================================")

    # 1. Verify
    print("\n>>> Step 1: Running verify_checkpoint.py...")
    if not run_cmd([sys.executable, "scripts/verify_checkpoint.py"]):
        print("Error: Verification step failed!")
        sys.exit(1)

    # 2. Accuracy
    print("\n>>> Step 2: Running reproduce_final_accuracy.py...")
    if not run_cmd([sys.executable, "scripts/reproduce_final_accuracy.py"]):
        print("Error: Accuracy reproduction step failed!")
        sys.exit(1)

    # 3. Benchmark
    print("\n>>> Step 3: Running benchmark_final_vol_regime_audit.py...")
    if not run_cmd([sys.executable, "scripts/benchmark_final_vol_regime_audit.py"]):
        print("Error: Speed benchmarking step failed!")
        sys.exit(1)

    # 4. Generate local report JSON
    print("\n>>> Step 4: Generating local JSON report...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp004b.make_model(device)
    
    report = {
        "checkpoint_path": "results_vol_regime/best_val_accuracy.pt",
        "parameter_count": model.trainable_parameter_count(),
        "sha256": "b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4",
        "reproduce_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_AND_BENCHMARKED",
        "hardware_summary": {
            "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda or "N/A",
            "torch_version": torch.__version__,
        }
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {REPORT_OUT}")

    print("\n=================================================================")
    print("REPRODUCTION COMPLETE AND VERIFIED.")
    print("=================================================================")


if __name__ == "__main__":
    main()
