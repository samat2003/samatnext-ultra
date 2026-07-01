# Fast32: Volatility-Regime Reproducibility Guide

Follow these steps to verify setup, reproduce the final test accuracy, and run speed benchmarks.

> [!IMPORTANT]
> Because this repository is private, clean-clone replication relies on the GitHub CLI (`gh`) to securely download checkpoint and dataset release assets. Ensure `gh` is installed and authenticated (`gh auth login`) on your system before running the commands.

---

## 1. Quick Start

Execute the following commands in your terminal:

```bash
git clone https://github.com/samat2003/samatnext-ultra.git
cd samatnext-ultra
make setup
make reproduce-final
```

---

## 2. Step-by-Step Verification

### Setup Environment
Set up and verify your virtual environment:
```bash
make setup
```

### Run Setup Integrity Check
This verifies model parameters count, checkpoint SHA256 checksum, dataset path presence, and prompt format:
```bash
make verify
```

### Reproduce Accuracy Metrics
Runs test evaluation on the full `vol_regime_H15_C60` split (83,856 examples) and prints metrics:
```bash
make reproduce-accuracy
```
*Expected output:*
- Test Accuracy: **~71.38%**
- Macro F1: **~0.706**
- Invalid Rate: **0.00%**
- Single-Class Collapse: **NO**

### Run Speed Benchmarks
Measures execution speeds across different batch configurations:
```bash
make benchmark
```

---

## 3. Dataset and Checkpoint Verification
If the test dataset (`data/quant_decision/vol_regime_H15_C60/`) is not present in your local clone:
1. Generate the volatility dataset using:
   ```bash
   python scripts/prepare_vol_regime_dataset.py
   ```
2. Verify the dataset metadata JSON:
   - Max token sequence length must be $\le 128$ bytes.
   - Target prompt ends with `"A: "` (trailing space).

Checkpoints and configs are locked under `results_vol_regime/`. Checkpoint SHA256 matches:
```text
b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4
```
