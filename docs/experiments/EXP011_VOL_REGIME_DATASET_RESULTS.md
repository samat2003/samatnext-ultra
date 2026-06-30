# EXP011: Binary Volatility Regime Dataset Preparation Results

This document summarizes the dataset preparation and validation results for EXP011, moving from noisy raw direction labels to structurally learnable binary volatility-regime classification.

---

## Why EXP010C Failed
EXP010C tested binary direction classification on $H \in \{5, 15, 60\}$ minutes using a compact 5-bar context window. While token-level cross-entropy loss decreased, validation and test accuracy did not improve above the baseline, showing complete single-class collapse (the model predicted either all-`UP` or all-`DOWN`).
- **Insufficient context:** 5 raw return bars do not provide enough signal for directional prediction.
- **Microstructure noise:** Directional returns at 1-minute steps are extremely noisy and contain near-zero autocorrelation.
- **State bleed:** The stateful model carries SSM states across sequences, complicating sample-by-sample directional learning if no clear boundary or reset exists.

## Why Volatility Regime is the Next Task
Volatility has strong autocorrelation and exhibits clustering (high volatility is followed by high volatility, and low by low). Volatility regime classification is structurally much more learnable than return direction because the signals (realized volatility, high-low range, return magnitude) have direct predictive power over future volatility.

---

## Datasets Created & Specifications

All datasets were prepared using the full 12-month Binance Futures USDⓈ-M 1m market data for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `BNBUSDT` from June 2025 to May 2026.

### Summary of Datasets

| Dataset Combo | Train Size | Val Size | Test Size | Max Length (bytes) | Avg Length (bytes) | Middle-Band Dropped | Total Balanced Away | Leakage Audit |
|---|---|---|---|---|---|---|---|---|
| **`vol_regime_H15_C60`** | 882,576 | 156,998 | 83,856 | 128 | 123.50 | 835,672 | 141,790 | **PASSED** |
| **`vol_regime_H15_C120`** | 882,144 | 156,682 | 83,828 | 129 | 124.70 | 835,184 | 141,614 | **PASSED** |
| **`vol_regime_H60_C60`** | 882,576 | 139,558 | 78,158 | 128 | 123.50 | 838,610 | 161,810 | **PASSED** |
| **`vol_regime_H60_C120`** | 882,144 | 139,134 | 78,128 | 129 | 124.70 | 838,193 | 161,673 | **PASSED** |
| **`vol_regime_H240_C60`** | 882,144 | 131,608 | 68,644 | 129 | 124.50 | 841,903 | 172,813 | **PASSED** |
| **`vol_regime_H240_C120`** | 881,856 | 131,702 | 68,586 | 130 | 125.70 | 841,664 | 172,824 | **PASSED** |

### Formatting Features
- **Prompt suffix:** The prefix for labels is exactly `"A: "` (with trailing space).
- **Target max length:** All sequences are strictly $\le 130$ bytes (well below the target limit of 160 bytes).
- **Class Balance:** Independent exact 50/50 balance within `train`, `val`, and `test` splits.
- **Majority & Random Baseline:** Exactly **50.00%** due to exact class balancing.
- **Label Entropy:** Exactly **1.00** bits for all splits.

---

## Leakage Audit Report
All 6 combinations passed the leakage audit with 100% compliance:
- **`no_train_val_overlap`**: PASS
- **`no_train_test_overlap`**: PASS
- **`no_val_test_overlap`**: PASS
- **`train_labels_before_val_period`**: PASS (all training sequences future label window ends before `val_cutoff`)
- **`val_labels_before_test_period`**: PASS
- **`embargo_train_val_boundary`**: PASS (no examples exist within $max(C, H)$ bars of the train/val split boundary)
- **`embargo_val_test_boundary`**: PASS
- **`quantile_thresholds_train_only`**: PASS (thresholds calculated using ONLY the training period raw dataset and then fixed for inference)

---

## Sample Example Prompt/Response Format
Below is a sample sequence from the dataset:
```text
Q: BTCUSDT C=60 H=15 rv=0.03 ret=-0.08 abs=0.48 range=0.32 volchg=-0.05 last=[-0.01%,+0.00%,-0.00%,-0.00%,-0.00%]
A: LOW_VOL
```

---

## Recommended First Training Dataset
We recommend **`vol_regime_H15_C60`** as the starting point for EXP012 training:
1. **Large Sample Size:** It offers 882,576 training examples and 83,856 test examples.
2. **Compact Length:** The average sequence length is 123.5 bytes, which fits easily into the 128-byte training sequence length.
3. **Clean Baseline:** Class distribution is exactly 50/50, and it passed all leakage audits.
4. **Strong Autocorrelation:** The short H=15 volatility is strongly clustered with the C=60 context, offering the cleanest signal-to-noise ratio.

---

## Honesty Disclaimers
- **No Trading Signal Claims:** These volatility labels represent mathematical realized volatility thresholds. They are NOT buy/sell/hold trading signals.
- **No Backtesting or Profitability Evaluation:** No backtests were run, and no profitability claims are made.
- **Dataset Only:** This experiment contains only dataset preparation and validation. No model training was performed.
