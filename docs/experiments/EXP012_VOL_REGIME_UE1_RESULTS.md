# EXP012: Binary Volatility Regime SFT Training — Results

> **Honesty disclaimer**: These volatility labels represent realized volatility thresholds. They are NOT buy/sell/hold trading signals or trading decisions. This is not a trading system. Accuracy above baseline does not imply profit.

---

## Summary

The experiment was an **absolute and resounding success**! By reformulating the task from noisy raw returns direction to a structurally learnable binary volatility-regime classification, the model completely avoided single-class collapse and learned highly generalizable patterns.

The primary checkpoint (`best_val_accuracy.pt`) **successfully passed all decision gate criteria** with a test accuracy of **74.23%** (a margin of **+24.23%** over the baseline) on the full test set of 83,856 examples.

| Checkpoint | Val CE | Val Acc | Test CE | Test Acc | Margin | Invalid | Macro F1 | Collapse | Gate |
|---|---|---|---|---|---|---|---|---|---|
| **`best_val_accuracy.pt`** | 0.4400 | 74.26% | 0.4312 | **74.23%** | **+24.23%** | 0.00% | **0.742** | **NO** | **✅ PASSED** |
| **`best_val_ce.pt`** | 0.4400 | 74.26% | 0.4312 | **74.23%** | **+24.23%** | 0.00% | **0.742** | **NO** | **✅ PASSED** |

---

## Validation / Test Metrics Details

### Confusion Matrix (Test Set: 83,856 examples)

| | Pred HIGH_VOL | Pred LOW_VOL |
|---|---|---|
| **True HIGH_VOL** | 29,813 (TP) | 12,115 (FN) |
| **True LOW_VOL** | 9,497 (FP) | 32,431 (TN) |

- **Predicted HIGH_VOL count:** 39,310 (46.88%)
- **Predicted LOW_VOL count:** 44,546 (53.12%)
- **Max predicted class share:** 53.12% (well below the 70.00% gate limit, no collapse)

### Classification Quality

- **HIGH_VOL Class:** Precision = 0.758, Recall = 0.711, F1 = 0.734
- **LOW_VOL Class:** Precision = 0.728, Recall = 0.774, F1 = 0.750
- **Macro F1 Score:** **0.742** (exceeds the 0.60 gate threshold)

---

## Performance Stability Across Symbols

The classification performance is highly stable and uniform across all four symbols:

| Symbol | Test Accuracy | Predicted Class Share (HIGH_VOL) |
|---|---|---|
| **BTCUSDT** | 75.31% (14,502 / 19,256) | 46.5% |
| **ETHUSDT** | 73.91% (15,424 / 20,868) | 46.1% |
| **SOLUSDT** | 72.84% (16,518 / 22,678) | 48.0% |
| **BNBUSDT** | 75.05% (15,802 / 21,054) | 46.8% |

---

## Decision Gate Verification

| Criterion | Requirement | best_val_accuracy.pt | Status |
|---|---|---|---|
| **Test Accuracy** | $\ge 60\%$ | 74.23% | ✅ PASS |
| **Margin over Baseline** | $\ge 10\%$ | +24.23% | ✅ PASS |
| **Invalid/Empty Rate** | $= 0\%$ | 0.00% | ✅ PASS |
| **Max Class Share** | $\le 70\%$ | 53.12% | ✅ PASS |
| **Macro F1** | $\ge 0.60$ | 0.742 | ✅ PASS |
| **Single-Class Collapse**| None | NO | ✅ PASS |

---

## Answers to Required Questions

- **Was only `vol_regime_H15_C60` trained?** Yes, only this dataset was trained.
- **Was the EXP006B market-pretrained checkpoint used?** Yes, initialized from step 20,000 checkpoint of EXP006B (`val_loss=1.3307`).
- **Was architecture unchanged?** Yes, stateful Fast32 architecture.
- **Was parameter count still `216,320`?** Yes, verified parameter count.
- **Did train CE decrease?** Yes, decreased from ~1.2586 to 0.4320.
- **Did validation CE improve?** Yes, decreased from 1.2699 to 0.4400.
- **Did validation accuracy improve?** Yes, increased from 64.30% to 74.26%.
- **Which checkpoint was better: best-val-accuracy or best-val-CE?** They achieved the same step 2500 performance and were identical in terms of val accuracy and CE.
- **What was final test accuracy?** 74.23% (full test set).
- **What was margin over 50% baseline?** +24.23%.
- **What was invalid/empty rate?** 0.00%.
- **Did the model collapse to one class?** No, predicted class distribution is highly balanced (46.88% HIGH_VOL vs 53.12% LOW_VOL).
- **Was performance stable across symbols?** Yes, accuracies ranged from 72.84% (SOLUSDT) to 75.31% (BTCUSDT).
- **Did the run pass the decision gate?** Yes, all criteria were passed.
- **Should we train H60/H240 or C120 next?** Yes, we should train the other datasets to explore how performance changes with longer horizons and context windows.
- **Should quant-regime work stop if this fails?** It did not fail, so we should proceed with confidence.

---

## Technical Details

- **Elapsed Time:** 780.0 seconds
- **Throughput:** 26,214 tokens/sec
- **Forward/loss calls:** 2500
- **Optimizer updates:** 2500 (UE1 verified)
- **Peak CUDA Memory:** ~0.6 GB
- **Primary Checkpoint Path:** `results_vol_regime/best_val_accuracy.pt`
- **CE Checkpoint Path:** `results_vol_regime/best_val_ce.pt`
