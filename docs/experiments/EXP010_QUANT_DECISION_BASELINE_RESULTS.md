# EXP010: Fast32 Quant Decision Classification UE1 Baseline Results

This document summarizes the results of the Quant Decision SFT UE1 Baseline run of the Fast32 model.

---

## Leakage and Embargo Audit Report

An automated chronological embargo audit was performed prior to SFT training:

- **`context_window`:** 32 bars (minutes)
- **`max_horizon`:** 60 bars (minutes)
- **`embargo` size:** 60 minutes (`3,600,000` ms)
- **Train/Val Boundary:** `1752952320000`
- **Val/Test Boundary:**   `1753479360000`
- **Examples Removed by Embargo:**
  - Train: 40 examples
  - Validation: 35 examples
- **Leakage Verifications:**
  - **Passed.** Programmatic checks confirmed that no training prediction window reaches into the validation period (all train example timestamp + horizon_ms <= `1752952320000`).
  - **Passed.** Programmatic checks confirmed that no validation prediction window reaches into the test period (all validation example timestamp + horizon_ms <= `1753479360000`).

---

## SFT UE1 Training Results

| Metric | Value |
|---|---|
| **Audited Dataset** | `fast32_quant_decision_v1_audited` |
| **Excluded Task** | `trend_range` (due to extreme class imbalance of 99.45% majority baseline) |
| **Steps** | 2500 |
| **Batch Size** | 64 |
| **Sequence Length** | 128 |
| **AMP Precision** | **bf16** |
| **First Train CE** | 5.9748 |
| **Final Train CE** | 0.2585 |
| **Best Validation CE** | **0.3566** (at Step 1750) |
| **Final Test CE (Best Checkpoint)** | **0.3421** |
| **Peak CUDA Memory** | 0.96 GB |
| **Throughput** | 34,429 tokens/sec |
| **Elapsed Training Time** | 594.8 seconds (9.9 minutes) |
| **Forward/Loss Calls** | 2500 |
| **Optimizer Updates** | 2500 |

---

## Exact Answer Accuracy (Test Set Breakdown)

Evaluation was performed using answer-only prompts (`Q: <task>\nA:`) on 100 test samples per task:

- **Overall Test Accuracy:** **44.00%**
- **Invalid / Empty Answer Rate:** **0.00%**

### Performance by Task Category:

| Task Family | Labels | Test Accuracy | Majority Baseline | Difference | Status |
|---|---|---|---|---|---|
| **`binary_dir`** | `UP`, `DOWN` | **52.00%** | 50.66% | **+1.34%** | **Passed** |
| **`three_class_dir`** | `UP`, `DOWN`, `FLAT` | **42.00%** | 38.15% | **+3.85%** | **Passed** |
| **`vol_regime`** | `VOL_UP`, `VOL_DOWN`, `VOL_FLAT` | 27.00% | 43.95% | -16.95% | Failed |
| **`breakout`** | `BREAKOUT`, `NO_BREAKOUT` | 55.00% | 55.54% | -0.54% | Failed |

---

## Detailed Findings & Checklist Answers

1. **Did leakage/embargo audit pass?** Yes, audited successfully and all constraints were met.
2. **Was architecture unchanged?** Yes. Trainable parameter count remains exactly **`216,320`**.
3. **Were frozen artifacts unchanged?** Yes. SHA256 checksums match.
4. **Was `trend_range` excluded?** Yes.
5. **Did train CE decrease?** Yes, from 5.9748 to 0.2585.
6. **Did validation CE improve?** Yes, from 6.4035 (step 250) to 0.3566 (step 1750).
7. **What was final test CE?** **0.3421**.
8. **What was overall test accuracy?** **44.00%**.
9. **Which tasks beat majority baseline?**
   - **`binary_dir`** (+1.34% improvement)
   - **`three_class_dir`** (+3.85% improvement)
10. **Which tasks failed to beat majority baseline?**
    - **`vol_regime`** (-16.95% below baseline)
    - **`breakout`** (-0.54% below baseline)
11. **What was invalid/empty rate?** **0.00%**.
12. **Is the model ready for UE32 quant-decision speed sweeps?**
    Yes. The decision gate requires beating the baseline for at least two task families. Since both `binary_dir` and `three_class_dir` beat their baselines, the gate is passed and the model is ready.
13. **What should be changed if accuracy is near baseline?**
    To improve accuracy on harder classification tasks (like volatility regime and breakouts), we should extend the context window (e.g. from 32 to 64 bars) to capture longer-term trends or increase the parameter capacity of the model.

---

## Honesty & Disclaimers
- **No Trading Decision Signals:** The classification labels represent programmatic return thresholds, NOT live buy/sell decisions.
- **No Profit Evaluation:** No trading simulations or profit metrics were evaluated.
- **No Claims of Profitability:** Exceeding the majority-class baseline on direction classification does NOT imply or guarantee trading profitability.
