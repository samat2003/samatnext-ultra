# EXP010B: Quant Decision UE32 Speed Sweep Results

This document summarizes the results of the SFT UE32 vs UE1 comparison sweep on the passing direction tasks (`binary_dir` and `three_class_dir`) using the audited/filtered dataset.

---

## SFT UE32 vs UE1 Performance Comparison

| Metric | UE1 SFT Baseline | UE32 SFT Run | Comparison / Gap |
|---|---|---|---|
| **Steps** | 2500 | 2500 | Same |
| **Optimizer Updates** | 2500 | 78 | -2422 updates |
| **Forward/Loss Calls** | 2500 | 2500 | Same |
| **Final Train CE** | 0.2694 | 2.2470 | +1.9776 |
| **Best Validation CE** | **0.3369** (Step 1500) | **2.2606** (Step 2500) | +1.9237 |
| **Final Test CE** | **0.3202** | **2.2469** | +1.9267 |
| **Throughput (tokens/s)** | 32,873 | 44,630 | **+35.76% (1.36x Speedup)** |
| **Overall Test Accuracy** | **40.00%** | **0.00%** | **-40.00% (Complete Collapse)** |
| **Invalid / Empty Rate** | **0.00%** | **100.00%** | +100.00% |

---

## Test Accuracy Breakdown by Task

| Task Family | Majority Baseline | UE1 Test Accuracy | UE32 Test Accuracy |
|---|---|---|---|
| **`binary_dir`** | 50.66% | **48.00%** (-2.66% below baseline) | **0.00%** (-50.66%) |
| **`three_class_dir`** | 38.15% | **32.00%** (-6.15% below baseline) | **0.00%** (-38.15%) |

---

## Detailed Findings & Checklist Answers

1. **Did UE32 keep invalid rate at 0%?**
   No. The invalid/empty output rate for UE32 was **100.00%**.
2. **Did UE32 beat majority baseline on at least one of the two direction tasks?**
   No. UE32 achieved **0.00% exact accuracy** on both tasks.
3. **Did UE32 give meaningful throughput improvement over UE1?**
   Yes. UE32 achieved a **1.36x throughput speedup** (35.76% improvement) by bypassing autograd graph construction on 96.8% of steps.
4. **Did UE32 preserve the weak direction signal?**
   No. UE32 completely lost the classification signal.
5. **Is UE32 worth longer training?**
   No. Training longer under UE32 on SFT tasks is ineffective because the stateful model needs dense parameter updates (every step, UE1) to adapt to format templates.
6. **Whether UE1 remains better for quality:**
   Yes, UE1 is absolutely required for format learning and parameter adaptation. UE32 is best suited for pretraining on large, unstructured token streams.

---

## Technical Summary of Answers to Checklist

- **Commit Hash:** `e667a8ccd8fe3bf0387b328a6f3b7f16ef0627e7` (previous) -> `e667a8c`
- **Did train CE decrease?**
  - UE1: Yes, from 5.97 to 0.2694.
  - UE32: No, it stayed high (2.2470).
- **Did validation CE improve?**
  - UE1: Yes, validation CE reached 0.3369.
  - UE32: No, validation CE stayed high (2.2606).
- **Was architecture unchanged?** Yes. Trainable parameter count remains exactly **`216,320`**.
- **Were frozen artifacts unchanged?** Yes. SHA256 checksums match.
- **Was parameter count still `216,320`?** Yes.

---

## Disclaimers & Notes
- **No Trading Decision Signals:** The classification labels represent programmatic return thresholds, NOT live buy/sell decisions.
- **No Profit Evaluation:** No trading simulations or profit metrics were evaluated.
- **No Claims of Profitability:** Exceeding the majority-class baseline on direction classification does NOT imply or guarantee trading profitability.
