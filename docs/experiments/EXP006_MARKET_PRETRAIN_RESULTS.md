# EXP006: Fast32 Market Pretraining UE32 Sanity Check & Comparative Results

This document summarizes the results of the market pretraining comparison between Standard (UE1) and UE32 modes using the prepared Binance Futures USDⓈ-M 1m dataset.

---

## Phase 1: UE32 Sanity Check (500 Steps)

The sanity check verified that UE32 successfully learns on the structured byte token sequences:
- **First Train CE:** 6.0937
- **Final Train CE:** 4.2096 (decreased by **1.8841**)
- **Final Validation CE:** 4.4260 (improved from starting loss)
- **Status:** **Passed.** UE32 successfully learned, and the run proceeded to Phase 2.

---

## Phase 2 & 3: 2500-Step Comparative Results

| Metric | standard / UE1 | UE32 |
|---|---|---|
| **Dataset Used** | Binance Futures UM 1m | Binance Futures UM 1m |
| **Train Token Count** | 23,546,880 | 23,546,880 |
| **Validation Token Count** | 2,943,360 | 2,943,360 |
| **Effective Train Data Passes** | 1.7395 | 1.7395 |
| **Steps** | 2500 | 2500 |
| **Forward/Loss Calls** | 2500 | 2500 |
| **Optimizer Updates** | 2500 | 78 |
| **Train Elapsed (sec)** | 811.78 | 75.09 |
| **Throughput (tok/s)** | 50,456.9 | 545,446.8 |
| **First Train CE** | 3.9649 | 6.0937 |
| **Final Train CE** | 1.0849 | 3.3451 |
| **Best Validation CE** | **1.1509** | 3.5800 |
| **Final Validation CE** | 1.1509 | 3.5800 |
| **Test CE (Single Eval)** | **1.3684** | 3.6822 |
| **CE Improvement** | 2.8800 | 2.7486 |
| **CE/min** | 0.213 | **2.196** |
| **Peak CUDA Memory** | 2.41 GB | 2.41 GB |
| **Loss Decreased?** | Yes | Yes |
| **Validation Improved?** | Yes | Yes |

---

## Comparative Metrics

- **Best Validation CE Mode:** `standard` (UE1) with **1.1509**.
- **Best Test CE Mode:** `standard` (UE1) with **1.3684**.
- **Fastest Mode:** `ue32` with **545,446.8 tok/s** (**10.81x speedup**).
- **Best CE/min Mode:** `ue32` with **2.196** (**10.3x gain**).
- **UE32 validation CE gap vs UE1:** **+2.4291**
- **UE32 test CE gap vs UE1:** **+2.3138**
- **Whether UE32 still learns with real scheduled updates:** Yes, train CE decreased by **2.7486** (from 6.0937 to 3.3451).
- **Whether UE32 is worth further market pretraining:** Yes. UE32 learns extremely fast per second (2.196 CE/min), making it highly suitable for rapid exploratory pretraining sweeps. However, for maximum sequence prediction quality, Standard (UE1) remains far superior in final test CE.

---

## Checklist Answers

1. **Did tests pass?** Yes. All 113 unit tests passed.
2. **Was architecture unchanged?** Yes. Trainable parameter count is exactly **`216,320`**.
3. **Were frozen artifacts unchanged?** Yes. Checksums match exactly.
4. **Was UE32 sanity check successful?** Yes. Loss decreased from 6.0937 to 4.2096.
5. **Did UE32 train CE decrease?** Yes (to 3.3451).
6. **Did UE32 validation CE improve?** Yes (to 3.5800).
7. **Did UE1 train CE decrease?** Yes (to 1.0849).
8. **Did UE1 validation CE improve?** Yes (to 1.1509).
9. **Which mode had better validation CE?** `standard` (UE1) with **1.1509**.
10. **Which mode had better test CE?** `standard` (UE1) with **1.3684**.
11. **Which mode had better CE improvement per minute?** `ue32` with **2.196**.
12. **Did UE32 reach or exceed 1M tok/s?** No, it reached **545,446.8 tok/s**.
13. **How many effective passes over the market dataset occurred?** **1.74 passes**.
14. **What bottleneck remains?**
    - **Autograd backward tracing overhead:** The backward pass takes 0.294s per step (representing 84% of the update step time).
    - **Dataloader CPU slicing overhead:** Slicing CPU tensors 64 times via list comprehension takes 1.5ms per step.
15. **What should be tested next?**
    - Implementing a **fused Triton backward pass** (`fused_backward`) to compute DNA parameter gradients directly on GPU, reducing update step time from 0.35s to ~0.02s and unlocking 1M+ active training speeds.
