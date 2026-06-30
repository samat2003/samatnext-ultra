# EXP004D: UE1 vs UE32 on Real Wikipedia-Scale Data

This document summarizes the results of the 2500-step training comparison between Standard (UE1) and UE32 modes on real English Wikipedia data.

## Comparative Results Table (2500 Steps)

| Metric | standard / UE1 | UE32 |
|---|---|---|
| **Dataset Used** | `wikimedia/wikipedia (20231101.en)` | `wikimedia/wikipedia (20231101.en)` |
| **Train Token Count** | 40,503,074 | 40,503,074 |
| **Validation Token Count** | 4,500,342 | 4,500,342 |
| **Effective Train Data Passes** | 1.0113 | 1.0113 |
| **Steps** | 2500 | 2500 |
| **Forward/Loss Calls** | 2500 | 2500 |
| **Optimizer Updates** | 2500 | 78 |
| **Train Elapsed (sec)** | 847.41 | 76.16 |
| **Throughput (tok/s)** | 48,335.4 | 537,810.6 |
| **First Train CE** | 3.4127 | 5.9433 |
| **Final Train CE** | 2.5148 | 3.0875 |
| **Best Validation CE** | **2.5198** | 3.0857 |
| **Final Validation CE** | 2.5198 | 3.0857 |
| **CE Improvement** | 0.8979 | 2.8558 |
| **CE/min** | 0.0636 | **2.250** |
| **Peak CUDA Memory** | 2.41 GB | 2.41 GB |
| **Loss Decreased?** | Yes | Yes |
| **Validation Improved?** | Yes | Yes |

---

## Tiny Shakespeare Comparison (EXP004B)

* **Tiny Shakespeare UE1:** 45,685 tok/s, best val CE **2.4287**, CE/min **0.155**
* **Tiny Shakespeare UE32:** 517,722.8 tok/s, best val CE **3.1286**, CE/min **5.298**

**Analysis of the Tradeoff Change:**
On larger Wikipedia-scale text, the overall behavior remains highly consistent with Tiny Shakespeare. 
- **Standard (UE1)** achieves the best final quality (validation CE **2.5198** vs UE32's **3.0857**).
- **UE32** achieves the best wall-clock efficiency, outperforming standard mode by **35.4x on CE/min** (**2.250** vs **0.0636**) and delivering an **11.1x speedup** on raw throughput.
- This confirms that the scheduled-update cadence holds on real Wikipedia-scale text: UE32 learns extremely fast per second but trades off some validation CE quality compared to full chain-rule standard updates.

---

## Checklist Answers

1. **Did tests pass?** Yes. All 113 unit tests passed successfully.
2. **Was architecture unchanged?** Yes. Trainable parameter count remains exactly **`216,320`**.
3. **Were frozen artifacts unchanged?** Yes. SHA256 checksums match exactly.
4. **Which dataset/config was used?** `wikimedia/wikipedia` (English config `20231101.en`) loaded via Hugging Face datasets streaming.
5. **Was Tiny Shakespeare avoided?** Yes.
6. **How many real text tokens were used?** `45,003,416` total tokens (`40,503,074` train / `4,500,342` val).
7. **How many effective passes over data occurred?** **1.01 passes** (almost exactly 1 epoch).
8. **Did UE1 loss decrease?** Yes (from 3.4127 to 2.5148).
9. **Did UE32 loss decrease?** Yes (from 5.9433 to 3.0875).
10. **Which mode had better validation CE?** `standard` (UE1) with **2.5198**.
11. **Which mode had better CE improvement per minute?** `ue32` with **2.250**.
12. **Did UE32 reach or exceed 1M tok/s?** No, it reached **537,810.6 tok/s**.
    - *Reason:* Slicing/dataloader CPU overhead on the streaming data loader limits the non-update step throughput to ~819K tok/s. Coupled with the 78 active update steps (taking 0.35s each), active training throughput is capped at 537K tok/s.
13. **What bottleneck remains?**
    - **Autograd tracing overhead:** The backward pass takes 0.298s per step (representing 83% of the update step time).
    - **Dataloader CPU slicing overhead:** Slicing CPU tensors 64 times via list comprehension takes 1.5ms per step.
14. **What should be tested next?**
    - A **fused Triton backward pass** (`fused_backward`) that computes parameter gradients directly on the GPU, bypassing PyTorch Autograd and sequence-loop overhead entirely. This would drop update step time from 0.35s to ~0.02s, unlocking 1M+ active UE32 training speed!
