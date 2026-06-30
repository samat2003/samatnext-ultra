# EXP006B: Fast32 Long UE32 Market Pretraining Results

This document summarizes the results of the long 20,000-step UE32 pretraining run using the prepared Binance Futures USDⓈ-M 1m pretraining dataset.

---

## 20,000-Step Pretraining Results

| Metric | Value |
|---|---|
| **Dataset Used** | Binance Futures UM 1m |
| **Train Token Count** | 23,546,880 |
| **Validation Token Count** | 2,943,360 |
| **Effective Train Data Passes** | **13.916** |
| **Steps Run** | 20,000 |
| **Forward/Loss Calls** | 20,000 |
| **Optimizer Updates** | 625 |
| **Train Elapsed (sec)** | 627.96 (10.47 minutes) |
| **Throughput (tok/s)** | **521,817.8** |
| **First Train CE** | 6.1712 |
| **Final Train CE** | 1.2532 |
| **Best Validation CE** | **1.3307** (at Step 20,000) |
| **Final Validation CE** | 1.3307 |
| **Total CE Improvement** | **4.9180** (Train) / **4.8405** (Val) |
| **CE/min** | **0.470** |
| **Peak CUDA Memory** | 2.41 GB |
| **Loss Decreased?** | Yes |
| **Validation Improved?** | Yes |
| **Early Stopping Triggered?** | No (Full run completed successfully) |

---

## Chronological Progress (Every 1,000 Steps)

| Step | Train CE | Val CE | CE/min |
|---|---|---|---|
| 0 | 6.1712 | - | - |
| 1,000 | 3.5987 | 3.8810 | 4.827 |
| 2,000 | 3.4204 | 3.6688 | 2.589 |
| 3,000 | 3.0058 | 3.2574 | 2.035 |
| 4,000 | 2.5511 | 2.8389 | 1.765 |
| 5,000 | 2.1929 | 2.4589 | 1.567 |
| 6,000 | 1.9846 | 2.2566 | 1.360 |
| 7,000 | 1.8702 | 2.0645 | 1.193 |
| 8,000 | 1.7495 | 1.9181 | 1.069 |
| 9,000 | 1.6569 | 1.8416 | 0.971 |
| 10,000 | 1.5745 | 1.7610 | 0.890 |
| 11,000 | 1.5359 | 1.6872 | 0.819 |
| 12,000 | 1.5264 | 1.6425 | 0.747 |
| 13,000 | 1.4307 | 1.5759 | 0.700 |
| 14,000 | 1.3695 | 1.5044 | 0.656 |
| 15,000 | 1.3513 | 1.4611 | 0.616 |
| 16,000 | 1.2939 | 1.4164 | 0.585 |
| 17,000 | 1.2785 | 1.3772 | 0.552 |
| 18,000 | 1.2576 | 1.3605 | 0.523 |
| 19,000 | 1.2708 | 1.3462 | 0.492 |
| 20,000 | 1.2532 | 1.3307 | 0.470 |

---

## Technical Summary & Analysis

1. **Learning Curve Stability:** 
   The model exhibits highly stable learning throughout the entire 20,000 steps under UE32 scheduled updates. Train CE decreased from 6.1712 to 1.2532, and Validation CE decreased from 3.8809 (at step 1000) to 1.3307.
2. **Overfitting Analysis:**
   No overfitting or divergence is observed. The final validation loss (**1.3307**) matches the best validation loss, and the generalization gap (Val CE - Train CE) remains extremely small at **0.0775**, even after 13.9 effective dataset passes.
3. **Validation CE Gap vs UE1:**
   While UE32 demonstrates stable learning and a high throughput of **521,818 tok/s**, a significant quality gap remains compared to standard training (UE1). In the 2,500-step pretraining run, Standard mode reached a validation CE of **1.1509** in only 14.1 minutes. UE32 requires substantially more steps to approach this loss level (reaching 1.3307 at 20,000 steps).
4. **Recommendation for Next Steps:**
   To bridge this gap while preserving speed, we recommend exploring:
   - **UE16 updates** to double the frequency of parameter updates (trading some throughput for faster CE reduction).
   - A **hybrid recovery phase** where the model is pre-trained with UE32 for 20,000 steps, followed by a short UE1 (standard) recovery phase to fine-tune and bridge the validation CE gap.
