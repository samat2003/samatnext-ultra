# EXP008: Fast32 UE1 Reasoning Fine-Tuning Results

This document summarizes the results of the supervised reasoning fine-tuning (SFT) run of the Fast32 model under Standard (UE1) mode.

---

## SFT Reasoning Training Results

| Metric | Value |
|---|---|
| **Dataset Used** | Fast32 Reasoning v1 (`fast32_reasoning_v1`) |
| **Train Examples** | 100,000 |
| **Validation Examples** | 10,000 |
| **Test Examples** | 10,000 |
| **Steps** | 2500 |
| **Batch Size** | 32 |
| **Sequence Length** | 512 (max example length: 522) |
| **AMP Precision** | **bf16** (changed from fp16 to prevent SSM overflow on 512-length sequences) |
| **First Train CE** | 3.3373 |
| **Final Train CE** | 1.5990 (decreased by **1.7383**) |
| **Best Validation CE** | **1.6458** (at Step 2500) |
| **Final Validation CE** | 1.6458 |
| **Final Test CE (Single Eval)** | **1.6484** |
| **Peak CUDA Memory** | 5.92 GB |
| **Loss Decreased?** | Yes |
| **Validation Improved?** | Yes |

---

## Exact Answer Accuracy (Test Set Breakdown)

Evaluation was performed using answer-only prompts on 100 random test samples per category (400 samples total):

- **Overall Exact Accuracy:** **3.25%**
- **Arithmetic Accuracy:** **0.00%**
- **Symbolic Reasoning Accuracy:** **0.00%**
- **Boolean Logic Accuracy:** **13.00%**
- **Market-bar Reasoning Accuracy:** **0.00%**
- **Invalid / Empty Answer Rate:** **68.75%**

### Easiest Category
- **Boolean Logic** was the easiest category with **13.00% accuracy**. The model successfully learned to answer Modus Ponens rules (e.g. concluding "Yes" when a rule antecedence was satisfied).

### Hardest Categories
- **Arithmetic, Symbolic Reasoning, and Market-bar Reasoning** were the hardest categories, all yielding **0.00% exact accuracy**.

---

## Detailed Findings & Checklist Answers

1. **Did train CE decrease?** Yes, from 3.3373 to 1.5990.
2. **Did validation CE improve?** Yes, from 2.4849 (at step 250) to 1.6458.
3. **What was test CE?** **1.6484**.
4. **What was exact answer accuracy?** **3.25%** overall (13.00% on boolean logic).
5. **Which category was easiest?** Boolean Logic.
6. **Which category was hardest?** Arithmetic, Symbolic Reasoning, and Market-bar Reasoning.
7. **Did the model learn format only, or actual answers?**
   The model primarily learned the structural SFT format and vocabulary. It learned to format text as `### Task: ... ### Answer:` but failed to generate correct answers for complex tasks (with the exception of simple rule-based boolean questions). The high invalid/empty output rate (68.75%) suggests the 216,320 parameter capacity is insufficient to model step-by-step reasoning paths at length 512.
8. **Was architecture unchanged?** Yes. Trainable parameter count is exactly **`216,320`**.
9. **Were frozen artifacts unchanged?** Yes. SHA256 checksums match exactly.
10. **Numerical Stability (bf16 Switch):**
    Training in FP16 on 512-length sequences was numerically unstable due to decay accumulation overflow, triggering NaN loss at steps 1243 and 1494. Switching to **BF16 precision** with **gradient clipping (max_norm=1.0)** resolved all instability, enabling training to complete the full 2500 steps successfully.
11. **Honesty & Disclaimers:**
    - Low exact accuracy (3.25%) shows the model does NOT possess general reasoning capability.
    - Market reasoning tasks are purely programmatic checks of sequence comparisons, NOT trading signals. Lower CE does not imply profitable trading capability.
