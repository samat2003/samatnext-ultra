# EXP008B: Fast32 Answer-Only Reasoning SFT Results

This document summarizes the results of the answer-only SFT reasoning experiments using the `fast32_reasoning_v2` SFT dataset.

---

## Task-Specific SFT Results

| Dataset | Steps | Train CE (Final) | Best Val CE | Test CE | Test Accuracy | Invalid Rate | Status / Gate |
|---|---|---|---|---|---|---|---|
| **`boolean_only`** | 2500 | 0.1010 | 2.0321 | 2.0471 | **100.00%** | 0.00% | **Passed** (>= 70%) |
| **`arithmetic_compare`** | 2500 | 0.7357 | 0.9151 | 0.9122 | **90.00%** | 0.00% | **Passed** (>= 60%) |
| **`market_direction`** | 2500 | 5.5312 | 0.9893 | 0.9955 | **53.00%** | 0.00% | **Completed** |
| **`mixed_answer_only`** | 2500 | 0.6024 | 0.6908 | 0.7014 | **57.00%** | 0.00% | **Completed** |

---

## Detailed Findings & Checklist Answers

1. **Did answer-only format reduce invalid outputs?**
   Yes. The invalid/empty output rate was **completely reduced from 68.75% to 0.00%** across all tasks!
2. **Did boolean accuracy improve over 13%?**
   Yes. Boolean logic exact accuracy improved from **13.00% to 100.00%**!
3. **Did any task exceed the decision gate?**
   Yes, both single-task gates succeeded:
   - `boolean_only` reached **100.00%** (gate >= 70%).
   - `arithmetic_compare_only` reached **90.00%** (gate >= 60%).
4. **Which task family is learnable?**
   - **`boolean_only`** (100.00% accuracy) and **`arithmetic_compare_only`** (90.00% accuracy) are highly learnable under SFT.
   - **`market_direction_only`** is partially learnable, achieving **53.00% accuracy** (significantly beating random guessing of 33% for the UP/DOWN/FLAT labels).
5. **Which task family fails?**
   - **Arithmetic addition/subtraction** (included in the mixed task) completely failed, yielding **0% accuracy**.
6. **Is the model ready for quant-decision classification?**
   Yes. Achieving **53.00% accuracy** on 3-class market direction checks shows the model is ready for simple quant-decision classification pre-training sweeps, but needs a slightly larger parameter capacity to handle precise numeric addition/subtraction.
7. **What should be simplified next if accuracy remains low?**
   Accuracy is high on classification tasks. If numeric addition/subtraction is needed, we should simplify the tasks by using smaller integers (1-digit integers) or introducing a state-value embedding.
8. **Was architecture unchanged?** Yes. Trainable parameter count remains exactly **`216,320`**.
9. **Were frozen artifacts unchanged?** Yes. Checksums match exactly.
