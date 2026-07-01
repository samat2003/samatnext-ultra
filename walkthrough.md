# Walkthrough: REMOVED Final Benchmark and Stability Audit

I have successfully performed the final benchmark and stability audit for the best REMOVED volatility-regime checkpoint (`best_val_accuracy.pt`), created the benchmark runner, and wrote and committed the results report to GitHub.

## Deployed Files
- **`scripts/benchmark_final_vol_regime.py`**: Deployed final benchmark and stability runner containing:
  - Checkpoint integrity audits (SHA256, Git hash, parameters, config).
  - Reproducibility evaluation across 3 seeds on the full test set.
  - Multi-batch inference latency timer (warmup + CUDA synchronization).
  - Jitter ratio checks.
  - Stability verification (NaN/inf checks).
  - Final metadata JSON generator.
- **`REMOVED/REMOVED_FINAL_BENCHMARK_AUDIT.md`**: Detailed final audit report.
- **`results_vol_regime/final_checkpoint_metadata.json`**: Frozen checkpoint metadata.

---

## 1. Checkpoint Integrity
- **Path:** `results_vol_regime/best_val_accuracy.pt`
- **SHA256:** `b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4`
- **Source commit:** `981b77e4a2a278603ab3598bc8495105feae7d87`
- **Parameter count:** `216,320` (verified matching original stateful Fast32 architecture)
- **Dataset:** `vol_regime_H15_C60` (test split: 83,856 examples)

---

## 2. Accuracy Reproducibility (3 Runs)
- **Run 1 (seed=0):** 71.38% test accuracy | Macro F1 = 0.706 | Invalid rate = 0.00%
- **Run 2 (seed=1):** 71.38% test accuracy | Macro F1 = 0.706 | Invalid rate = 0.00%
- **Run 3 (seed=2):** 71.38% test accuracy | Macro F1 = 0.706 | Invalid rate = 0.00%
- **Determinism:** Yes, all three runs produced exactly identical predictions.
- **Divergence from REMOVED:** The saved checkpoint yields exactly 71.38% test accuracy. The reference 74.23% in some earlier notes was based on an unpadded sequential run, whereas the actual saved checkpoint `best_val_accuracy.pt` achieves exactly 71.3831% test accuracy under both batched and sequential evaluations. This is consistent with `combined_results.json` from the training run.

---

## 3. Latency & Jitter Profiles (RTX 5070 Ti mobile)
- **Batch 1:** Mean = 0.449 ms, p99 = 0.490 ms | 2,229 ex/sec | Jitter: Stable ✅
- **Batch 16:** Mean = 0.505 ms, p99 = 0.596 ms | 31,692 ex/sec | Jitter: Stable ✅
- **Batch 64:** Mean = 0.767 ms, p99 = 0.865 ms | 83,401 ex/sec | Jitter: Stable ✅
- **Batch 256:** Mean = 2.146 ms, p99 = 2.302 ms | 119,307 ex/sec | Jitter: Stable ✅

---

## 4. Stability Indicators
- **NaN / Inf loss or logits:** None detected.
- **Single-class collapse:** None (predicted distribution is 33.60% HIGH_VOL / 66.40% LOW_VOL).
- **Invalid predictions:** 0.00%.

---

## Final Freeze Decision
**✅ APPROVED / FROZEN.** All stability, latency, and reproducibility gates are passed.

---

## REMOVED Speed Benchmark Definition Audit

This audit resolved the apparent difference between the **`13.75 µs`** token step latency and the **`449 µs`** prompt classification latency.

### 1. Key Finding: No Slowdown Occurred
The trained volatility checkpoint (`best_val_accuracy.pt`) runs on the exact same stateful Fast32 architecture. On the identical single-token CUDA-graph Triton path, the trained model runs in **`13.82 µs`**, matching the baseline base model (**`14.87 µs`**).

### 2. Definition Discrepancy Explained
- **13.75 µs**: Measures **single-token cached/fused inference step latency** using custom Triton kernels + CUDA Graph capture (`seq_len=1`, `batch=1`).
- **449 µs**: Measures **full-prompt sequence forward pass latency** over **`seq_len=128`** through the higher-level Python validation pipeline (CPU-GPU transfers, prompt text bytes encoding, answer decoding).

### 3. Apples-to-Apples Latency Metrics (RTX 5070 Ti Laptop GPU)

- **Pure Model Forward (`seq_len=128`, batch=1)**: **365.8 µs per call** (equivalent to **2.86 µs per token**).
- **Full Classifier Pipeline (batch=1)**: **442.8 µs per example** (includes tokenization, padding, execution, and decoding).
- **Trained Model Stateful Token Step (Triton + Graph)**: **13.82 µs per token**.
- **Trained Model Stateless Fused Speed Ablation (Triton + Graph)**: **7.37 µs per token**.

The audit report is saved at [`REMOVED/REMOVED_SPEED_BENCHMARK_DEFINITION_AUDIT.md`](file:///C:/Users/Samat%20Zharassov/.gemini/antigravity/brain/8488c71b-83fc-4776-85ba-e0daa5fd6d83/scratch/REMOVED_SPEED_BENCHMARK_DEFINITION_AUDIT.md).

