# Fast32: Final Trained Volatility-Regime Results

## 1. Project Summary
Fast32 is a 216,320-parameter stateful byte-level recurrent model designed for quantitative order book sequence forecasting. While it failed on noisy raw future direction tasks (collapsing to single-class predictions), reformulating the task into binary volatility-regime classification (`HIGH_VOL` vs `LOW_VOL` on `vol_regime_H15_C60`) resolved the collapse. Fast32 successfully learned highly predictive structural regime transitions, achieving a final test accuracy of **71.38%** with **0.00% invalid predictions** and **no class collapse** across Binance futures test splits. At inference time, the model achieves high execution speeds on an RTX 5070 Ti Laptop GPU.

---

## 2. Final Accuracy Metrics

| Metric | Target Gate | best_val_accuracy.pt Result | Status |
|---|---|---|---|
| **Test Accuracy** | $\ge 60.00\%$ | **71.38%** (+21.38% over baseline) | ✅ PASSED |
| **Macro F1 Score** | $\ge 0.600$ | **0.706** | ✅ PASSED |
| **Invalid/Empty Rate** | $= 0.00\%$ | **0.00%** | ✅ PASSED |
| **Max predicted class share** | $\le 70.00\%$ | **66.40%** | ✅ PASSED |
| **Single-class collapse** | None | **None** | ✅ PASSED |

---

## 3. Final Speed & Throughput Metrics (RTX 5070 Ti Laptop GPU)

| Benchmark Split | Batch Size | Latency per Call / Step | Throughput (Ex/sec) | Throughput (Tok/sec) |
|---|---|---|---|---|
| **Full Classifier Pipeline** | 1 | 442.8 µs / call | 2,258 | 289,067 |
| **Full Classifier Pipeline** | 256 | 8.26 µs / example | **121,029** | **15.49M** |
| **Pure Model Forward Only** | 256 | 7.16 µs / example | **139,578** | **17.87M** |
| **Original-Stateful Recurrence** | 1 | **13.82 µs / token** | — | 72,359 |
| **Stateless Fused Ablation** | 1 | **7.37 µs / token** | — | 135,685 |

---

## 4. Latency and Throughput Definition
* **13.82 µs / token**: Measures **single-token cached/fused stateful step recurrence latency** using CUDA Graphs and custom Triton kernels. This represents the core model inference speed when updating hidden state $h$.
* **7.37 µs / token**: Measures the stateless speed ablation path (depth precomposed, no hidden state updates). This is a speed baseline only.
* **442.8 µs / example**: Represents the **honest full-pipeline classification latency** at batch=1. This includes text encoding, sequence padding, GPU data transfer, forward execution over 128 tokens, argmax extraction, and string decoding.
* **8.26 µs / example**: Represents the amortized full-pipeline classification latency when batched at 256.

---

## 5. Claims and Limitations (What NOT to claim)
* **Do not claim profitability or live trading readiness**: These volatility regime labels represent realized volatility thresholds, not buy/sell/hold decisions. This is not a trading strategy.
* **Do not claim 7.65 µs full-pipeline classification speed**: The 7.65 µs / 13.82 µs metrics represent a single token step. The full 128-token classification takes ~442.8 µs.
* **Do not claim to beat classical quant ML baselines**: High-performance quant ML baselines (such as LightGBM, XGBoost, or logistic regressions) have not been trained or compared yet.

---

## 6. Historical Experiment Documentation
* [REMOVED: Volatility Regime Dataset Preparation](REMOVED/REMOVED_VOL_REGIME_DATASET_RESULTS.md)
* [REMOVED: Volatility Regime UE1 Training](REMOVED/REMOVED_VOL_REGIME_UE1_RESULTS.md)
* [REMOVED: Final Benchmark and Stability Audit](REMOVED/REMOVED_FINAL_BENCHMARK_AUDIT.md)
* [REMOVED: Speed Benchmark Definition Audit](REMOVED/REMOVED_SPEED_BENCHMARK_DEFINITION_AUDIT.md)
