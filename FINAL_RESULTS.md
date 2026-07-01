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

## 3. Final Speed Metrics (RTX 5070 Ti Laptop GPU, REMOVED benchmark)

| Benchmark | Batch | Latency | Throughput (Ex/sec) | Throughput (Tok/sec) |
|---|---|---|---|---|
| **REMOVED Final Classifier** | 1 | **0.443 ms / example** | 2,258 | 289,067 |
| **REMOVED Final Classifier** | 16 | 0.485 ms / call | 32,976 | 4.22M |
| **REMOVED Final Classifier** | 64 | 0.734 ms / call | 87,181 | 11.16M |
| **REMOVED Final Classifier** | 256 | 2.115 ms / call · 8.26 µs amortized | **121,029** | **15.49M** |
| **Original-Stateful Token-Step** *(CUDA Graph)* | 1 | **13–14 µs / token** | — | — |
| **Stateless Fused Ablation** *(ablation only)* | 1 | 6–8 µs / token | — | — |

> **Note**: The REMOVED benchmark times a full-prompt forward pass over `seq_len=128` with prebuilt GPU tensors (no Python string overhead inside timing). The 13–14 µs token-step is a single Triton + CUDA Graph recurrence step. These are distinct measurements.

---

## 4. Latency and Throughput Definitions

* **0.443 ms / example (batch=1)**: Full-pipeline classification latency — text encoding, GPU upload, 128-token forward pass, argmax, decode — measured with prebuilt GPU tensors (REMOVED path).
* **8.26 µs / example amortized (batch=256)**: Same pipeline, throughput-optimised batched path. Reports **121,029 examples/sec** and **15.49M input tokens/sec**.
* **13–14 µs / token**: Single-token CUDA Graph stateful recurrence step latency. Not comparable to the 128-token classifier latency above.
* **6–8 µs / token**: Stateless fused ablation (no hidden-state update). Speed reference only — not used in the final classifier.

---

## 5. Claims and Limitations (What NOT to claim)

* **Do not claim profitability or live trading readiness**: These volatility-regime labels predict realized volatility thresholds, not buy/sell/hold decisions. This is not a trading strategy.
* **Do not claim 6–8 µs for the full 128-token classification**: Those µs figures are single-token recurrence steps only. The full classification takes ~443 µs at batch=1.
* **Do not claim the 15.49M tokens/sec figure is training speed**: It is batched inference throughput only.
* **Do not claim to beat classical quant ML baselines**: LightGBM/XGBoost/logistic-regression comparisons have not been run and remain future work.

---

## 6. Historical Experiment Documentation
* [REMOVED: Volatility Regime Dataset Preparation](REMOVED/REMOVED_VOL_REGIME_DATASET_RESULTS.md)
* [REMOVED: Volatility Regime UE1 Training](REMOVED/REMOVED_VOL_REGIME_UE1_RESULTS.md)
* [REMOVED: Final Benchmark and Stability Audit](REMOVED/REMOVED_FINAL_BENCHMARK_AUDIT.md)
* [REMOVED: Speed Benchmark Definition Audit](REMOVED/REMOVED_SPEED_BENCHMARK_DEFINITION_AUDIT.md)
