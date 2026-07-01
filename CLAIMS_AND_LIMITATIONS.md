# Fast32: Claims and Limitations

To maintain absolute quantitative and scientific honesty, please adhere to the following limitations when discussing Fast32 results:

---

## 1. No Trading or Profitability Claims
* **Realized Volatility only**: The classification task predicts whether the future volatility (realized range over a 15-minute horizon) will be high or low relative to the median. 
* **Not a trading system**: This model does NOT predict price direction, buy/sell entry points, order execution routing, or slippage-adjusted profit. 
* **No alpha or trading readiness**: This model has not been backtested as a trading strategy and should not be used for live trading. Accuracy above the baseline does not imply profitability.

---

## 2. Baselines and Future ML Work
* **Majority/Random Baseline**: The primary baseline used for this project is the balanced **50.00% majority/random baseline**. 
* **No classical quant ML comparison**: This model has not been benchmarked against standard quantitative machine learning models (such as LightGBM, XGBoost, or logistic regressions). Comparing Fast32 to these classical models remains future work.

---

## 3. Discrepancies in Latency Definitions
Do not misrepresent execution speeds:
* **Token Step vs. Full Sequence**:
  * **13.82 µs** is the speed of a **single token step** (stateful recurrent kernel execution).
  * **442.8 µs** is the speed of processing the **full 128-byte sequence** under a single-batch classification pipeline (text to class).
* **Throughput vs. Latency**:
  * The batched throughput of **121,029 examples/sec** (batch size 256) is an amortized metric and should not be claimed as the latency for a single-batch, single-item transaction.

---

## 4. Hardware Dependency
* All reported benchmarks were evaluated on an **NVIDIA GeForce RTX 5070 Ti Laptop GPU**. Latency and throughput figures will vary depending on your system's GPU cores, memory bandwidth, and thermal limits.
* Triton kernels require compatible Linux environments with CUDA capability. Running on CPU is possible via standard PyTorch fallbacks but will not achieve low-latency figures.
