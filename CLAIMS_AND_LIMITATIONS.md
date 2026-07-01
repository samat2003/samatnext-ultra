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
  * **13–14 µs** is the speed of a **single token step** (stateful CUDA Graph Triton recurrence). Not comparable to the full 128-token classification.
  * **0.443 ms (443 µs)** is the **full-pipeline classification latency** at batch=1. This includes text encoding, GPU upload, 128-token forward pass, argmax, and decode.
* **Throughput vs. Latency**:
  * **121,029 examples/sec** (batch=256) is amortised batched *inference* throughput. Do not describe it as update speed, training speed, or single-item latency.
  * **15.49M input tokens/sec** is the corresponding input-token throughput for batch=256 inference. Do not describe it as training throughput.
* **Ablation vs. Production**:
  * **6–8 µs / token** is the stateless fused ablation speed. This ablation path does not update hidden state and is not the production inference path.

---

## 4. Hardware Dependency
* All reported benchmarks were evaluated on an **NVIDIA GeForce RTX 5070 Ti Laptop GPU**. Latency and throughput figures will vary depending on your system's GPU cores, memory bandwidth, and thermal limits.
* Triton kernels require compatible Linux environments with CUDA capability. Running on CPU is possible via standard PyTorch fallbacks but will not achieve low-latency figures.
