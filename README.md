# Fast32: Stateful Byte-Level Sequence Model

The final Fast32 volatility-regime classifier reaches **71.38%** full-test accuracy with **0%** invalid outputs and no single-class collapse. On an RTX 5070 Ti Laptop GPU, the REMOVED final classifier benchmark runs at **443 µs/example** at batch=1 and **121K examples/sec** at batch=256 (amortized). The original-stateful CUDA-graph token-step path runs in the **13–14 µs/token** range. These are volatility-regime labels, not trading decisions.

See [FINAL_RESULTS.md](FINAL_RESULTS.md), [REPRODUCE.md](REPRODUCE.md), and [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md).
