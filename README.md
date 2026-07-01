# Fast32: Stateful Byte-Level Sequence Model

The final Fast32 volatility-regime classifier reaches **71.38%** full-test accuracy with **0%** invalid outputs and no single-class collapse. On an RTX 5070 Ti Laptop GPU, the REMOVED final classifier benchmark runs at **443 µs/example** at batch=1 and **121K examples/sec** at batch=256 (amortized). The original-stateful CUDA-graph token-step path runs in the **13–14 µs/token** range. These are volatility-regime labels, not trading decisions.

---

## Final Project Status

* **Model Architecture**: Stateful Fast32 (DynamicDnaSsmLM, d_model=256, 32 layers)
* **Parameter Count**: **216,320**
* **Task**: Binary volatility-regime classification (`HIGH_VOL` vs `LOW_VOL`)
* **Test Dataset**: `vol_regime_H15_C60` (83,856 examples)
* **Checkpoint SHA256**: `b2f304a0ff5dec4beaddc9d15fde8dad42d73338f8c4c8f25be9ef665f3c38a4`

---

## Quick Start (WSL/Linux + CUDA)

Verify setup and reproduce all final numbers. 

> [!IMPORTANT]
> Because this repository is private, clean-clone replication relies on the GitHub CLI (`gh`) to securely download checkpoint and dataset release assets. Ensure `gh` is installed and authenticated (`gh auth login`) on your system before running the commands.

```bash
git clone https://github.com/samat2003/samatnext-ultra.git
cd samatnext-ultra
make setup
make reproduce-final
```

For detailed setup instructions and options, see [REPRODUCE.md](REPRODUCE.md).

---

## Project Documentation
* [FINAL_RESULTS.md](FINAL_RESULTS.md) — Summary tables of final accuracy and speed metrics.
* [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md) — Critical guidelines on baselines, signal limits, and what *not* to claim.

---

## Historical Experiment Documentation
* [REMOVED: Volatility Regime Dataset Preparation](REMOVED/REMOVED_VOL_REGIME_DATASET_RESULTS.md)
* [REMOVED: Volatility Regime UE1 Training](REMOVED/REMOVED_VOL_REGIME_UE1_RESULTS.md)
* [REMOVED: Final Benchmark and Stability Audit](REMOVED/REMOVED_FINAL_BENCHMARK_AUDIT.md)
* [REMOVED: Speed Benchmark Definition Audit](REMOVED/REMOVED_SPEED_BENCHMARK_DEFINITION_AUDIT.md)

---

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
