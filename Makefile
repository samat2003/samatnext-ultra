# Makefile for Fast32 Volatility-Regime reproduction

.PHONY: setup verify reproduce-accuracy benchmark reproduce-final

setup:
	@echo "=== Setting up Environment ==="
	@bash -c "source .venv/bin/activate && python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.version.cuda); print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no CUDA\")'"

verify:
	@echo "=== Verifying Checkpoint & Setup ==="
	@bash -c "source .venv/bin/activate && python scripts/verify_checkpoint.py"

reproduce-accuracy:
	@echo "=== Reproducing Test Accuracy ==="
	@bash -c "source .venv/bin/activate && python scripts/reproduce_final_accuracy.py"

benchmark:
	@echo "=== Running Speed Benchmarks ==="
	@bash -c "source .venv/bin/activate && python scripts/benchmark_final_vol_regime_audit.py"

reproduce-final:
	@echo "=== Running Full Reproduction ==="
	@bash -c "source .venv/bin/activate && python scripts/reproduce_final.py"
