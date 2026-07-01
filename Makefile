# Makefile for Fast32 Volatility-Regime reproduction

.PHONY: setup download-artifacts verify reproduce-accuracy benchmark reproduce-final

setup:
	@echo "=== Setting up Environment ==="
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"

download-artifacts:
	@echo "=== Downloading Checkpoints & Dataset ==="
	.venv/bin/python scripts/download_final_artifacts.py

verify:
	@echo "=== Verifying Checkpoint & Setup ==="
	.venv/bin/python scripts/verify_checkpoint.py

reproduce-accuracy:
	@echo "=== Reproducing Test Accuracy ==="
	.venv/bin/python scripts/reproduce_final_accuracy.py

benchmark:
	@echo "=== Running Speed Benchmarks ==="
	.venv/bin/python scripts/benchmark_final_vol_regime_audit.py

reproduce-final: download-artifacts verify reproduce-accuracy benchmark
	@echo "=== Running Full Reproduction ==="
	.venv/bin/python scripts/reproduce_final.py
