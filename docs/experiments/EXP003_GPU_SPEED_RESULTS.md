# Experiment 003 GPU Speed Results

Status: implemented_cuda_measured

## Scope

Experiment 003 adds a benchmark script for the existing Dynamic DNA-SSM V0 architecture. It does not redesign `DynamicDnaSsmLM`, add attention, add FlashAttention, add a separate LM head, add a trainable ACT head, change the DNA hidden size, or add stored per-layer parameters.

## Commands

Correctness:

```bash
python -m pytest -q
```

GPU benchmark commands from the plan:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --forward-only
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --optimizer fused-adamw --update-every 1
python scripts/exp003_gpu_bench.py --device cuda --batch-size 512 --seq-len 512 --max-layers 1 --chunk-size 1 --amp bf16 --forward-only
```

CPU fallback command when CUDA is unavailable:

```bash
python scripts/exp003_gpu_bench.py --device cpu --batch-size 8 --seq-len 64 --max-layers 4 --chunk-size 2 --amp off --forward-only
```

## Results

## CUDA Environment Diagnosis

- `nvidia-smi`: available at `/usr/lib/wsl/lib/nvidia-smi`; reports `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, driver/KMD `610.47`, CUDA UMD `13.3`, 12,227 MiB memory.
- `nvcc`: not available; `nvcc not found`.
- `torch before`: `2.12.1+cpu`, `torch.version.cuda = null`, `torch.cuda.is_available() = false`, CPU-only build.
- `torch after`: `2.11.0+cu128`, `torch.version.cuda = 12.8`, `torch.cuda.is_available() = true`.
- `torch.cuda.is_available`: `true` after replacing the CPU-only wheel with the CUDA 12.8 PyTorch wheel.
- `device`: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, capability `(12, 0)`, device count `1`.
- `conclusion`: CUDA could be enabled from Python because the system driver/GPU was available. The blocker was the virtualenv's CPU-only PyTorch wheel. Installing the CUDA 12.8 PyTorch wheel fixed CUDA visibility.

Correctness:

- `python -m pytest -q` passed: `6 passed`.
- After CUDA PyTorch install, the previous NumPy warning is gone because NumPy was installed as a dependency.

GPU baseline 1:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --forward-only
```

- `torch_version`: `2.11.0+cu128`
- `cuda_version`: `12.8`
- `device_name`: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- `tokens_per_iter`: `32768`
- `layers_used`: `8.0`
- `forward_only_tok_s`: `306632.55355806707`
- `forward_only_layer_token_updates_s`: `2453060.4284645366`
- `peak_cuda_memory_bytes`: `145750016`
- `reached_100m_tok_s`: `false`

GPU baseline 2:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --optimizer fused-adamw --update-every 1
```

- `optimizer_requested`: `fused-adamw`
- `optimizer_used`: `fused-adamw`
- `optimizer_fallback`: `null`
- `tokens_per_iter`: `32768`
- `layers_used`: `8.0`
- `forward_only_tok_s`: `264179.8579041837`
- `forward_loss_tok_s`: `325293.97810410213`
- `train_step_tok_s`: `41876.347699175945`
- `train_layer_token_updates_s`: `335010.78159340756`
- `optimizer_updates`: `13` including warmup and measured iterations
- `peak_cuda_memory_bytes`: `969953792`
- `reached_100m_tok_s`: `false`

GPU baseline 3:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 512 --seq-len 512 --max-layers 1 --chunk-size 1 --amp bf16 --forward-only
```

- `tokens_per_iter`: `262144`
- `layers_used`: `1.0`
- `forward_only_tok_s`: `5832042.173296551`
- `forward_only_layer_token_updates_s`: `5832042.173296551`
- `peak_cuda_memory_bytes`: `1088673792`
- `reached_100m_tok_s`: `false`

Compile mode: `reduce-overhead`

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --compile reduce-overhead --forward-only
```

- `compile_mode`: `reduce-overhead`
- `compile_error`: `null`
- Graph break warning observed at `halt_score = float(halt_score_tensor.detach().cpu())`.
- `forward_only_tok_s`: `4072697.9568395736`
- `forward_only_layer_token_updates_s`: `32581583.65471659`
- `peak_cuda_memory_bytes`: `135618560`
- `reached_100m_tok_s`: `false`

Compile mode: `max-autotune`

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --compile max-autotune --forward-only
```

- `compile_mode`: `max-autotune`
- `compile_error`: `null`
- Graph break warning observed at `halt_score = float(halt_score_tensor.detach().cpu())`.
- Inductor reported: `Not enough SMs to use max_autotune_gemm mode`.
- `forward_only_tok_s`: `4989071.181368442`
- `forward_only_layer_token_updates_s`: `39912569.45094754`
- `peak_cuda_memory_bytes`: `139805184`
- `reached_100m_tok_s`: `false`

Earlier CPU fallback benchmark before CUDA was fixed:

- Command: `python scripts/exp003_gpu_bench.py --device cpu --batch-size 8 --seq-len 64 --max-layers 4 --chunk-size 2 --amp off --forward-only`
- `forward_only_tok_s`: `74428.22622987024`
- `forward_only_layer_token_updates_s`: `297712.90491948096`
- `forward_loss_tok_s`: `null`
- `train_step_tok_s`: `null`
- `peak_cuda_memory_bytes`: `null`

## Interpretation

- CUDA is now working in the virtualenv.
- The best measured GPU forward-only throughput in these runs was `5.83M tok/s` for the shallow `max_layers=1`, large-batch setting.
- The best measured compiled `max_layers=8` forward-only throughput was `4.99M tok/s`.
- The measured training throughput with fused AdamW was `41.9k tok/s`.
- `100M+ tok/s` was not reached.
- `torch.compile` helped the forward-only path significantly, but graph breaks remain around scalar metadata extraction in `DynamicDnaSsmLM.forward`.
- FlashAttention was not used because the Dynamic DNA-SSM path has no attention/QKV operation.
- Recommended next optimization step: remove benchmark-hostile scalar extraction from the timed compiled path without changing model semantics, then prototype a Triton or CUDA fused diagonal causal SSM chunk kernel if Python loop overhead still dominates.
