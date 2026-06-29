# Experiment 003 GPU Speed Plan

Status: waiting_for_user_approval

Title: Fast GPU Synthetic Training Benchmark for Dynamic DNA-SSM

## Objective

Move from Experiment 002 correctness to honest GPU speed measurement and optimization for Dynamic DNA-SSM.

The eventual target is `100M+` synthetic training tokens/sec, but Experiment 003 must not claim that until it is measured on real hardware.

## Files To Create / Edit

Implementation files to create/edit after approval:

- `scripts/exp003_gpu_bench.py` - GPU synthetic benchmark CLI for forward-only, forward+loss, and train-step throughput.
- `tests/test_exp002_model.py` - keep existing Experiment 002 correctness tests intact; add only minimal regression coverage if needed for benchmark-safe helpers.
- `docs/experiments/EXP003_GPU_SPEED_RESULTS.md` - benchmark results and interpretation after implementation and measurement.

Files not planned for architecture redesign:

- `samatnext_dna_ssm/model.py` - avoid architecture changes unless a small benchmark hook is strictly necessary and does not affect Experiment 002 correctness.
- No separate LM head.
- No trainable ACT head.
- No DNA hidden-size change.
- No stored per-layer parameters.

## Hard Correctness Rules

Experiment 003 must keep all Experiment 002 correctness guarantees passing:

- Total trainable parameters stay under `1,000,000`.
- Causal prefix invariance remains tested and passing.
- No separate LM head; output projection remains tied to `token_embed.weight.T`.
- No trainable ACT head; halting remains parameter-free.
- No stored per-layer SSM parameters.
- Generated SSM values remain chunked ephemeral vectors only: `A`, `B`, `C`, `G` each `[chunk_layers, d_model]`.
- No generated dense `[d_model, d_model]` SSM matrices.

## FlashAttention Decision

FlashAttention is not directly useful for the current Dynamic DNA-SSM path because the model has no attention operation, no Q/K/V tensors, and no attention softmax.

The main expected bottleneck is the Python loop over generated SSM layers and token positions:

```text
for layer in chunk:
  for token in sequence:
    h_t = sigmoid(A_i) * h_{t-1} + B_i * x_t
    y_t = C_i * h_t
    x_t = x_t + residual_scale * silu(G_i) * y_t
```

Do not add FlashAttention for the SSM path. It is only relevant if Experiment 003 later adds a separate attention baseline, which is not part of the initial benchmark implementation.

## Stage A: GPU Baseline Script

Create `scripts/exp003_gpu_bench.py`.

Benchmark constraints:

- Use synthetic static CUDA batches only for speed tests.
- Pre-generate input and target tensors on the target device before timing.
- Warm up before measured iterations.
- Use `torch.cuda.synchronize()` around timing when device is CUDA.
- Report honest CPU/CUDA availability and fail clearly if `--device cuda` is requested without CUDA.
- Do not claim throughput from CPU fallback as GPU throughput.

Required reported fields:

- device name
- torch version
- CUDA version
- dtype
- batch size
- seq len
- max_layers
- chunk_size
- layers_used
- forward-only tok/s
- forward+loss tok/s
- train-step tok/s
- layer-token-updates/sec
- peak CUDA memory
- optimizer used
- `torch.compile` mode used or disabled

Metric definitions:

- `tokens = batch_size * seq_len`
- `forward-only tok/s = tokens * measured_iters / forward_only_elapsed`
- `forward+loss tok/s = tokens * measured_iters / forward_loss_elapsed`
- `train-step tok/s = tokens * optimizer_steps_or_measured_steps / train_elapsed`
- `layer-token-updates/sec = tokens * layers_used * measured_iters / elapsed`

For UE4/UE8, report both:

- forward/loss calls per second
- optimizer updates count

This avoids overstating optimizer throughput when backward/optimizer is skipped on non-update steps.

## Stage B: PyTorch Accelerator Toggles

Add CLI flags:

- `--amp bf16/fp16/off`
- `--compile off/reduce-overhead/max-autotune`
- `--optimizer adamw/fused-adamw/sgd`
- `--update-every 1/4/8`
- `--forward-only`
- `--no-grad-only`

AMP behavior:

- `--amp off`: use fp32.
- `--amp bf16`: use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` on CUDA.
- `--amp fp16`: use `torch.autocast(device_type="cuda", dtype=torch.float16)` on CUDA.
- On CPU, either reject AMP modes or clearly report that CUDA AMP was not used.

Compile behavior:

- `--compile off`: do not compile.
- `--compile reduce-overhead`: use `torch.compile(..., mode="reduce-overhead")`.
- `--compile max-autotune`: use `torch.compile(..., mode="max-autotune")`.
- If graph breaks, compile failures, or slowdowns occur, report them honestly in results.

Optimizer behavior:

- `adamw`: use standard `torch.optim.AdamW`.
- `fused-adamw`: use `torch.optim.AdamW(fused=True)` when CUDA/PyTorch supports it.
- If fused AdamW is unavailable, fall back safely to AdamW and report the fallback.
- `sgd`: use `torch.optim.SGD`.

Update cadence:

- Keep Experiment 002 UE rule:
  - forward/loss every step
  - backward/optimizer only when `step % update_every == 0`
- Allow `--update-every 1`, `4`, or `8`.
- Report `update_every` and `optimizer_updates`.

## Stage C: Reduce Python Overhead

Try `torch.compile` on fixed-shape, fixed-depth benchmark settings.

Initial fixed-shape targets:

- fixed batch size
- fixed sequence length
- fixed `max_layers`
- fixed `chunk_size`
- fixed `max_chunks` or no halting threshold changes during the timed region

Expected risk:

- The current model has Python loops over layer count and sequence positions.
- `torch.compile` may graph break, specialize heavily, compile slowly, or fail to speed up the recurrence.
- If this happens, report the graph-break or compile overhead honestly.

Do not remove or weaken Experiment 002 tests to make compilation easier.

## Stage D: Real Speed Path

After baseline numbers exist, prepare a custom Triton or CUDA fused kernel for the diagonal causal SSM chunk.

Target recurrence:

```text
for each layer and token:
  h_t = sigmoid(A_i) * h_{t-1} + B_i * x_t
  y_t = C_i * h_t
  x_t = x_t + residual_scale * silu(G_i) * y_t
```

Candidate fusion scope:

- recurrence scan
- residual update
- maybe chunk loop
- maybe hidden/logit RMS measurement

Do not implement Triton or custom CUDA until baseline GPU numbers exist. The first Experiment 003 implementation should measure the current PyTorch path and identify where time is going.

## Parameter Count Estimate

Experiment 003 should preserve Experiment 002 parameter count:

```text
Token embedding:
  256 * 256 = 65,536

Tied output projection:
  0 additional parameters

DNA hypernetwork:
  Linear(16, 128): 2,176
  Linear(128, 128): 16,512
  Linear(128, 1024): 132,096
  DNA total = 150,784

Parameter-free halt score:
  0 parameters

Total:
  216,320
```

Any benchmark-only helper must not add trainable parameters.

## VRAM Strategy

- Use static synthetic CUDA batches to avoid data-loader noise.
- Reset CUDA peak memory stats before each measured benchmark section.
- Report `torch.cuda.max_memory_allocated()`.
- Keep generated SSM chunk tensors proportional to `chunk_size`, not `max_layers`.
- Do not claim 1M-layer training O(1) VRAM.
- Training memory is still normal PyTorch autograd memory through executed depth unless checkpointing/rematerialization is explicitly implemented and measured.

## Causality Test

Keep the existing future-token mutation test:

- Run original and suffix-mutated token sequences.
- Compare prefix logits.
- Assert prefix logits are unchanged within tolerance.

Experiment 003 benchmark code must not change model behavior in a way that breaks this test.

## Commands To Run

Correctness:

```bash
python -m pytest -q
```

GPU baseline commands:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --forward-only
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --optimizer fused-adamw --update-every 1
python scripts/exp003_gpu_bench.py --device cuda --batch-size 512 --seq-len 512 --max-layers 1 --chunk-size 1 --amp bf16 --forward-only
```

Optional comparison runs after the required baseline:

```bash
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp off --forward-only
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --compile reduce-overhead --forward-only
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --optimizer fused-adamw --update-every 4
python scripts/exp003_gpu_bench.py --device cuda --batch-size 128 --seq-len 256 --max-layers 8 --chunk-size 4 --amp bf16 --optimizer fused-adamw --update-every 8
```

## Success Criteria

- Establish an honest GPU baseline.
- Report actual measured tokens/sec before making any `100M+` claim.
- Identify whether Python loop overhead dominates.
- Keep Experiment 002 correctness intact.
- Document why FlashAttention is not used for the SSM path.
- Produce enough baseline evidence to decide whether `torch.compile`, Triton, or custom CUDA is the next practical optimization step.

## Risks / Limitations

- Current Python loops over layers and sequence positions may dominate runtime and prevent high GPU utilization.
- `torch.compile` may not help if dynamic control flow or Python loops cause graph breaks.
- Fused AdamW may be unavailable depending on PyTorch/CUDA build.
- BF16 performance depends on GPU architecture.
- Static synthetic batches measure model throughput, not real data pipeline throughput.
- Short benchmarks can overstate speed if warmup, synchronization, or memory accounting is wrong.
- Full 1M-layer training is not part of Experiment 003 baseline and must not be implied by small-depth benchmark results.

## Approval Status

Status: waiting_for_user_approval
