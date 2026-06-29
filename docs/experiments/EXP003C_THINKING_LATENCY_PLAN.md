# Experiment 003C: Batch-Size-1 Thinking Latency and Jitter

Status: waiting_for_user_approval

## Context

Experiment 003B measured real synthetic training speed and showed training is not the right optimization target right now.

Current facts:

- Fixed 1000-layer DNA-SSM trainable params: `216,320`
- True fixed 1000-layer training speed: best `65.85` real input tok/s
- Triton forward-only kernel reached about `30.19M` layer-token-updates/sec
- Triton forward-only did not support backward
- Training path still used PyTorch autograd
- Loss decreased, so learning works, but training speed is not close to `100M` input tok/s

## Pivot

Stop optimizing training speed for now.

Focus on inference/thinking speed for a fixed-depth model.

Terminology:

- Use `thinking speed` for inference/no-grad speed.
- Use `thinking_layer_token_updates/sec` for:

```text
layers * input_tokens / inference_elapsed_sec
```

For fixed 1000 layers:

```text
100M thinking_layer_token_updates/sec = 100K input tok/s = 10 microseconds per input token
```

Do not call this `100M input tokens/sec`.

## Goal

Verify batch-size-1 latency, jitter, and layer-depth scaling for the fixed-depth DNA-SSM thinking path.

## Non-Negotiables

- Same architecture.
- Same parameter budget: `216,320`.
- `d_model=256`.
- `vocab_size=256`.
- No training.
- No backward.
- No optimizer.
- No loss in the timed path.
- No ACT early halt in the timed fixed-depth path.
- No attention.
- No FlashAttention.
- No transformer baseline.
- No separate LM head.
- No trainable ACT head.
- No stored per-layer parameters.
- DNA still generates ephemeral `A/B/C/G`.
- Triton path must execute true fixed generated layers.
- Compare `layers=1` and `layers=1000`.
- Also run a layer-depth sweep.

## Files

Plan file to create now:

- `docs/experiments/EXP003C_THINKING_LATENCY_PLAN.md`

Later implementation files after approval:

- `scripts/exp003c_thinking_latency.py`
- optional additions to `samatnext_dna_ssm/triton_ssm.py`
- `tests/test_exp003c_thinking_latency.py`
- `docs/experiments/EXP003C_THINKING_LATENCY_RESULTS.md`

Do not implement these later files until the plan is approved.

## Measurement Scope

### 1. Batch-Size-1 Single-Token Latency

Measure:

- `batch_size=1`
- `seq_len=1`
- `layers=1000`
- `d_model=256`
- implementation: Triton forward-only fixed-depth path
- dtype: fp32, bf16, fp16 if supported

Report for each:

- mean latency microseconds
- std latency microseconds
- min latency microseconds
- p50 microseconds
- p90 microseconds
- p95 microseconds
- p99 microseconds
- p99.9 microseconds
- max latency microseconds
- coefficient of variation
- input tok/s from mean latency
- thinking_layer_token_updates/sec
- peak CUDA memory
- Triton kernel used
- fallback used
- true fixed-depth execution

### 2. 1-Layer vs 1000-Layer Comparison

Run both:

- `layers=1`
- `layers=1000`

with identical settings:

- `batch_size=1`
- `seq_len=1`
- `d_model=256`
- `amp=fp32`
- `amp=bf16`
- `amp=fp16`

Purpose:

Determine whether latency is dominated by:

- Triton kernel launch overhead
- fixed model overhead
- per-layer recurrence work
- memory/register pressure from 1000 generated layers

Compute:

```text
latency_ratio_1000_vs_1 = mean_latency_1000_layers / mean_latency_1_layer
effective_depth_scaling = thinking_layer_token_updates_1000_layers / thinking_layer_token_updates_1_layer
```

Interpretation:

- If 1000-layer latency is close to 1-layer latency, kernel launch/fixed overhead dominates.
- If 1000-layer latency is close to 1000x worse, recurrence work dominates.
- If 1000-layer latency is between those, report the effective scaling.
- Do not claim `100M` input tok/s.
- For 1000 layers, report whether `thinking_layer_token_updates/sec >= 100M`.
- For 1 layer, report input tok/s separately but do not count it as success for the 1000-layer goal.

### 3. Layer-Depth Sweep

Run:

```text
layers = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000
```

with:

- `batch_size=1`
- `seq_len=1`
- `d_model=256`
- `amp=fp16`

Report:

- mean latency microseconds
- p99 latency microseconds
- input tok/s
- thinking_layer_token_updates/sec
- latency ratio vs 1 layer

### 4. Jitter

Run many timed iterations:

- warmup: at least `100` iterations
- measured: at least `10,000` iterations if feasible

Report:

- mean latency microseconds
- std latency microseconds
- min latency
- p50
- p90
- p95
- p99
- p99.9
- max latency
- coefficient of variation
- input tokens/sec from mean latency
- thinking_layer_token_updates/sec
- peak CUDA memory

### 5. Batch-Size Scaling

Run:

```text
batch_size = 1, 2, 4, 8, 16, 32
seq_len = 1
layers = 1000
```

### 6. Sequence Scaling

Run:

```text
batch_size = 1
seq_len = 1, 4, 16, 64, 128
layers = 1000
```

### 7. Comparison Paths

Compare:

- PyTorch eager fixed-depth path
- Triton fixed-depth path
- existing forward-only sequence Triton kernel if applicable

Do not compare training speed in this experiment.

## Timing Rules

- Use CUDA events for GPU elapsed time.
- Optionally report host wall-clock timing separately.
- Synchronize correctly.
- Do not include setup, tensor allocation, DNA generation, or precomputation inside the timed inner loop unless explicitly running an end-to-end variant.
- Report kernel-only and end-to-end modes separately.

### A. Kernel-Only Thinking Latency

Kernel-only mode includes:

- precomputed `A_sig/B/C/G_silu`
- preallocated input/state/output
- timed loop only calls the kernel

### B. End-To-End Thinking Latency

End-to-end mode includes:

- token embedding
- DNA `A/B/C/G` generation
- Triton SSM kernel
- output projection/logits
- no sampling unless separately measured

## Stateful Inference Option

Plan a stateful batch-size-1 step kernel if practical:

- persistent state `h`
- one incoming token/event
- apply fixed generated layers
- update `h`
- return output/logits

This is closer to trading/event-stream inference than full sequence processing.

The first implementation may keep this as a planned extension if the current Triton kernel is sequence-tile oriented.

## Required Metrics

The benchmark script must report:

- `kernel_only_mean_us`
- `kernel_only_p50_us`
- `kernel_only_p90_us`
- `kernel_only_p95_us`
- `kernel_only_p99_us`
- `kernel_only_p999_us`
- `kernel_only_max_us`
- `kernel_only_std_us`
- `kernel_only_cv`
- `kernel_only_input_tok_s`
- `kernel_only_thinking_layer_token_updates_s`
- `end_to_end_mean_us`
- `end_to_end_p99_us`
- `end_to_end_p999_us`
- `end_to_end_max_us`
- `end_to_end_input_tok_s`
- `end_to_end_thinking_layer_token_updates_s`
- `peak_cuda_memory_bytes`
- `triton_kernel_used`
- `fallback_used`
- `true_fixed_depth_execution`
- `layers`
- `parameter_count`

## Target Interpretation

- `100M thinking_layer_token_updates/sec` is the first meaningful target.
- For 1000 layers, that means `100K input tok/s` or `10 us/token`.
- For layers=1, 100M thinking_layer_token_updates/sec equals 100M input tok/s, but this is only a shallow-kernel ceiling and must not be counted as success for the fixed 1000-layer target.
- Do not claim `100M input tok/s`.
- Report whether p99 latency is under `50 us`.
- Report whether p99.9 latency is under `100 us`.

## Commands To Include

Correctness:

```bash
python -m pytest -q
```

Batch-size-1 dtype commands:

```bash
python scripts/exp003c_thinking_latency.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 1 \
  --layers 1000 \
  --d-model 256 \
  --amp fp32 \
  --warmup-iters 100 \
  --measure-iters 10000 \
  --force-triton

python scripts/exp003c_thinking_latency.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 1 \
  --layers 1000 \
  --d-model 256 \
  --amp bf16 \
  --warmup-iters 100 \
  --measure-iters 10000 \
  --force-triton

python scripts/exp003c_thinking_latency.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 1 \
  --layers 1000 \
  --d-model 256 \
  --amp fp16 \
  --warmup-iters 100 \
  --measure-iters 10000 \
  --force-triton
```

1-layer vs 1000-layer commands:

```bash
for L in 1 1000; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size 1 \
    --seq-len 1 \
    --layers $L \
    --d-model 256 \
    --amp fp32 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done

for L in 1 1000; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size 1 \
    --seq-len 1 \
    --layers $L \
    --d-model 256 \
    --amp bf16 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done

for L in 1 1000; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size 1 \
    --seq-len 1 \
    --layers $L \
    --d-model 256 \
    --amp fp16 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done
```

Layer sweep command:

```bash
for L in 1 2 4 8 16 32 64 128 256 512 1000; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size 1 \
    --seq-len 1 \
    --layers $L \
    --d-model 256 \
    --amp fp16 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done
```

Batch scaling commands:

```bash
for B in 1 2 4 8 16 32; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size $B \
    --seq-len 1 \
    --layers 1000 \
    --d-model 256 \
    --amp fp16 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done
```

Sequence scaling commands:

```bash
for S in 1 4 16 64 128; do
  python scripts/exp003c_thinking_latency.py \
    --device cuda \
    --batch-size 1 \
    --seq-len $S \
    --layers 1000 \
    --d-model 256 \
    --amp fp16 \
    --warmup-iters 100 \
    --measure-iters 10000 \
    --force-triton
done
```

## Results Document

Create after approval:

- `docs/experiments/EXP003C_THINKING_LATENCY_RESULTS.md`

Required sections:

```markdown
## Batch-Size-1 Thinking Latency

| layers | amp | mean_us | std_us | p50_us | p90_us | p95_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s | peak_cuda_memory_bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
```

```markdown
## Layer Scaling: 1 vs 1000

| layers | amp | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s | peak_cuda_memory_bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
```

```markdown
## Layer Sweep

| layers | amp | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | latency_ratio_vs_1_layer |
|---:|---|---:|---:|---:|---:|---:|
```

```markdown
## Batch Scaling

| batch_size | seq_len | layers | amp | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s |
|---:|---:|---:|---|---:|---:|---:|---:|
```

```markdown
## Sequence Scaling

| batch_size | seq_len | layers | amp | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s |
|---:|---:|---:|---|---:|---:|---:|---:|
```

The results document must answer:

- What is batch-size-1 mean latency?
- What is batch-size-1 p99 latency?
- What is batch-size-1 p99.9 latency?
- What is batch-size-1 max latency?
- What is the jitter/std?
- How does 1 layer compare to 1000 layers?
- What is the latency ratio 1000 vs 1?
- Does depth scaling look fixed-overhead dominated or recurrence dominated?
- Did p99 stay below `50 us`?
- Did p99.9 stay below `100 us`?
- Did the model reach `100M thinking_layer_token_updates/sec`?
- What is the input tok/s equivalent?
- Was this kernel-only or end-to-end?
- Was Triton actually used?
- Was fallback used?
- Did it execute true fixed depth?
- Did parameter count remain `216,320`?
- What is the next bottleneck?

## Risks / Limitations

- CUDA event timing at very small latencies can be noisy; jitter stats require enough iterations.
- Host scheduling can affect wall-clock timing, so host timing must be reported separately if used.
- Kernel-only timing excludes DNA generation and logits projection by design.
- End-to-end timing may be dominated by embedding, DNA generation, or output projection rather than the SSM kernel.
- Current Triton SSM kernel is forward-only.
- Stateful single-token inference may require a new kernel shape with persistent `h`.
- Batch-size-1 latency can be launch-overhead dominated.
- `seq_len=1` may not reflect throughput for chunked sequence processing.

## Approval Status

Status: waiting_for_user_approval
