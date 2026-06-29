# Experiment 003B: Fused Triton Fixed-1000-Layer SSM Kernel

Status: waiting_for_user_approval

## Context

The fixed 1000-layer Experiment 003 audit showed that the current eager CUDA/PyTorch path is far too slow for the target workload:

- 1000-layer forward-only: about `78.84` input tok/s.
- 1000-layer layer-token-updates/sec: about `78,842`.
- 1000-layer UE4 train-step: about `27.56` input tok/s.
- Parameter count remains `216,320`.
- No stored layer modules.
- No separate LM head.
- No trainable ACT head.

The earlier `107M tok/s` result was a shallow `max_layers=1` compiled forward-only result. It must not be counted as success for the fixed 1000-layer target.

## Goal

Replace the Python/PyTorch layer/token loop with a fused Triton forward-only kernel for fixed 1000 generated SSM layers.

The first implementation should measure a true fixed-depth path:

- `max_layers=1000`
- `chunk_size=1000`
- no ACT early halt in the timed kernel path
- no `torch.compile` dependency for the main speed result

## Non-Negotiables

- `max_layers=1000`
- `chunk_size=1000`
- `d_model=256`
- `vocab_size=256`
- Total trainable parameters must stay `216,320`.
- No stored per-layer parameters.
- No separate LM head.
- No trainable ACT head.
- No attention.
- No FlashAttention.
- No transformer baseline yet.
- DNA hypernetwork still generates ephemeral `A/B/C/G` vectors in PyTorch.
- Triton applies `A/B/C/G` to `x`.
- Report input tok/s and layer-token-updates/sec separately.

## Metric Clarification

`100M` input tok/s through 1000 layers means:

```text
100,000,000 input tok/s * 1000 layers = 100,000,000,000 layer-token-updates/sec
```

Do not claim `100M` input tok/s unless that exact input-token metric is reached for the true 1000-layer path.

The first realistic kernel target is `100M` layer-token-updates/sec. That is still about `100,000` input tok/s for 1000 layers.

## Files

Plan file to create now:

- `docs/experiments/EXP003B_TRITON_1000_LAYER_PLAN.md`

Later implementation files after approval:

- `samatnext_dna_ssm/triton_ssm.py`
- `scripts/exp003b_triton_ssm_bench.py`
- `tests/test_exp003b_triton_ssm.py`
- `docs/experiments/EXP003B_TRITON_1000_LAYER_RESULTS.md`

Do not implement these later files until the plan is approved.

## Architecture Boundary

Experiment 003B should not redesign `DynamicDnaSsmLM`.

The PyTorch model still owns:

- token embedding
- tied output projection
- DNA hypernetwork
- generated ephemeral `A/B/C/G`
- parameter count and correctness tests

The Triton kernel only replaces the inner fixed-depth SSM application:

```text
x, A_sig, B, C, G_silu -> out
```

This keeps the parameter budget independent of generated layer count.

## Parameter Count

The trainable parameter count remains the Experiment 002/003 count:

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

Generated Triton inputs `A_sig/B/C/G_silu` are ephemeral tensors produced from the DNA hypernetwork. They are not stored per-layer trainable parameters.

## Kernel Strategy

Precompute before the Triton kernel where possible:

```python
A_sig = torch.sigmoid(A)
G_silu = torch.nn.functional.silu(G)
```

Triton kernel inputs:

```text
x:      [batch, seq, d_model]
A_sig:  [1000, d_model]
B:      [1000, d_model]
C:      [1000, d_model]
G_silu: [1000, d_model]
out:    [batch, seq, d_model]
```

Initial Triton operation:

```text
1. Tile over batch and channel dimension.
2. Load a tile of x for one batch block and one channel block.
3. Keep the tile hot inside the Triton program as much as register/shared-memory pressure allows.
4. Initialize h = zeros for the channel block.
5. For layer in 0..999:
    a. Load A_sig[layer, channel_block], B[layer, channel_block],
       C[layer, channel_block], G_silu[layer, channel_block].
    b. For token t in 0..BLOCK_SEQ-1:
        h = A_sig[layer] * h + B[layer] * x[t]
        y = C[layer] * h
        x[t] = x[t] + residual_scale * G_silu[layer] * y
6. Store the processed x tile back to HBM.
```

Initial tile candidates:

- `BLOCK_D`: `16`, `32`, `64`
- `BLOCK_SEQ`: `16`, `32`, `64`, `128`

Benchmark multiple tile sizes. Do not assume one tile shape is optimal.

## Implementation Cautions

- Do not claim the kernel cannot bottleneck on memory bandwidth.
- It can still bottleneck on `A/B/C/G` streaming, register pressure, occupancy, memory bandwidth, or recurrence dependency.
- Do not try to keep the entire `[seq_len=512, d_model=256]` tensor in registers.
- Do not load all `[1000, 256]` `A/B/C/G` tensors into SRAM at once.
- Stream `A/B/C/G` by layer and channel block.
- Start with fp32 correctness.
- Then run bf16 speed tests.
- Keep the 1000-layer path fixed-depth for initial benchmarking.
- Do not add ACT halting to the timed Triton path.

## Correctness Tests

Add `tests/test_exp003b_triton_ssm.py` after approval.

Tests:

1. Compare Triton output against PyTorch reference for small shapes:
   - `batch=1`
   - `seq_len=8` or `16`
   - `d_model=16` or `32`
   - `layers=4`, `16`, `64`
2. Compare `d_model=256`, `layers=1000` on small batch/seq.
3. Report:
   - `max_abs_error`
   - `mean_abs_error`
4. Keep causal prefix invariance.
5. Skip Triton tests cleanly if Triton is unavailable.
6. Keep all existing Experiment 002 and Experiment 003 tests passing.

Expected tolerance should be separate for fp32 and bf16:

- fp32: strict enough to catch indexing/order bugs.
- bf16: looser, documented tolerance due to reduced precision.

## Benchmark Plan

Required correctness command:

```bash
python -m pytest -q
```

Required benchmark commands:

```bash
python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 16 \
  --d-model 256 \
  --layers 1000 \
  --amp fp32

python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 4 \
  --seq-len 32 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16

python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 16 \
  --seq-len 64 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16

python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16
```

Required benchmark metrics:

- PyTorch reference input tok/s
- Triton input tok/s
- PyTorch layer-token-updates/sec
- Triton layer-token-updates/sec
- Triton speedup
- `max_abs_error`
- `mean_abs_error`
- peak CUDA memory
- tile sizes used
- whether `100M` input tok/s was reached
- whether `100M` layer-token-updates/sec was reached

## Explicit Answers

1. Can this preserve `216,320` trainable parameters?

Yes. The Triton kernel applies generated vectors. It does not add trainable weights. The DNA hypernetwork and token embedding remain the only trainable components.

2. Does fixed 1000 layers increase stored parameters?

No. `A/B/C/G` for 1000 layers are generated ephemerally from the DNA hypernetwork. They are tensors for the current forward/benchmark, not stored per-layer model parameters.

3. Does the Triton kernel execute a true 1000 generated-layer path?

That is the core requirement. The timed Triton path must loop over `layer in 0..999` and apply all generated `A_sig/B/C/G_silu` vectors. No ACT early halt is allowed in the timed path.

4. What is the first realistic target: `100M` input tok/s or `100M` layer-token-updates/sec?

The first realistic target is `100M` layer-token-updates/sec. For 1000 layers, `100M` input tok/s would mean `100B` layer-token-updates/sec, which is a much harder target and must not be claimed unless directly measured.

5. Why is this more relevant than FlashAttention for the SSM path?

The Dynamic DNA-SSM path has no attention operation, no Q/K/V tensors, and no attention softmax. The measured bottleneck is the Python/PyTorch recurrence loop over layers and tokens. A fused SSM kernel targets that bottleneck directly; FlashAttention does not.

6. Why is this forward-only first?

Forward-only isolates the SSM recurrence throughput and avoids the extra complexity of custom backward, optimizer behavior, and activation rematerialization. It also gives a clear baseline for whether a fused 1000-layer path is viable before investing in training kernels.

7. What would be needed later for true training speed?

True training speed would require:

- a custom backward kernel or a rematerialization strategy
- memory strategy for activations through 1000 layers
- gradient validation against PyTorch reference
- optimizer benchmark with UE1/UE4/UE8 reporting
- honest peak VRAM measurement
- possibly checkpointing or recomputation to control memory

Do not claim 1000-layer training speed from forward-only numbers.

8. What are the main risks?

- Recurrence dependency across tokens limits parallelism.
- Register pressure from keeping sequence/channel tiles hot.
- Occupancy loss from large `BLOCK_SEQ` or `BLOCK_D`.
- Streaming `A/B/C/G` can still bottleneck on memory bandwidth.
- Triton compile time or codegen limits for a 1000-layer loop.
- Numerical drift between Triton and PyTorch, especially in bf16.
- Tile sizes may behave differently across GPUs.
- A forward-only kernel does not prove trainability or training throughput.

## Success Criteria

- Existing Experiment 002/003 tests still pass.
- Triton correctness matches PyTorch reference within documented tolerance.
- The benchmark reports both input tok/s and layer-token-updates/sec.
- The benchmark reports true fixed 1000-layer execution.
- The results clearly distinguish `100M` input tok/s from `100M` layer-token-updates/sec.
- No FlashAttention, attention, transformer baseline, stored per-layer parameters, separate LM head, or trainable ACT head are added.

## Approval Status

Status: waiting_for_user_approval
