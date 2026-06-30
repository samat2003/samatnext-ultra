# Experiment 003H: Original Stateful Fast32 Fused End-to-End Results

## Summary

Experiment 003H optimizes the architecture-preserving Fast32 path:

- original stateful 32-layer DNA-SSM recurrence
- cached DNA-generated `A/B/C/G`
- precomputed `A_sig` and `G_silu`
- token embedding lookup
- stateful 32-layer Triton kernel
- Triton RMS normalization plus tied full-logits projection
- CUDA Graph replay

Best required result:

- amp: `fp16`
- output: full logits
- CUDA Graph: true
- mean: `13.75 us`
- p50: `13.95 us`
- p99: `14.62 us`
- p99.9: `18.08 us`
- max: `25.86 us`
- input tok/s: `72,743`

This meets the 003H targets:

- mean `10-15 us`: yes
- p99 `<50 us`: yes
- p99.9 `<100 us`: yes

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

```text
95 passed in 4.68s
```

Tests verify:

- cached DNA params match generated params
- original stateful full logits match a reference recurrence
- output shape is full logits `[256]`
- no fallback under `--force-triton`
- CUDA Graph smoke path reports successful graph usage
- parameter count remains `216,320`

## CUDA / Triton Environment

- device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- torch: 2.11.0+cu128
- CUDA: 12.8
- Triton available: true
- parameter_count: `216,320`
- layers: `32`
- d_model: `256`
- vocab_size: `256`
- batch_size: `1`
- seq_len: `1`

## Results

| amp | cuda_graph | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | full_logits | fallback_used | targets_met |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| fp16 | true | 13.75 | 13.95 | 14.62 | 18.08 | 25.86 | 72743 | true | false | true |
| bf16 | true | 13.77 | 13.95 | 14.62 | 16.26 | 26.53 | 72631 | true | false | true |
| fp16 | false | 18.11 | 19.07 | 21.28 | 65.28 | 153.89 | 55224 | true | false | p99/p999 only |

CUDA Graph worked for both fp16 and bf16. It improved fp16 mean latency from `18.11 us` to `13.75 us` and reduced tail latency substantially.

## Comparison vs 003G

| path | architecture | mean_us | p99_us | p999_us | max_us | notes |
|---|---|---:|---:|---:|---:|---|
| 003G precomposed_stateless_32_fused_e2e | stateless speed ablation | 7.65 | 8.48 | 11.87 | 18.82 | Removes hidden state and precomposes stateless depth |
| 003H original_stateful_32_fused_e2e | original stateful recurrence | 13.75 | 14.62 | 18.08 | 25.86 | Preserves `h[32,256]` recurrence |

003H recovers much of the 003G end-to-end speed while preserving the original stateful 32-layer recurrence. It is about `1.8x` slower than the 003G precomposed stateless fused path, which is the cost of preserving the stateful architecture and running the 32-layer recurrence at inference time.

## Bottleneck Analysis

The 003H path uses three GPU launches per inference step:

1. embedding lookup via `torch.index_select`
2. stateful 32-layer Triton kernel
3. Triton RMS normalization plus tied projection to 256 full logits

DNA generation and activation preprocessing are cached and not in the timed per-token path.

Remaining bottlenecks:

- the original stateful 32-layer recurrence
- persistent `h[32,256]` state updates
- separate launch for embedding lookup
- separate launch for RMS/projection

Unlike 003G, this does not use stateless/precomposed shortcuts.

## Conclusion

The original stateful Fast32 end-to-end path meets the requested latency targets:

- mean in target range: yes
- p99 under `50 us`: yes
- p99.9 under `100 us`: yes
- full logits produced: yes
- CUDA Graph used: yes
- fallback used: no
- parameter count: `216,320`

This is not a 1000-layer result. It is an architecture-preserving 32-layer Fast32 inference result.

## Next Step

The next optimization is a truly fused original-stateful kernel that combines:

- embedding lookup
- 32-layer stateful recurrence
- RMS normalization
- tied projection

That could reduce the three-launch path toward the 003G single-launch fused latency while preserving the original recurrence.
