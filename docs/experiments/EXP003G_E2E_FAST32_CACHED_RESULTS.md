# Experiment 003G: End-to-End Fast32 Cached Inference Results

## Summary

Experiment 003G optimized end-to-end Fast32 inference by caching DNA-generated parameters and adding fused precomposed-stateless inference kernels.

Fastest true end-to-end full-logits result:

- variant: `precomposed_stateless_32_fused_e2e`
- amp: `fp16`
- projection: `fused`
- output_mode: `full_logits`
- CUDA Graph: true
- mean: `7.65 us`
- p50: `7.84 us`
- p99: `8.48 us`
- p99.9: `11.87 us`
- max: `18.82 us`
- input tok/s: `130,760`

Fastest top-1-only result:

- variant: `precomposed_stateless_32_fused_e2e`
- amp: `fp16`
- projection: `fused`
- output_mode: `top1_only`
- CUDA Graph: true
- mean: `11.27 us`
- p99: `12.32 us`

Top-1-only was slower than full logits because the one-kernel top1 reduction over 256 vocab entries is heavier than writing the 256 logits in this implementation.

The fastest path is not the original stateful SSM. It is a precomposed stateless speed ablation.

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

```text
87 passed in 4.50s
```

Tests cover:

- cached DNA params match generated params
- cached master coefficient matches 32 stateless updates
- fused logits match unfused PyTorch logits
- fused top1 matches full-logit argmax
- original stateful cached path matches reference recurrence
- deterministic fused output
- parameter count remains `216,320`

## CUDA / Triton Environment

- device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- torch: 2.11.0+cu128
- CUDA: 12.8
- Triton available: true
- parameter_count: `216,320`
- batch_size: `1`
- seq_len: `1`
- layers: `32`
- d_model: `256`
- vocab_size: `256`

## Prior 003F Baseline

| path | mean_us | p99_us | p999_us | max_us | notes |
|---|---:|---:|---:|---:|---|
| precomposed stateless kernel-only | 5.77 | 31.17 | 48.70 | 54.30 | Kernel-only, no embedding/projection |
| original stateful 32 kernel-only | 10.77 | 30.40 | 49.18 | 68.42 | Original recurrence, `h[32,256]` |
| precomposed stateless uncached end-to-end | 776.50 | 2087.58 | 2984.31 | 12773.28 | Prior 003F run; included DNA generation/projection |

The 003G uncached reproduction measured `560.41 us` mean. It was lower than the prior `776.50 us` run but showed the same bottleneck: DNA generation and allocation-heavy PyTorch postprocessing dominate.

## End-to-End Component Breakdown

Component timings were measured independently with CUDA events and can add instrumentation overhead. On cached paths, component percentages can exceed 100% because timing individual PyTorch/Triton operations changes the launch structure relative to the captured total.

| variant | component | mean_us | p50_us | p99_us | p999_us | max_us | percent_of_total_mean | allocates | cuda_graph_compatible | kernel_launches |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| precomposed_stateless_32_e2e | dna_generation | 295.18 | 297.74 | 671.50 | 1413.85 | 1473.79 | 52.7 | true | false | 3 |
| precomposed_stateless_32_e2e | tied_projection_logits | 24.04 | 24.16 | 25.00 | 67.94 | 69.41 | 4.3 | true | false | 1 |
| precomposed_stateless_32_e2e | rms_normalization | 22.23 | 22.18 | 30.14 | 34.53 | 34.88 | 4.0 | true | false | 3 |
| precomposed_stateless_32_e2e | master_coeff_precomposition | 18.10 | 18.08 | 18.72 | 51.11 | 53.12 | 3.2 | true | false | 1 |
| precomposed_stateless_32_e2e | activation_preprocessing | 16.12 | 16.03 | 22.47 | 24.03 | 24.45 | 2.9 | true | false | 2 |
| precomposed_stateless_32_e2e | thinking_kernel | 9.47 | 9.92 | 10.72 | 10.89 | 23.62 | 1.7 | false | true | 1 |
| precomposed_stateless_32_fused_e2e | fused_embedding_thinking_rms_projection | 7.66 | 7.87 | 8.48 | 8.67 | 8.67 | 100.2 | false | true | 1 |
| precomposed_stateless_32_fused_e2e | fused_embedding_thinking_rms_top1 | 11.18 | 11.55 | 17.18 | 17.66 | 18.30 | 99.3 | false | true | 1 |

Old end-to-end time went mainly to:

- DNA generation: about `295 us` mean in component profiling
- allocation-heavy RMS/projection/preprocessing: about `16-24 us` each when measured separately
- launch and allocation overhead across many small operations

## Allocation Audit

| variant | allocations_in_timed_path | graph_capture_failure_reason | peak_cuda_memory_bytes | notes |
|---|---|---|---:|---|
| precomposed_stateless_32_e2e | true | none | 10912256 | Uncached DNA/preprocessing and PyTorch projection allocate |
| original_stateful_32_cached_params | true | none | 10978304 | PyTorch RMS/projection allocate, but CUDA Graph capture succeeded |
| precomposed_stateless_32_cached_master | true | none | 10912256 | PyTorch RMS/projection allocate, but CUDA Graph capture succeeded |
| precomposed_stateless_32_fused_e2e full_logits | false | none | 10912256 | Cached master, fused full logits, no timed allocation reported |
| precomposed_stateless_32_fused_e2e top1_only | false | none | 10912256 | Cached master, fused top1-only, no timed allocation reported |

Caching removed per-token DNA generation and per-token coefficient preprocessing. Precomputing `master_coeff` removed per-token stateless coefficient composition.

## Cached End-to-End Results

| variant | projection | output_mode | cuda_graph | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | allocations_in_timed_path |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| precomposed_stateless_32_e2e | pytorch | full_logits | false | 560.41 | 499.20 | 1064.27 | 1698.63 | 2538.18 | 1784 | true |
| original_stateful_32_cached_params | pytorch | full_logits | true | 21.33 | 21.76 | 22.53 | 24.32 | 32.32 | 46889 | true |
| precomposed_stateless_32_cached_master | pytorch | full_logits | true | 16.14 | 16.13 | 16.99 | 19.94 | 28.64 | 61963 | true |
| precomposed_stateless_32_cached_master | triton | full_logits | true | 8.56 | 8.22 | 9.95 | 12.29 | 19.36 | 116801 | false |
| precomposed_stateless_32_fused_e2e | fused | full_logits | true | 7.65 | 7.84 | 8.48 | 11.87 | 18.82 | 130760 | false |
| precomposed_stateless_32_fused_e2e | fused | top1_only | true | 11.27 | 11.65 | 12.32 | 13.98 | 17.79 | 88766 | false |

Fastest true end-to-end latency is `7.65 us` mean with full logits.

Fastest full-logits latency is also `7.65 us`.

Fastest top1-only latency is `11.27 us`.

## Original vs Precomposed

| variant | original_stateful | stateless | precomposed | mean_us | p99_us | architecture_sacrifice |
|---|---|---|---|---:|---:|---|
| original_stateful_32_cached_params | true | false | false | 21.33 | 22.53 | None; original recurrence with cached params |
| precomposed_stateless_32_cached_master | false | true | true | 16.14 | 16.99 | Removes hidden state and precomposes stateless depth |
| precomposed_stateless_32_fused_e2e | false | true | true | 7.65 | 8.48 | Removes hidden state, precomposes depth, fuses output path |

The fused path is much faster, but it is architecturally weaker than the original stateful SSM.

## Projection Optimization

| projection | output_mode | mean_us | p99_us | p999_us | max_us | notes |
|---|---|---:|---:|---:|---:|---|
| pytorch | full_logits | 16.14 | 16.99 | 19.94 | 28.64 | Cached master, PyTorch RMS/projection |
| triton | full_logits | 8.56 | 9.95 | 12.29 | 19.36 | Cached master, Triton RMS/projection after thinking kernel |
| fused | full_logits | 7.65 | 8.48 | 11.87 | 18.82 | Embedding + master multiply + RMS + logits in one Triton launch |
| fused | top1_only | 11.27 | 12.32 | 13.98 | 17.79 | Top1-only one-kernel reduction; slower than full logits |

Fused full logits helped. Top-1-only did not help in this implementation.

## CUDA Graph Impact

| variant | mean_us_no_graph | mean_us_graph | speedup | p99_no_graph | p99_graph |
|---|---:|---:|---:|---:|---:|
| original_stateful_32_cached_params pytorch | 50.54 | 21.33 | 2.37 | 145.32 | 22.53 |
| precomposed_stateless_32_cached_master pytorch | 50.72 | 16.14 | 3.14 | 160.87 | 16.99 |
| precomposed_stateless_32_fused_e2e full_logits | 7.74 | 7.65 | 1.01 | 8.51 | 8.48 |
| precomposed_stateless_32_fused_e2e top1_only | 11.37 | 11.27 | 1.01 | 12.42 | 12.32 |

CUDA Graph was valuable for PyTorch-heavy cached paths. It mattered little once the path was already a single fused Triton launch.

## Bottleneck Analysis

The old end-to-end bottleneck was DNA generation plus many allocation-heavy small operations. Caching removes the DNA and coefficient preprocessing from the per-token path. Fusing removes the PyTorch RMS/projection overhead.

After fusion, the remaining bottleneck is mostly the single fused Triton launch and the 256x256 tied projection work. The fused full-logits path is only about `1.33x` slower than the 003F `5.77 us` kernel-only precomposed vector multiply.

The top1-only path is slower because the current top1 kernel computes and reduces all 256 logits inside one heavy Triton program. Writing 256 full logits with parallel vocab blocks is faster.

## Conclusion

003G reduced end-to-end latency from the prior `776.50 us` class down to `7.65 us` for fused full logits.

Best results:

- fastest true end-to-end: `7.65 us` mean
- fastest full logits: `7.65 us` mean
- fastest top1-only: `11.27 us` mean
- parameter count: `216,320`
- CUDA Graph worked on cached paths
- fused projection helped substantially
- top1-only did not help

The fastest path is `precomposed_stateless_32_fused_e2e`, which is a speed ablation, not the original stateful SSM.

## Next Step

The next optimization target is the fused projection/top1 kernel:

- keep full logits as the default because it is faster here
- optimize top1 with a two-stage reduction or better warp-level layout
- test caching/fusing the original stateful path if quality preservation matters
- measure sampling separately only after full-logits inference is stable
