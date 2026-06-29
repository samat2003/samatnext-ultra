# Experiment 003F: Fast 32-Layer Thinking Variant Results

## Summary

Experiment 003F pivots from the fixed 1000-layer latency target to a speed-first 32-layer thinking variant.

The fastest 32-layer kernel-only result was:

- variant: `precomposed_stateless_32`
- amp: `fp16`
- CUDA Graph: true
- mean: `5.77 us`
- p50: `5.41 us`
- p99: `31.17 us`
- p99.9: `48.70 us`
- max: `54.30 us`
- input tok/s: `173,337`
- thinking_layer_token_updates/sec: `5.55M`

This meets the 32-layer latency targets:

- mean `<10 us`: yes
- p99 `<50 us`: yes
- p99.9 `<100 us`: yes

It is not the original stateful SSM. It is a precomposed stateless architectural speed ablation where the 32 stateless coefficients are multiplied into one `master_coeff` outside the kernel-only timed path.

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

```text
76 passed in 5.47s
```

Tests cover:

- output/state shapes
- original stateful 32-layer Triton vs PyTorch reference
- stateless 32-layer Triton vs PyTorch reference
- precomposed stateless output vs running stateless updates
- shared-state and compressed-state variants
- layer counts `1`, `2`, `8`, `16`, `32`
- parameter count remains `216,320`

## CUDA / Triton Environment

- device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- torch: 2.11.0+cu128
- CUDA: 12.8
- Triton available: true
- main optimized paths used Triton: true
- fallback used: false
- trainable parameter count: `216,320`

## Prior Baselines

| experiment | variant | layers | mean_us | p99_us | p999_us | thinking_layer_token_updates_s |
|---|---|---:|---:|---:|---:|---:|
| EXP003C | fixed seq_len=1 kernel-only | 1000 | 154.25 | 155.23 | 166.75 | 6482849 |
| EXP003D | stateful single-token CUDA Graph | 1000 | 176.06 | 639.71 | 1117.76 | 5679809 |
| EXP003E | stateless combined coeff CUDA Graph | 1000 | 155.53 | 583.04 | 599.39 | 6429778 |

003F results are 32-layer thinking variant results. They must not be read as 1000-layer results.

## 32-Layer Kernel-Only Results

| variant | layers | amp | cuda_graph | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s | mean_under_10us | p99_under_50us | p999_under_100us |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| original_stateful_32 | 32 | fp16 | false | 10.74 | 9.15 | 33.82 | 56.23 | 77.54 | 93127 | 2980073 | false | true | true |
| original_stateful_32 | 32 | fp16 | true | 10.77 | 9.95 | 30.40 | 49.18 | 68.42 | 92856 | 2971407 | false | true | true |
| stateless_32 | 32 | fp16 | false | 6.81 | 5.89 | 30.78 | 54.72 | 64.26 | 146815 | 4698087 | true | true | true |
| stateless_32 | 32 | fp16 | true | 6.17 | 5.60 | 28.90 | 48.74 | 69.76 | 162089 | 5186851 | true | true | true |
| precomposed_stateless_32 | 32 | fp16 | false | 6.88 | 5.86 | 31.30 | 54.53 | 66.05 | 145247 | 4647905 | true | true | true |
| precomposed_stateless_32 | 32 | fp16 | true | 5.77 | 5.41 | 31.17 | 48.70 | 54.30 | 173337 | 5546779 | true | true | true |
| shared_state_32 | 32 | fp16 | false | 10.99 | 9.25 | 34.62 | 57.18 | 76.61 | 90992 | 2911736 | false | true | true |
| shared_state_32 | 32 | fp16 | true | 10.86 | 9.98 | 31.42 | 48.61 | 59.74 | 92067 | 2946129 | false | true | true |
| compressed_state_32_r1 | 32 | fp16 | false | 10.84 | 9.38 | 32.32 | 50.05 | 60.80 | 92214 | 2950856 | false | true | true |
| compressed_state_32_r1 | 32 | fp16 | true | 10.89 | 9.98 | 32.19 | 49.79 | 63.36 | 91814 | 2938038 | false | true | true |
| compressed_state_32_r4 | 32 | fp16 | false | 38.71 | 37.50 | 61.34 | 78.27 | 93.28 | 25835 | 826714 | false | false | true |
| compressed_state_32_r4 | 32 | fp16 | true | 37.79 | 36.74 | 58.78 | 73.76 | 85.44 | 26462 | 846781 | false | false | true |
| compressed_state_32_r8 | 32 | fp16 | false | 24.29 | 23.17 | 46.37 | 60.35 | 76.03 | 41163 | 1317218 | false | true | true |
| compressed_state_32_r8 | 32 | fp16 | true | 23.77 | 22.62 | 42.82 | 58.85 | 90.11 | 42074 | 1346381 | false | true | true |
| compressed_state_32_r16 | 32 | fp16 | false | 18.92 | 17.47 | 39.39 | 58.66 | 71.33 | 52844 | 1690995 | false | true | true |
| compressed_state_32_r16 | 32 | fp16 | true | 17.21 | 16.22 | 37.47 | 51.10 | 61.82 | 58096 | 1859065 | false | true | true |

## Original vs Stateless vs Precomposed

| variant | state_size | runtime_depth | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| original_stateful_32 | 8192 | 32 | 10.77 | 30.40 | 92856 | 2971407 | Original recurrence with `h[32,256]` |
| stateless_32 | 0 | 32 | 6.17 | 28.90 | 162089 | 5186851 | No hidden state; runs 32 stateless updates |
| precomposed_stateless_32 | 0 | 1 kernel vector multiply | 5.77 | 31.17 | 173337 | 5546779 | Precomposes 32 stateless coefficients outside kernel-only timing |

The original stateful 32-layer path is much faster than the 1000-layer stateful path, but it does not reach mean `<10 us`. The stateless and precomposed variants meet the latency targets but sacrifice the original stateful recurrence.

## Shared / Compressed State Results

| variant | state_size | mean_us | p99_us | p999_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---:|---:|---:|---:|---:|
| shared_state_32 | 256 | 10.86 | 31.42 | 48.61 | 92067 | 2946129 |
| compressed_state_32_r1 | 256 | 10.89 | 32.19 | 49.79 | 91814 | 2938038 |
| compressed_state_32_r4 | 1024 | 37.79 | 58.78 | 73.76 | 26462 | 846781 |
| compressed_state_32_r8 | 2048 | 23.77 | 42.82 | 58.85 | 42074 | 1346381 |
| compressed_state_32_r16 | 4096 | 17.21 | 37.47 | 51.10 | 58096 | 1859065 |

The rank-4/8/16 compressed kernels have more per-program state handling overhead than expected and lose to shared-state and original stateful for this implementation.

## Layer Sweep

Variant: `precomposed_stateless_32`, CUDA Graph, fp16. These are effective layer counts folded into one precomputed `master_coeff`; they are not runtime recurrent depth.

| layers | variant | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | latency_ratio_vs_32 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | precomposed_stateless_32 | 6.00 | 31.78 | 166669 | 166669 | 1.01 |
| 8 | precomposed_stateless_32 | 5.87 | 30.27 | 170450 | 1363602 | 0.99 |
| 16 | precomposed_stateless_32 | 5.88 | 32.03 | 170174 | 2722781 | 0.99 |
| 32 | precomposed_stateless_32 | 5.92 | 30.94 | 168868 | 5403776 | 1.00 |
| 64 | precomposed_stateless_32 | 6.96 | 32.48 | 143678 | 9195369 | 1.18 |
| 128 | precomposed_stateless_32 | 6.81 | 32.16 | 146859 | 18797902 | 1.15 |
| 1000 | precomposed_stateless_32 | 6.64 | 31.10 | 150542 | 150542048 | 1.12 |

The apparent 1000-layer LTU/s in this sweep is an effective-layer accounting number for precomposed stateless coefficients. It is not a 1000-step runtime recurrence.

## CUDA Graph Impact

| variant | layers | mean_us_no_graph | mean_us_graph | speedup | p99_us_no_graph | p99_us_graph |
|---|---:|---:|---:|---:|---:|---:|
| original_stateful_32 | 32 | 10.74 | 10.77 | 1.00 | 33.82 | 30.40 |
| stateless_32 | 32 | 6.81 | 6.17 | 1.10 | 30.78 | 28.90 |
| precomposed_stateless_32 | 32 | 6.88 | 5.77 | 1.19 | 31.30 | 31.17 |
| shared_state_32 | 32 | 10.99 | 10.86 | 1.01 | 34.62 | 31.42 |

CUDA Graph helps most on the lowest-work variants, where launch overhead is the dominant cost.

## BF16 Comparison

CUDA Graph, kernel-only, layers=32.

| variant | amp | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s |
|---|---|---:|---:|---:|---:|
| original_stateful_32 | bf16 | 11.60 | 31.81 | 86236 | 2759549 |
| stateless_32 | bf16 | 7.42 | 31.94 | 134849 | 4315157 |
| precomposed_stateless_32 | bf16 | 6.88 | 31.26 | 145272 | 4648711 |

fp16 was faster than bf16 for the fastest variants on this GPU.

## End-to-End Timing

End-to-end mode includes token embedding, DNA generation, coefficient preprocessing, Triton thinking kernel, RMS normalization, and tied output projection/logits. CUDA Graph was requested by the benchmark command but disabled because DNA generation allocates tensors.

| variant | layers | amp | cuda_graph_requested | cuda_graph_used | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| precomposed_stateless_32 | 32 | fp16 | true | false | 776.50 | 610.74 | 2087.58 | 2984.31 | 12773.28 | 1288 | 41210 |

End-to-end latency is dominated by DNA generation and tied output projection, not the 32-layer kernel.

## Bottleneck Analysis

For the fastest kernel-only path, runtime recurrence is eliminated by precomposition. The remaining bottleneck is the launch/vector floor plus CUDA event/runtime jitter.

For original stateful 32:

- one Triton launch per inference step
- true 32 runtime layers
- `h[32,256]` state
- mean around `10.77 us`
- p99 below `50 us`

For stateless 32:

- one Triton launch per inference step
- 32 runtime stateless updates
- no hidden state
- mean around `6.17 us`

For precomposed stateless 32:

- one Triton launch per inference step
- one vector multiply in the timed kernel
- no runtime depth recurrence
- mean around `5.77 us`, close to the 003E empty/vector Triton floor around `6 us`

The remaining end-to-end bottleneck is DNA generation, coefficient preprocessing, and output projection.

## Conclusion

The speed-first 32-layer pivot works for kernel-only latency:

- fastest variant: `precomposed_stateless_32`
- best mean latency: `5.77 us`
- p99: `31.17 us`
- p99.9: `48.70 us`
- max: `54.30 us`
- mean `<10 us`: yes
- p99 `<50 us`: yes
- p99.9 `<100 us`: yes

The original stateful 32-layer variant also meets p99 and p99.9 targets, but misses mean `<10 us` narrowly at `10.77 us`.

The best result is architecturally weaker than the original SSM because it removes state and precomposes depth. It should be treated as a speed ablation, not as a quality-preserving replacement.

## Next Step

The next step is to attack end-to-end latency:

- cache DNA-generated coefficients for fixed 32-layer inference
- precompute `master_coeff` once per model/config instead of per token
- fuse embedding, master multiply, RMS norm, and tied projection where practical
- separately measure sampling/logit selection if this becomes an interactive inference target

Do not return to training speed or transformer baselines from this experiment.
