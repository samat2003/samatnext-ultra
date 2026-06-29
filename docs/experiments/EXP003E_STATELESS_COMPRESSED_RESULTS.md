# Experiment 003E: Stateless / Compressed-State Thinking Kernel Results

## Summary

Experiment 003E tested architectural variants for batch-size-1, seq_len=1 thinking latency:

- `stateless_x_only`: no hidden state
- `shared_state_d`: one persistent `h[d]`
- `compressed_state_r{1,4,8,16,32}`: compressed persistent state `[r, d_model]`
- latency-floor kernels: empty Python loop, CUDA event no-op, empty Triton, vector-only Triton

These variants are not identical to the original Dynamic DNA-SSM. They are latency ablations. The original model and parameter budget are preserved as baselines, and parameter count reporting stayed `216,320`.

The fastest 1000-layer batch-size-1 kernel-only result was:

- variant: `stateless_x_only`
- layout: `combined_coeff`
- block_d: `256`
- CUDA Graph: true
- mean: `155.53 us`
- p50: `136.96 us`
- p99: `583.04 us`
- p99.9: `599.39 us`
- max: `619.58 us`
- thinking_layer_token_updates/sec: `6.43M`

This did not beat the 003C mean latency baseline by a meaningful amount and did not reach `100M thinking_layer_token_updates/sec`.

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

```text
61 passed in 4.60s
```

Correctness tests compare Triton outputs against PyTorch references for:

- `stateless_x_only`
- `shared_state_d`
- `compressed_state_r`
- packed and combined-coefficient stateless layouts
- layer counts including `1`, `2`, `8`, `32`, and `1000`

## CUDA / Triton Environment

- device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- torch: 2.11.0+cu128
- CUDA: 12.8
- Triton available: true
- main optimized paths used Triton: true
- fallback used in optimized paths: false
- parameter_count: `216,320`

## Latency Floor Audit

| variant | cuda_graph | mean_us | p50_us | p99_us | p999_us | max_us | std_us | cv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| empty_python | false | 0.11 | 0.10 | 0.19 | 0.66 | 11.70 | 0.14 | 1.27 |
| event_noop | false | 3.16 | 2.11 | 29.28 | 48.67 | 69.38 | 7.13 | 2.26 |
| empty_triton | false | 7.64 | 5.89 | 48.10 | 78.91 | 124.13 | 7.89 | 1.03 |
| vector_only | false | 7.45 | 5.86 | 45.66 | 72.83 | 89.82 | 7.36 | 0.99 |
| empty_triton | true | 6.23 | 5.38 | 42.37 | 67.39 | 83.74 | 5.62 | 0.90 |
| vector_only | true | 6.32 | 5.44 | 43.36 | 67.90 | 114.14 | 5.89 | 0.93 |

The empty Python number is host-loop overhead only. The CUDA event no-op and empty Triton measurements show that sub-1us GPU kernel timing is not meaningful here for this benchmark style. CUDA Graph replay floor is still about `6.2 us` mean with p99 above `40 us`.

Conclusion: `<=1 us` mean latency is not physically plausible on this setup for a measured Triton inference step using this timing method.

## Prior Baselines

| experiment | variant | mean_us | p99_us | p999_us | max_us | thinking_layer_token_updates_s |
|---|---|---:|---:|---:|---:|---:|
| EXP003C | fixed seq_len=1 stateless-in-kernel SSM | 154.25 | 155.23 | 166.75 | 187.58 | 6482849 |
| EXP003D | stateful single-token, CUDA Graph | 176.06 | 639.71 | 1117.76 | 1381.92 | 5679809 |

003D proved the bottleneck was not 1000 per-layer launches. The 003C and 003D optimized paths were already one Triton launch per inference step.

## Stateless 1000-Layer Results

`stateless_x_only` formula:

```text
x = x + residual_scale * silu(G_i) * C_i * (A_sig_i + B_i) * x
```

This removes hidden state entirely and is an architectural variant, not the original stateful SSM.

| variant | layers | amp | block_d | layout | cuda_graph | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s | target_1us | target_100m_ltu |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| stateless_x_only | 1000 | fp16 | 256 | separate_layers_d | false | 181.85 | 154.27 | 645.80 | 1239.18 | 1338.62 | 5499 | 5498954 | false | false |
| stateless_x_only | 1000 | fp16 | 256 | separate_layers_d | true | 185.03 | 155.04 | 1022.91 | 1226.98 | 1524.22 | 5405 | 5404504 | false | false |
| stateless_x_only | 1000 | fp16 | 256 | combined_coeff | false | 165.13 | 137.02 | 595.20 | 1623.33 | 1658.59 | 6056 | 6055816 | false | false |
| stateless_x_only | 1000 | fp16 | 256 | combined_coeff | true | 160.16 | 137.02 | 597.35 | 1167.30 | 1306.75 | 6244 | 6243929 | false | false |
| stateless_x_only | 1000 | fp16 | 256 | combined_coeff | true | 155.53 | 136.96 | 583.04 | 599.39 | 619.58 | 6430 | 6429778 | false | false |

The final row is the fastest row from the block/layout autotune. It used `combined_coeff`, which precomputes:

```text
coeff_i = silu(G_i) * C_i * (A_sig_i + B_i)
```

outside the kernel-only timed path. That reduces parameter streaming from four vectors per layer to one coefficient vector per layer.

## Compressed-State Results

| variant | state_size | layers | cuda_graph | mean_us | p99_us | p999_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| shared_state_d | 256 | 1000 | false | 193.02 | 1039.68 | 1157.26 | 5181 | 5180725 |
| shared_state_d | 256 | 1000 | true | 178.82 | 644.61 | 1240.88 | 5592 | 5592229 |
| compressed_state_r1 | 256 | 1000 | false | 190.00 | 1045.09 | 1265.96 | 5263 | 5263067 |
| compressed_state_r1 | 256 | 1000 | true | 187.72 | 1057.88 | 1326.47 | 5327 | 5327154 |
| compressed_state_r4 | 1024 | 1000 | false | 1435.72 | 4877.88 | 8873.20 | 697 | 696514 |
| compressed_state_r4 | 1024 | 1000 | true | 1735.47 | 3550.44 | 3763.82 | 576 | 576214 |
| compressed_state_r8 | 2048 | 1000 | false | 680.98 | 1593.93 | 1833.03 | 1468 | 1468474 |
| compressed_state_r8 | 2048 | 1000 | true | 669.95 | 1587.74 | 1804.87 | 1493 | 1492643 |
| compressed_state_r16 | 4096 | 1000 | false | 465.73 | 1251.56 | 1440.65 | 2147 | 2147161 |
| compressed_state_r16 | 4096 | 1000 | true | 483.10 | 1270.34 | 1505.19 | 2070 | 2069946 |
| compressed_state_r32 | 8192 | 1000 | false | 380.45 | 1513.82 | 2280.07 | 2628 | 2628471 |
| compressed_state_r32 | 8192 | 1000 | true | 365.91 | 1280.09 | 1892.96 | 2733 | 2732897 |

Compressed-state variants reduce hidden-state traffic versus 003D but still add enough register pressure and state movement to lose to `stateless_x_only`.

## Layer Sweep

Fastest stateless layout, `combined_coeff`, CUDA Graph, `block_d=256`.

| layers | variant | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | latency_ratio_vs_1_layer |
|---:|---|---:|---:|---:|---:|---:|
| 1 | stateless_x_only | 6.24 | 43.04 | 160367 | 160367 | 1.00 |
| 2 | stateless_x_only | 6.30 | 44.64 | 158736 | 317473 | 1.01 |
| 4 | stateless_x_only | 6.10 | 42.46 | 163903 | 655611 | 0.98 |
| 8 | stateless_x_only | 6.21 | 42.82 | 161005 | 1288041 | 1.00 |
| 16 | stateless_x_only | 6.55 | 45.02 | 152691 | 2443058 | 1.05 |
| 32 | stateless_x_only | 6.61 | 44.80 | 151322 | 4842310 | 1.06 |
| 64 | stateless_x_only | 7.13 | 43.62 | 140241 | 8975404 | 1.14 |
| 128 | stateless_x_only | 24.17 | 59.81 | 41377 | 5296300 | 3.88 |
| 256 | stateless_x_only | 42.04 | 112.64 | 23788 | 6089705 | 6.74 |
| 512 | stateless_x_only | 78.37 | 218.08 | 12760 | 6533002 | 12.57 |
| 1000 | stateless_x_only | 167.20 | 595.90 | 5981 | 5981047 | 26.81 |

The floor dominates through roughly 64 layers. Past that, parameter streaming and recurrence dependency dominate.

## BLOCK_D / Layout Autotune

All rows use CUDA Graph and `layers=1000`.

| variant | block_d | layout | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---|---:|---:|---:|---:|
| stateless_x_only | 16 | separate_layers_d | 182.86 | 633.63 | 5469 | 5468760 |
| stateless_x_only | 32 | separate_layers_d | 191.02 | 1031.94 | 5235 | 5235031 |
| stateless_x_only | 64 | separate_layers_d | 176.63 | 638.11 | 5661 | 5661494 |
| stateless_x_only | 128 | separate_layers_d | 178.29 | 644.77 | 5609 | 5608866 |
| stateless_x_only | 256 | separate_layers_d | 177.37 | 640.58 | 5638 | 5637815 |
| stateless_x_only | 16 | packed_layers_d_param | 173.74 | 601.22 | 5756 | 5755755 |
| stateless_x_only | 32 | packed_layers_d_param | 175.02 | 611.01 | 5714 | 5713530 |
| stateless_x_only | 64 | packed_layers_d_param | 174.13 | 630.37 | 5743 | 5742915 |
| stateless_x_only | 128 | packed_layers_d_param | 173.28 | 625.79 | 5771 | 5771162 |
| stateless_x_only | 256 | packed_layers_d_param | 214.11 | 1052.71 | 4671 | 4670516 |
| stateless_x_only | 16 | combined_coeff | 165.58 | 587.20 | 6039 | 6039253 |
| stateless_x_only | 32 | combined_coeff | 162.85 | 578.66 | 6141 | 6140550 |
| stateless_x_only | 64 | combined_coeff | 165.14 | 590.02 | 6056 | 6055508 |
| stateless_x_only | 128 | combined_coeff | 163.61 | 591.78 | 6112 | 6112142 |
| stateless_x_only | 256 | combined_coeff | 155.53 | 583.04 | 6430 | 6429778 |

Fastest layout: `combined_coeff`.

Fastest block size: `BLOCK_D=256`.

Coalescing rationale: `combined_coeff` streams one `[layers, d_model]` coefficient matrix instead of four separate A/B/C/G-like matrices. For this architecture variant, reducing parameter memory traffic helped more than packing four parameters adjacent in `[layers, d_model, param]`.

## CUDA Graph Impact

| variant | layers | mean_us_no_graph | mean_us_graph | speedup | p99_us_no_graph | p99_us_graph |
|---|---:|---:|---:|---:|---:|---:|
| empty_triton | 1 | 7.64 | 6.23 | 1.23 | 48.10 | 42.37 |
| vector_only | 1 | 7.45 | 6.32 | 1.18 | 45.66 | 43.36 |
| stateless_x_only separate | 1000 | 181.85 | 185.03 | 0.98 | 645.80 | 1022.91 |
| stateless_x_only combined | 1000 | 165.13 | 160.16 | 1.03 | 595.20 | 597.35 |
| shared_state_d | 1000 | 193.02 | 178.82 | 1.08 | 1039.68 | 644.61 |

CUDA Graph helps the latency floor and some variants modestly, but it does not solve the 1000-layer target. For the fastest stateless combined-coeff path, graph replay improved mean latency only about `1.03x`.

## End-to-End Timing

End-to-end mode includes token embedding, DNA A/B/C/G generation, coefficient preprocessing, Triton kernel, RMS normalization, and tied output projection/logits. CUDA Graph was requested by the command, but disabled and reported because DNA generation allocates tensors.

| variant | layers | layout | cuda_graph_requested | cuda_graph_used | mean_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| stateless_x_only | 1000 | combined_coeff | true | false | 814.47 | 562.10 | 2588.83 | 6729.64 | 9484.13 | 1228 | 1227790 |

## Bottleneck Analysis

Hidden-state traffic removed versus 003D:

- 003D stateful: reads/writes persistent `h[1000, 256]`, or 256,000 state values per token.
- `stateless_x_only`: reads/writes no hidden state.
- `shared_state_d`: reads/writes 256 state values per token.
- `compressed_state_r32`: reads/writes 8,192 state values per token.

Removing hidden-state traffic did help conceptually, but it did not produce top-tier latency because the remaining costs are large:

- CUDA/Triton launch and event-measurement floor is about `6 us` even with CUDA Graph.
- 1000-layer stateless still has a strict recurrence over depth.
- Separate layout streams four parameter vectors per layer.
- Combined-coeff layout still streams 1000 x 256 coefficients per token.
- Register pressure and occupancy limit how much of the channel vector can stay hot.
- Tail latency is dominated by launch/runtime jitter, not just arithmetic.

The stretch target is missed:

- mean `<=1 us`: no
- p99 `<=5 us`: no
- p99.9 `<=10 us`: no
- p99 `<50 us`: no for 1000 layers
- p99.9 `<100 us`: no for 1000 layers
- `100M thinking_layer_token_updates/sec`: no

Best 1000-layer result: `6.43M thinking_layer_token_updates/sec`, a `15.6x` gap to `100M`.

## Conclusion

`<=1 us` is not physically plausible on this setup for this measured Triton path. The CUDA Graph replay floor alone is around `6.2 us`, and a true 1000-effective-layer stateless recurrence lands around `155.5 us` mean even after removing hidden-state traffic and precomputing one combined coefficient per layer/channel.

The fastest variant was `stateless_x_only` with `combined_coeff`, `BLOCK_D=256`, and CUDA Graph replay. It is architecturally weaker than the original SSM because it removes hidden state.

## Next Step

The next bottleneck is the 1000-step depth recurrence plus coefficient streaming, not persistent hidden-state traffic. Useful follow-ups:

- reduce effective depth through a mathematically fused product/scan for the stateless coefficient recurrence
- precompose layer coefficients into fewer equivalent blocks
- test a lower-depth or logarithmic-depth thinking variant as an explicit architecture ablation
- use a custom CUDA kernel only if it can reduce jitter or improve coefficient streaming beyond Triton

Do not pursue training speed or transformer baselines from this experiment.
