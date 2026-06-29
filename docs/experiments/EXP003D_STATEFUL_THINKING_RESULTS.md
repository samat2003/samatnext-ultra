# Experiment 003D: Stateful Single-Token Thinking Kernel Results

## Summary

Experiment 003D added a stateful single-token Triton kernel for fixed-depth DNA-SSM thinking. It executes a true generated-layer path with persistent hidden state `h[layers, d_model]`.

The optimized path is exactly one Triton launch per inference step for the SSM kernel. It removes Python loops, DNA generation, A/B/C/G preprocessing, allocation, embedding, logits projection, loss, backward, optimizer, and training from the kernel-only timed path.

The result did not improve the Experiment 003C batch-size-1 baseline. Best measured 1000-layer stateful kernel-only mean latency was `176.06 us` with CUDA Graph replay, versus the 003C stateless seq_len=1 kernel-only baseline of about `154 us`.

Reason: true statefulness introduces persistent hidden-state traffic. The kernel reads and writes `h[layer, d]` for every layer and channel each token. The 003C stateless single-token path kept the per-layer hidden state transient inside the kernel and avoided storing the full `[1000, 256]` state.

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

```text
32 passed in 2.85s
```

The tests compare the stateful Triton kernel to the PyTorch stateful reference for:

- `layers=1`
- `layers=2`
- `layers=8`
- `layers=32`
- `layers=1000`
- `d_model=256`
- `fp32`, `fp16`, and `bf16`

Parameter count remained `216,320`.

## CUDA / Triton Environment

- device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- torch: 2.11.0+cu128
- CUDA: 12.8
- Triton available: true
- Triton used in optimized path: true
- fallback used in optimized path: false

## Baseline From Experiment 003C

- fixed 1000-layer DNA-SSM
- parameter_count: `216,320`
- d_model: `256`
- vocab_size: `256`
- batch-size-1
- seq_len=1
- fp16 kernel-only mean latency: about `154 us`
- p99: about `155 us`
- p99.9: about `167 us`
- max: about `188 us`
- input tok/s: about `6,483`
- thinking_layer_token_updates/sec: about `6.48M`
- 1-layer fp16 mean latency: about `8.76 us`
- 1000-layer fp16 mean latency: about `155 us`
- latency ratio 1000 vs 1: about `17.7x`

Important instrumentation correction: the 003C fixed-depth Triton path already launches one Triton kernel per inference call. It was not doing 1000 separate per-layer kernel launches.

## Optimized Stateful Kernel

Kernel-only timed path:

- one Triton launch per inference step
- no Python loop inside the kernel-only compute path
- Python dispatch is included without CUDA Graph and excluded by CUDA Graph replay
- no allocation in the optimized kernel-only timed path
- no DNA generation in kernel-only timed path
- no A/B/C/G preprocessing in kernel-only timed path
- no embedding/logits projection in kernel-only timed path
- A/B/C/G are read from global memory each layer
- persistent hidden state has one read and one write per layer/channel per step

| implementation | layers | amp | block_d | cuda_graph | mean_us | std_us | p50_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s | p99_under_50us | target_100m_ltu_reached |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| stateful_single_token_triton | 1000 | fp16 | 256 | false | 195.49 | 159.37 | 158.40 | 1062.22 | 1301.57 | 1407.90 | 5115 | 5115465 | false | false |
| stateful_single_token_triton | 1000 | fp16 | 256 | true | 176.06 | 105.01 | 151.65 | 639.71 | 1117.76 | 1381.92 | 5680 | 5679809 | false | false |
| stateful_single_token_triton | 1000 | bf16 | 256 | false | 180.62 | 118.09 | 152.48 | 643.07 | 1298.92 | 1461.57 | 5537 | 5536500 | false | false |
| stateful_single_token_triton | 1000 | fp16 | 256 | false | 184.68 | 134.73 | 152.48 | 1020.23 | 1288.93 | 1394.85 | 5415 | 5414884 | false | false |

The repeated non-graph fp16 rows come from the required main benchmark and the block-D autotune run. The best mean was CUDA Graph fp16 at `176.06 us`; the best p50 was `151.65 us`, but tail latency stayed high.

## 1-Layer vs 1000-Layer

| implementation | layers | amp | cuda_graph | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | latency_ratio_vs_1_layer |
|---|---:|---|---|---:|---:|---:|---:|---:|
| stateful_single_token_triton | 1 | fp16 | false | 7.45 | 48.90 | 134284 | 134284 | 1.00 |
| stateful_single_token_triton | 1000 | fp16 | false | 182.76 | 1032.45 | 5472 | 5471534 | 24.54 |
| stateful_single_token_triton | 1 | fp16 | true | 6.25 | 44.48 | 160127 | 160127 | 1.00 |
| stateful_single_token_triton | 1000 | fp16 | true | 179.94 | 645.38 | 5557 | 5557461 | 28.81 |

The 1-layer result does not solve the 1000-layer target. For `layers=1`, `100M thinking_layer_token_updates/sec` would equal `100M input tok/s`, but this is only a shallow-kernel ceiling.

## Layer Sweep

Kernel-only, `amp=fp16`, `block_d=256`, no CUDA Graph.

| layers | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s | latency_ratio_vs_1_layer |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.51 | 50.24 | 133178 | 133178 | 1.00 |
| 2 | 7.26 | 49.47 | 137675 | 275351 | 0.97 |
| 4 | 7.19 | 47.84 | 139071 | 556282 | 0.96 |
| 8 | 8.51 | 50.56 | 117480 | 939842 | 1.13 |
| 16 | 9.46 | 49.28 | 105761 | 1692179 | 1.26 |
| 32 | 11.72 | 49.66 | 85313 | 2730006 | 1.56 |
| 64 | 17.37 | 51.01 | 57562 | 3683973 | 2.31 |
| 128 | 26.99 | 66.94 | 37049 | 4742279 | 3.59 |
| 256 | 52.86 | 132.83 | 18918 | 4843023 | 7.04 |
| 512 | 88.57 | 247.07 | 11290 | 5780687 | 11.80 |
| 1000 | 189.43 | 1156.81 | 5279 | 5279086 | 25.23 |

Depth scaling is far below 1000x, so there is still fixed overhead amortization. But beyond 256 layers, persistent state traffic and recurrence work dominate enough that the stateful path no longer beats the 003C stateless single-token kernel.

## Block-D Autotune

Kernel-only, `layers=1000`, `amp=fp16`, no CUDA Graph.

| block_d | mean_us | p99_us | input_tok_s | thinking_layer_token_updates_s |
|---:|---:|---:|---:|---:|
| 16 | 207.63 | 1078.18 | 4816 | 4816253 |
| 32 | 208.42 | 1073.96 | 4798 | 4798057 |
| 64 | 206.09 | 1082.63 | 4852 | 4852313 |
| 128 | 200.84 | 1080.32 | 4979 | 4979127 |
| 256 | 184.68 | 1020.23 | 5415 | 5414884 |

`BLOCK_D=256` was best in the full 10,000-iteration run.

## CUDA Graph Impact

| layers | mean_us_no_graph | mean_us_graph | speedup | p99_us_no_graph | p99_us_graph |
|---:|---:|---:|---:|---:|---:|
| 1000 | 195.49 | 176.06 | 1.11 | 1062.22 | 639.71 |

CUDA Graph reduced mean latency by about `1.11x` and improved p99, but did not get close to the p99 `<50 us` target. This means Python/Triton launch overhead is not the dominant bottleneck for the 1000-layer stateful kernel.

## End-to-End Timing

End-to-end mode includes token embedding, DNA `A/B/C/G` generation, A/G preprocessing, stateful Triton SSM, RMS output normalization, and tied output projection/logits. It does not include sampling.

| implementation | layers | amp | mean_us | p99_us | p999_us | max_us | input_tok_s | thinking_layer_token_updates_s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| stateful_single_token_triton | 1000 | fp16 | 809.16 | 2685.02 | 3347.60 | 3964.70 | 1236 | 1235851 |

End-to-end timing is much slower than kernel-only because it includes DNA generation and tied output projection.

## Bottleneck Analysis

The optimized stateful path did execute as one Triton launch per inference step. CUDA Graph replay reduced overhead, which confirms launch/Python overhead is present, but not enough to explain the 1000-layer latency.

The dominant new cost is persistent hidden-state memory traffic:

- `h` has shape `[1000, 256]`.
- Each token reads and writes 256,000 state elements.
- The kernel also streams A/B/C/G vectors from global memory each layer.
- The recurrence dependency prevents parallelizing across layers.
- Register pressure and occupancy are constrained by keeping active channel tiles live through the layer loop.

The 003C baseline was already one Triton launch and kept hidden state transient for `seq_len=1`; therefore, 003D's true statefulness adds memory traffic and regresses mean/tail latency.

Targets:

- p99 `<50 us`: not reached
- p99.9 `<100 us`: not reached
- `100M thinking_layer_token_updates/sec` for 1000-layer batch-size-1: not reached
- best 1000-layer batch-size-1 thinking_layer_token_updates/sec: about `5.68M`
- target gap: about `17.6x`

## Next Step

The next bottleneck is not per-layer kernel launch overhead. It is persistent-state traffic plus recurrence dependency.

The next useful ablation is to change the state representation, not the parameter budget:

- grouped/shared state across layer blocks
- lower-precision or packed hidden state
- state update only every `k` layers
- two-kernel design that separates hot transient micro-layers from sparse persistent state
- specialized CUDA, not Triton, if state traffic and occupancy cannot be improved enough

Any such change should be documented as an architectural/state ablation because it changes inference semantics, even if trainable parameters remain `216,320`.
