# Experiment 003B Triton 1000-Layer Results

Status: implemented_measured

## Scope

Experiment 003B implements a forward-only Triton kernel for the fixed generated SSM chunk. It does not change `DynamicDnaSsmLM` architecture, add attention, add FlashAttention, add a transformer baseline, add stored per-layer parameters, add a separate LM head, or add trainable ACT.

The target path is:

- `max_layers=1000`
- `chunk_size=1000`
- `d_model=256`
- fixed depth, no ACT early halt in the timed Triton path
- DNA hypernetwork generates ephemeral `A/B/C/G` in PyTorch
- Triton applies the generated vectors to `x`

## Correctness

Command:

```bash
python -m pytest -q
```

Result:

- `14 passed`

The Triton tests cover:

- small-shape PyTorch-vs-Triton comparisons for `layers=4`, `16`, and `64`
- `d_model=16` and `32` small cases
- `d_model=256`, `layers=1000`, small sequence correctness
- causal prefix invariance
- skip behavior if CUDA/Triton is unavailable

## CUDA / Triton Availability

- CUDA available: yes
- Triton available: yes
- Device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- Torch: `2.11.0+cu128`
- CUDA runtime: `12.8`

## Kernel

The implemented Triton path takes:

```text
x:      [batch, seq, d_model]
A_sig:  [layers, d_model]
B:      [layers, d_model]
C:      [layers, d_model]
G_silu: [layers, d_model]
out:    [batch, seq, d_model]
```

The benchmark precomputes:

```python
A_sig = torch.sigmoid(A)
G_silu = torch.nn.functional.silu(G)
```

V0 kernel limits:

- The whole tested sequence must fit in one `BLOCK_SEQ` tile.
- It tiles over batch and channel block.
- It streams `A/B/C/G` by layer and channel block.
- It uses Triton `tl.range` loops, not fully unrolled `tl.static_range`, to avoid codegen stalls for 1000 layers.

## Benchmarks

### Batch 1, Seq 16, FP32

Command:

```bash
python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 1 \
  --seq-len 16 \
  --d-model 256 \
  --layers 1000 \
  --amp fp32
```

Results:

- PyTorch reference input tok/s: `22.918096109296133`
- Triton input tok/s: `5696.845906167386`
- PyTorch layer-token-updates/sec: `22918.096109296133`
- Triton layer-token-updates/sec: `5696845.906167385`
- Triton speedup: `248.57413456157937`
- `max_abs_error`: `0.0`
- `mean_abs_error`: `0.0`
- PyTorch peak CUDA memory bytes: `16648192`
- Triton peak CUDA memory bytes: `18691072`
- Tile sizes used: `BLOCK_SEQ=16`, `BLOCK_D=32`
- 100M input tok/s reached: `false`
- 100M layer-token-updates/sec reached: `false`
- True fixed 1000-layer execution: `true`

### Batch 4, Seq 32, BF16

Command:

```bash
python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 4 \
  --seq-len 32 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16
```

Results:

- PyTorch reference input tok/s: `61.23788328842867`
- Triton input tok/s: `4640.188654581962`
- PyTorch layer-token-updates/sec: `61237.88328842867`
- Triton layer-token-updates/sec: `4640188.654581961`
- Triton speedup: `75.77317185714611`
- `max_abs_error`: `0.0`
- `mean_abs_error`: `0.0`
- PyTorch peak CUDA memory bytes: `12753920`
- Triton peak CUDA memory bytes: `12743680`
- Tile sizes used: `BLOCK_SEQ=32`, `BLOCK_D=32`
- 100M input tok/s reached: `false`
- 100M layer-token-updates/sec reached: `false`
- True fixed 1000-layer execution: `true`

### Batch 16, Seq 64, BF16

Command:

```bash
python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 16 \
  --seq-len 64 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16
```

Results:

- PyTorch reference input tok/s: `260.4813622296764`
- Triton input tok/s: `11348.819289843628`
- PyTorch layer-token-updates/sec: `260481.36222967642`
- Triton layer-token-updates/sec: `11348819.289843628`
- Triton speedup: `43.568642273288404`
- `max_abs_error`: `0.0`
- `mean_abs_error`: `0.0`
- PyTorch peak CUDA memory bytes: `14619648`
- Triton peak CUDA memory bytes: `14578688`
- Tile sizes used: `BLOCK_SEQ=64`, `BLOCK_D=32`
- 100M input tok/s reached: `false`
- 100M layer-token-updates/sec reached: `false`
- True fixed 1000-layer execution: `true`

### Batch 32, Seq 128, BF16

Command:

```bash
python scripts/exp003b_triton_ssm_bench.py \
  --device cuda \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 256 \
  --layers 1000 \
  --amp bf16
```

Results:

- PyTorch reference input tok/s: `449.29015007679783`
- Triton input tok/s: `30185.41961445625`
- PyTorch layer-token-updates/sec: `449290.1500767978`
- Triton layer-token-updates/sec: `30185419.614456248`
- Triton speedup: `67.18469035944058`
- `max_abs_error`: `5.820766091346741e-11`
- `mean_abs_error`: `7.090682208055199e-17`
- PyTorch peak CUDA memory bytes: `20952064`
- Triton peak CUDA memory bytes: `20870144`
- Tile sizes used: `BLOCK_SEQ=128`, `BLOCK_D=32`
- 100M input tok/s reached: `false`
- 100M layer-token-updates/sec reached: `false`
- True fixed 1000-layer execution: `true`

## Parameter Count

- Trainable parameter count stayed `216,320`.
- Fixed 1000 generated layers did not add stored per-layer parameters.
- The Triton benchmark uses `DynamicDnaSsmLM.generate_chunk()` for `d_model=256` benchmark vectors, so `A/B/C/G` are generated ephemerally by the existing DNA hypernetwork.

## Interpretation

- The Triton kernel executes a true fixed 1000-layer path.
- The best measured Triton result was `30.19M` layer-token-updates/sec and `30.19k` input tok/s at batch `32`, sequence length `128`.
- `100M` input tok/s was not reached.
- `100M` layer-token-updates/sec was not reached.
- The first realistic target remains `100M` layer-token-updates/sec, not `100M` input tok/s.
- The V0 Triton kernel is a large improvement over the eager Python/PyTorch loop, but still below target.
- FlashAttention remains irrelevant for this path because there is no attention/QKV operation.

## Next Recommendation

Optimize Experiment 003B before considering a transformer baseline:

- benchmark `BLOCK_D=16`, `32`, `64`
- benchmark smaller/larger `BLOCK_SEQ` where sequence carry requirements allow it
- profile occupancy, register pressure, and A/B/C/G streaming
- consider splitting layers into subchunks if the 1000-layer loop limits occupancy
- later, design a backward/rematerialization strategy before claiming training speed
