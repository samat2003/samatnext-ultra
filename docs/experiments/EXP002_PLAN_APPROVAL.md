# Experiment 002 Plan Approval

Status: codex_plan_filled_waiting_for_user_approval

## Proposed Files

Implementation files to create/edit after approval:

- `samatnext_dna_ssm/__init__.py` - export the experiment model/config helpers.
- `samatnext_dna_ssm/model.py` - implement `DynamicDnaSsmConfig`, DNA hypernetwork, chunked causal SSM forward, ACT metadata, and parameter counting.
- `scripts/exp002_smoke.py` - small CLI smoke/benchmark script for parameter count, causality, no-grad 1M-layer chunking, `--max-chunks`, and short training runs.
- `tests/test_exp002_model.py` - focused tests for shapes, parameter budget, no stored per-layer modules, chunked execution metadata, and causality.
- `docs/experiments/EXP002_WALKTHROUGH.md` - implementation walkthrough written after code and commands are run.

This approval step edits only this file.

## Architecture Plan

Model class:

- Add a compact PyTorch language model, tentatively `DynamicDnaSsmLM`, configured by `DynamicDnaSsmConfig`.
- Defaults: `vocab_size=256`, `d_model=256`, `max_layers=1_000_000`, `chunk_size=1_000`.
- Inputs: integer token IDs shaped `[batch, seq]`.
- Outputs: logits shaped `[batch, seq, vocab_size]` plus metadata with `layers_used`, `chunks_used`, `halt_score`, and `halted`.

Trainable components:

- Token embedding: `nn.Embedding(256, 256)`.
- DNA hypernetwork: a small MLP that maps a continuous, parameter-free layer index embedding to generated vectors `A`, `B`, `C`, `G`.

Output projection:

- Use tied output projection with the token embedding weights: logits are computed from hidden states and `token_embed.weight.T`.
- No separate LM head module.
- No LM-head bias.

Layer index embedding:

- Use deterministic continuous Fourier features of normalized layer index `i / max_layers`.
- No learned per-layer table.
- No `ModuleList` of generated layers.

DNA hypernetwork:

- Input shape per generated layer: `[layer_embed_dim]`, planned `layer_embed_dim=16`.
- MLP: `Linear(16, 128) -> SiLU -> Linear(128, 128) -> SiLU -> Linear(128, 4 * d_model)`.
- Output is reshaped/split into `A`, `B`, `C`, `G`, each `[chunk_layers, d_model]`.
- It never outputs `[d_model, d_model]` matrices.
- Generated SSM tensors are ephemeral chunk tensors, not stored parameters.

Chunked MonoForward:

- Iterate `start_layer` from `0` to `max_layers` in increments of `chunk_size`.
- Generate only the current chunk's `A/B/C/G`, with shape `[current_chunk_size, d_model]` per vector.
- Apply those generated layers sequentially to the sequence.
- Delete/drop references to generated chunk tensors before the next chunk.
- Evaluate ACT once per chunk and stop when the halt score crosses the configured threshold, or when `max_layers` is reached.

Causal ephemeral SSM layer:

- For each generated layer `i`, run a left-to-right scan over token positions.
- State `h` is initialized to zeros for each batch item and layer.
- Update:

```text
h_t = sigmoid(A_i) * h_{t-1} + B_i * x_t
y_t = C_i * h_t
x_t = x_t + silu(G_i) * y_t
```

- Broadcasting is only over batch and sequence dimensions; generated vectors remain `[d_model]`.
- No generated dense matrices and no stored per-layer parameters.

ACT / dynamic halting:

- V0 halts only at chunk boundaries.
- Use a parameter-free chunk stability halt score.
- Before each chunk, keep the current hidden sequence tensor as the chunk input reference.
- After the chunk, compute a stability score from the mean absolute hidden-state delta, for example `delta = mean(abs(x_after - x_before))`.
- Convert that delta to a bounded halt score without trainable parameters, for example `halt_score = 1 / (1 + delta)`, where smaller chunk changes mean a higher halt score.
- Stop if `halt_score >= halt_threshold` after at least `min_chunks` chunks, or if a smoke-test `max_chunks` cap is reached.
- Return:
  - `layers_used`: actual number of generated/applied layers.
  - `chunks_used`: number of chunks executed.
  - `halt_score`: final scalar or batch-mean scalar.
  - `halted`: boolean indicating the stability halt stopped before exhausting `max_layers`.

## Parameter Count Formula

Planned default parameter estimate:

```text
Token embedding:
  vocab_size * d_model
  256 * 256 = 65,536

LM head:
  tied to token embedding
  no separate parameters, no bias = 0

DNA hypernetwork:
  Linear(16, 128): 16 * 128 + 128 = 2,176
  Linear(128, 128): 128 * 128 + 128 = 16,512
  Linear(128, 1024): 128 * (4 * 256) + 1024 = 132,096
  DNA total = 150,784

ACT / halt score:
  parameter-free chunk stability score = 0

Estimated total:
  65,536 + 0 + 150,784 + 0 = 216,320
```

This leaves about `783,680` parameters of margin under the `1,000,000` limit.

The estimate intentionally excludes generated SSM layers because they are ephemeral tensors produced during forward, not trainable stored parameters. The implementation will include a parameter counting helper and tests asserting total trainable parameters are `< 1,000,000`.

## VRAM Plan

Chunk generation:

- Never allocate `[max_layers, ...]` generated SSM tensors.
- At default settings, each generated chunk contains four `[1000, 256]` tensors: `1,024,000` scalar values.
- In fp32 this is about `4.1 MB` for raw generated `A/B/C/G` chunk values, plus intermediate activations and model state.
- After the chunk is applied and ACT is evaluated, references to generated chunk tensors are discarded before the next chunk.

Inference / `torch.no_grad()`:

- The 1M-layer smoke path will run under `torch.no_grad()` with small batch/sequence sizes.
- It should avoid max-layer-sized tensors and keep generated layer tensor memory proportional to `chunk_size`, not `max_layers`.
- Runtime may still be very slow because up to 1,000,000 recurrent layers are actually applied unless ACT halts early.
- Add a smoke option such as `--max-chunks` so the no-grad VRAM/chunking test can force exactly 2-3 chunks without running all 1,000,000 layers.
- The 1M-layer no-grad test with early halt or `--max-chunks` proves the code path does not require max-layer-sized generated allocations. It does not prove full non-halting 1M-layer execution speed or memory over the entire depth.

Training:

- V0 will not claim O(1) VRAM training through 1M layers.
- Normal PyTorch autograd stores graph/activation history through the executed depth, so training memory grows with `layers_used` unless checkpointing/rematerialization is implemented.
- V0 plan does not include full rematerialized 1M-layer training.
- Training scripts/tests will use small `max_layers` values and report the configured depth, update cadence, and optimizer updates honestly.

VRAM reporting:

- If CUDA is available, use `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()` for peak VRAM.
- If CUDA is unavailable, report CPU-only and skip VRAM claims.

## Causality Plan

Future-token mutation test:

- Put the model in eval mode.
- Create a token batch `x` with sequence length `T`.
- Clone it to `x_mut`.
- Choose a cutoff `k`, then mutate only future positions `k+1:T` in `x_mut`.
- Run both inputs through the model with identical configuration and deterministic settings.
- Compare logits for prefix positions `0:k+1`.
- Assert `max_abs_diff(prefix_logits_original, prefix_logits_mutated) <= tolerance`.

This directly checks that output at position `t <= k` does not depend on future tokens. The causal scan is left-to-right and uses only `h_{t-1}` and `x_t`, so a suffix mutation should not affect prefix outputs. Dropout will not be used in the default model to keep this test deterministic.

## UE4 Training Plan

Training loop behavior:

- Compute forward and loss every step for both UE1 and UE4.
- UE1: `update_every = 1`; run backward and optimizer step every training step.
- UE4: `update_every = 4`; run backward and optimizer step only when `step % 4 == 0` using the experiment's specified rule.
- Report both `update_every` and `optimizer_updates`.

Important limitation:

- This is not gradient accumulation across four forward passes unless explicitly implemented later. Per the UE4 rule, V0 will simply skip backward/optimizer work on non-update steps while still reporting losses every step.

## Commands To Run

After approval, run the following commands from the repository root:

```bash
python -m compileall samatnext_dna_ssm scripts tests
python -m pytest -q
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 4 --update-every 1
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 8 --update-every 4
python scripts/exp002_smoke.py --device cpu --max-layers 1000000 --chunk-size 1000 --seq-len 4 --batch-size 1 --no-grad-only --max-chunks 3
```

If CUDA is available, also run:

```bash
python scripts/exp002_smoke.py --device cuda --max-layers 1000 --chunk-size 1000 --seq-len 16 --batch-size 2 --no-grad-only
```

The 1M-layer command uses `--max-chunks 3` for a fast memory-path check that forces multiple chunks while avoiding full non-halting 1M-layer execution. This proves the smoke path can run with `max_layers=1_000_000` without allocating generated tensors for every layer at once; it does not prove full 1M-layer execution.

## Risks / Limitations

- A Python-level nested loop over layers and sequence positions is simple and auditable but slow. This experiment prioritizes correctness and reproducibility over throughput.
- ACT can halt early only at chunk boundaries in V0, so `layers_used` granularity is `chunk_size`.
- The generated SSM update is diagonal/vector-only by requirement; it is much less expressive than a dense learned SSM transition.
- The DNA hypernetwork may generate unstable dynamics. `sigmoid(A)` bounds recurrence retention, but `B/C/G` may still need conservative initialization or clipping if training is unstable.
- Training through large executed depths is not O(1) VRAM in V0. Any 1M-layer training claim would require checkpointing/rematerialization and separate measurement.
- Full 1M-layer non-halting inference is compute-heavy even when generated tensors are chunked, because every layer is still applied.
- CPU smoke tests can validate behavior, but they do not prove practical GPU throughput or large-batch training viability.

## Approval Status

Status: waiting_for_user_approval
