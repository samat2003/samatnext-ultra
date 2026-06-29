# Experiment 002 Walkthrough

Status: implemented

## Files Changed

- `samatnext_dna_ssm/__init__.py` - exports the Experiment 002 model, config, and output dataclass.
- `samatnext_dna_ssm/model.py` - implements `DynamicDnaSsmLM`, chunked DNA-generated causal SSM vectors, tied token-embedding output projection, parameter counting, and parameter-free stability halting.
- `scripts/exp002_smoke.py` - adds the smoke/benchmark CLI, including `--max-chunks`, UE1/UE4 update cadence, causality measurement, hidden/logit RMS reporting, byte-corpus training, and CPU/CUDA memory reporting.
- `tests/test_exp002_model.py` - covers parameter count, absence of separate LM/ACT heads, chunk metadata, generated vector shapes, max-chunk execution, and future-token causality.
- `docs/experiments/EXP002_WALKTHROUGH.md` - records actual implementation results.

## Architecture Implemented

- Defaults: `vocab_size=256`, `d_model=256`, `max_layers=1_000_000`, `chunk_size=1_000`.
- Trainable token embedding only for vocabulary projection. There is no separate LM head and no LM-head bias.
- Logits use tied projection through `F.linear(hidden, token_embed.weight)`, which applies `token_embed.weight.T`.
- Hidden states are RMS-normalized before the tied projection: `x / sqrt(mean(x^2) + eps)`.
- DNA MLP is `Linear(16,128) -> SiLU -> Linear(128,128) -> SiLU -> Linear(128,4*d_model)`.
- Generated SSM outputs are split into `A`, `B`, `C`, `G`, each shaped `[chunk_layers, d_model]`.
- SSM residual updates use `residual_scale=0.01`.
- Token embeddings and final DNA projection are initialized conservatively to avoid initial logit/state explosions.
- No generated dense `[d_model, d_model]` matrices are used.
- No `ModuleList` of layers is used.
- Generated SSM vectors are produced one chunk at a time and discarded before the next chunk.
- ACT V0 halting is parameter-free: `halt_score = 1 / (1 + mean(abs(x_after - x_before)))`, evaluated at chunk boundaries.

## Commands Run

Installed missing local dependencies first because the virtualenv did not have `torch` or `pytest`:

```bash
python -m pip install pytest torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Required commands:

```bash
python -m compileall samatnext_dna_ssm scripts tests
python -m pytest -q
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 4 --update-every 1
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 8 --update-every 4
python scripts/exp002_smoke.py --device cpu --max-layers 1000000 --chunk-size 1000 --seq-len 4 --batch-size 1 --no-grad-only --max-chunks 3
```

Numerical stabilization reruns:

```bash
python -m pytest -q
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 20 --update-every 1
python scripts/exp002_smoke.py --device cpu --max-layers 8 --chunk-size 4 --seq-len 16 --batch-size 2 --train-steps 20 --update-every 4
```

Optional CUDA command was not run because `torch.cuda.is_available()` returned `False`.

## Results

Compile:

- `python -m compileall samatnext_dna_ssm scripts tests` passed.

Tests:

- `python -m pytest -q` passed: `6 passed`.
- Warning observed: PyTorch reported NumPy was not installed. This did not affect the tests or smoke runs.

Parameter count:

- Reported trainable parameters: `216,320`.
- Budget result: under `1,000,000`.

Causality:

- Future-token mutation smoke result: `causality_max_abs_diff = 0.0` in all smoke runs.
- Unit causality test also passed with prefix logits unchanged after suffix mutation.

VRAM / chunking:

- CPU-only environment, so CUDA peak VRAM was not available and `peak_vram_bytes` was `null`.
- No-grad 1M configured smoke used `max_layers=1,000,000`, `chunk_size=1,000`, and `--max-chunks 3`.
- Actual no-grad chunking result: `chunks_used=3`, `layers_used=3000`, `halt_score=0.9965213537216187`, `elapsed_sec=0.09199233100025594`.
- This verifies the 1M configured path can execute multiple chunks without generating all 1,000,000 layers at once. It does not prove full non-halting 1M-layer execution.

Training / speed:

- Initial UE1 command result:
  - `update_every=1`
  - `train_steps=4`
  - `optimizer_updates=4`
  - `layers_used_last=8`
  - `chunks_used_last=2`
  - `first_loss=250.47503662109375`
  - `last_loss=261.6123962402344`
  - `elapsed_sec=0.07195457599982547`

- Initial UE4 command result:
  - `update_every=4`
  - `train_steps=8`
  - `optimizer_updates=2`
  - `layers_used_last=8`
  - `chunks_used_last=2`
  - `first_loss=250.47503662109375`
  - `last_loss=260.8984069824219`
  - `elapsed_sec=0.08146698399968955`

UE4 followed the required rule: forward/loss every step, backward/optimizer only when `step % 4 == 0`.

Numerical stabilization rerun results:

- UE1 byte-corpus command result:
  - `update_every=1`
  - `train_steps=20`
  - `optimizer_updates=20`
  - `first_loss=5.940623760223389`
  - `last_loss=5.9557294845581055`
  - `hidden_rms_last=0.020133910700678825`
  - `logits_rms_last=1.5618746280670166`
  - `causality_max_abs_diff=0.0`
  - `elapsed_sec=0.18215277600029367`

- UE4 byte-corpus command result:
  - `update_every=4`
  - `train_steps=20`
  - `optimizer_updates=5`
  - `first_loss=5.940623760223389`
  - `last_loss=5.97138786315918`
  - `hidden_rms_last=0.019076578319072723`
  - `logits_rms_last=0.5004057884216309`
  - `causality_max_abs_diff=0.0`
  - `elapsed_sec=0.15613967399985995`

## Overfit Sanity Test

Command:

```bash
python scripts/exp002_smoke.py \
  --device cpu \
  --max-layers 8 \
  --chunk-size 4 \
  --seq-len 32 \
  --batch-size 4 \
  --train-steps 300 \
  --update-every 1
```

Result:

- `first_loss`: `5.777731895446777`
- `last_loss`: `2.9097657203674316`
- `optimizer_updates`: `300`
- `hidden_rms_last`: `0.06373050808906555`
- `logits_rms_last`: `5.166011333465576`
- `elapsed_sec`: `5.573469758999636`

Interpretation:

- Loss clearly decreased on the tiny built-in byte corpus, so this run provides a basic learnability sanity check for the accepted V0 structure.
- This is still not a quality benchmark; it only shows the model can overfit a small deterministic byte stream under the smoke settings.

## Limitations

- The implementation uses Python loops over layers and sequence positions, so it is correctness-oriented and slow for large non-halting depths.
- The no-grad 1M smoke used `--max-chunks 3`; it proves chunked allocation behavior, not full 1M-layer runtime.
- CUDA VRAM was not measured because CUDA is unavailable in this environment.
- Training through huge depth is not O(1) VRAM in V0. No checkpointing/rematerialization is implemented.
- ACT halting is only at chunk boundaries.
- The SSM is diagonal/vector-only by design and less expressive than dense generated transitions.
- CPU smoke losses are short built-in byte-corpus sanity checks, not a quality benchmark.
