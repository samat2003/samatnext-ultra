# Experiment 004: Tiny Shakespeare Fast32 Health Results

Status: completed bounded health check.

This experiment starts from `fast32-frozen-v1` and trains only the frozen `original_stateful_32` architecture as a learning health check.

## Configuration

- Base tag: `fast32-frozen-v1`
- Branch: `exp004-tiny-shakespeare-health`
- Main model: `original_stateful_32`
- Parameter count: `216,320`
- `vocab_size`: `256`
- `d_model`: `256`
- Runtime layers: `32`
- Architecture frozen: yes
- Frozen artifacts modified: no

## Dataset

- Dataset requested: `tiny_shakespeare`
- Dataset names tried: `tiny_shakespeare`, `karpathy/tiny_shakespeare`
- Dataset actually used: `karpathy/tiny_shakespeare`
- Encoding: UTF-8 bytes

## Overfit One Batch

Command:

```bash
python scripts/train_tiny_shakespeare_fast32.py \
  --device cuda \
  --dataset-name tiny_shakespeare \
  --seq-len 128 \
  --batch-size 16 \
  --steps 300 \
  --eval-every 50 \
  --lr 3e-4 \
  --amp fp16 \
  --overfit-one-batch \
  --out-dir checkpoints/tiny_shakespeare_fast32/overfit
```

Result:

- first_loss: `5.9524335861206055`
- final_loss: `2.8827364444732666`
- loss_decreased: yes
- tokens/sec: `1592.7490805553864`
- checkpoint: `checkpoints/tiny_shakespeare_fast32/overfit/best.pt`

Train loss curve:

| step | loss |
|---:|---:|
| 1 | 5.9524335861206055 |
| 50 | 3.5337071418762207 |
| 100 | 3.0603456497192383 |
| 150 | 2.9886832237243652 |
| 200 | 2.9710988998413086 |
| 250 | 2.9522364139556885 |
| 300 | 2.8827364444732666 |

## Small Train Run

The requested 1000-step command was started, but stopped at the user's request because it was taking too long in the PyTorch training path. The run had already produced a valid `best.pt` checkpoint at step 400.

Partial command:

```bash
python scripts/train_tiny_shakespeare_fast32.py \
  --device cuda \
  --dataset-name tiny_shakespeare \
  --seq-len 128 \
  --batch-size 32 \
  --steps 1000 \
  --eval-every 100 \
  --lr 3e-4 \
  --amp fp16 \
  --out-dir checkpoints/tiny_shakespeare_fast32/small_run
```

Result at interruption:

- completed through checkpoint step: `400`
- best validation loss: `3.0079738795757294`
- checkpoint: `checkpoints/tiny_shakespeare_fast32/small_run/best.pt`
- approximate tokens/sec at step 400: `2717.147698050648`

Train loss curve:

| step | train_loss |
|---:|---:|
| 1 | 5.9306254386901855 |
| 100 | 3.1510696411132812 |
| 200 | 3.0641088485717773 |
| 300 | 3.082186222076416 |
| 400 | 3.009464740753174 |

Validation loss curve:

| step | val_loss |
|---:|---:|
| 1 | 5.883140504360199 |
| 100 | 3.151858925819397 |
| 200 | 3.0714968144893646 |
| 300 | 3.055682420730591 |
| 400 | 3.0079738795757294 |

## Generation Smoke Test

Command:

```bash
python scripts/eval_tiny_shakespeare_fast32.py \
  --checkpoint checkpoints/tiny_shakespeare_fast32/small_run/best.pt \
  --device cuda \
  --prompt "ROMEO:" \
  --max-new-bytes 300 \
  --temperature 0.8
```

Generated sample:

```text
ROMEO:


;RDI



dnil'noI
lleehitrrrhh r o ousise nngr   drohtherso luffouowttnnedgdcathttsmmeedeeebp  ea er hha  hiiw de   yaeeeelueieiaro  nwee rrn dgi th

AAiw e  eoaunost  eoodn   tircw smahhistt rhwh mtllm, ihonie   oooeasaearlptheeeadnnes,eemythae  no lllceOARAthieaiooaifo lrei!

Uwwae   yott ssttl 
```

This is a generation smoke test only. It shows byte-level text-like output after a short run, not quality.

## Tests

- `python -m pytest -q`: `103 passed in 9.67s`

## Limitations

- This is a Tiny Shakespeare health/learning check only.
- This does not claim ChatGPT-like ability.
- This does not claim useful code ability.
- This does not claim general language modeling quality.
- The `precomposed_stateless_32` speed ablation is not the main training target.
- The 1000-step small run was intentionally stopped at step 400 to avoid spending excessive wall-clock time.
- Training uses the architecture-preserving PyTorch model path, not the fused inference-only kernels.
