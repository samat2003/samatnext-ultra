# Experiment 003B Synthetic Train Speed Results

Status: implemented_measured

## Real Synthetic Training Speed

Configuration:

- `parameter_count`: `216320`
- `layers`: `1000`
- `chunk_size`: `1000`
- `d_model`: `256`
- `vocab_size`: `256`
- `fixed_depth`: `true`
- `dynamic_halt_used`: `false`
- `trainable_params_only_embedding_and_dna`: `true`
- `triton_available`: `true`
- `triton_supports_backward`: `false`
- `training_path_implementation`: `pytorch_autograd_dynamic_dna_ssm`

Training metric definition:

- `input training tok/s = batch_size * seq_len * measured_steps / train_elapsed_sec`
- This includes forward, loss, backward on cadence steps, optimizer step on cadence steps, and CUDA synchronization around timing.
- The cadence is called `mono-forward update cadence`: every step runs forward/loss, and backward/optimizer only runs when `step % update_every == 0`.
- Forward-only Triton speed is not reported as training speed.
- Layer-token-updates/sec is not reported as input training tok/s.

## Results Table

| amp | optimizer | update_every | batch_size | seq_len | train_input_tok_s | forward_loss_input_tok_s | layer_token_updates_s | optimizer_updates | forward_loss_calls | first_loss | last_loss | loss_decreased | peak_cuda_memory_bytes | fallback_used | triton_kernel_used | triton_supports_backward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| bf16 | fused-adamw | 1 | 4 | 32 | 13.39886440361257 | 80.18304647905083 | 13398.864403612572 | 5 | 5 | 5.80419921875 | 4.7891845703125 | true | 440579584 | false | false | false |
| bf16 | fused-adamw | 4 | 4 | 32 | 32.79624208769716 | 64.08592470956987 | 32796.24208769716 | 1 | 5 | 6.078369140625 | 5.80419921875 | true | 857959424 | false | false | false |
| bf16 | fused-adamw | 4 | 8 | 64 | 63.69419065247718 | 158.07518709094415 | 63694.19065247718 | 1 | 5 | 6.043701171875 | 5.847648620605469 | true | 3260977152 | false | false | false |
| fp16 | fused-adamw | 4 | 8 | 64 | 65.84581737073425 | 142.29325549757723 | 65845.81737073426 | 1 | 5 | 6.043793678283691 | 5.847456932067871 | true | 3260977152 | false | false | false |
| bf16 | adamw | 4 | 4 | 32 | 32.35441645266426 | 70.7941667122607 | 32354.416452664263 | 1 | 5 | 6.078369140625 | 5.80419921875 | true | 857955840 | false | false | false |
| fp16 | adamw | 4 | 4 | 32 | 34.45574113890947 | 61.78367370953327 | 34455.74113890947 | 1 | 5 | 6.078216552734375 | 5.80511474609375 | true | 857955840 | false | false | false |
| bf16 | fused-adamw | 4 | 4 | 32 | 25.28507510713116 | 69.48453577001982 | 25285.07510713116 | 12 | 50 | 6.078369140625 | 3.42431640625 | true | 857959424 | false | false | false |

The final row is the 50-step overfit-fixed-batch check.

## Required Commands

Correctness:

```bash
python -m pytest -q
```

Result: `14 passed`.

Real training speed tests:

```bash
python scripts/exp003b_train_speed.py \
  --device cuda \
  --batch-size 4 \
  --seq-len 32 \
  --layers 1000 \
  --chunk-size 1000 \
  --amp bf16 \
  --optimizer fused-adamw \
  --update-every 1 \
  --warmup-iters 1 \
  --measure-iters 5

python scripts/exp003b_train_speed.py \
  --device cuda \
  --batch-size 4 \
  --seq-len 32 \
  --layers 1000 \
  --chunk-size 1000 \
  --amp bf16 \
  --optimizer fused-adamw \
  --update-every 4 \
  --warmup-iters 1 \
  --measure-iters 5

python scripts/exp003b_train_speed.py \
  --device cuda \
  --batch-size 8 \
  --seq-len 64 \
  --layers 1000 \
  --chunk-size 1000 \
  --amp bf16 \
  --optimizer fused-adamw \
  --update-every 4 \
  --warmup-iters 1 \
  --measure-iters 5

python scripts/exp003b_train_speed.py \
  --device cuda \
  --batch-size 8 \
  --seq-len 64 \
  --layers 1000 \
  --chunk-size 1000 \
  --amp fp16 \
  --optimizer fused-adamw \
  --update-every 4 \
  --warmup-iters 1 \
  --measure-iters 5
```

Overfit check:

```bash
python scripts/exp003b_train_speed.py \
  --device cuda \
  --batch-size 4 \
  --seq-len 32 \
  --layers 1000 \
  --chunk-size 1000 \
  --amp bf16 \
  --optimizer fused-adamw \
  --update-every 4 \
  --train-iters 50 \
  --overfit-fixed-batch
```

Result:

- `first_loss`: `6.078369140625`
- `last_loss`: `3.42431640625`
- `loss_decreased`: `true`
- `optimizer_updates`: `12`
- `train_input_tok_s`: `25.28507510713116`

## Optimizer / Precision Notes

- 8-bit AdamW was not run because `bitsandbytes` is not installed: `ModuleNotFoundError: No module named 'bitsandbytes'`.
- `fused-adamw` worked with no fallback.
- On the larger tested shape, fp16 fused AdamW was slightly faster than bf16 fused AdamW: `65.85` vs `63.69` real training input tok/s.
- On the smaller standard AdamW comparison, fp16 was also faster than bf16: `34.46` vs `32.35` real training input tok/s.

## Interpretation

- Real training input tok/s is far lower than forward-only Triton throughput because the current Triton SSM kernel is forward-only and has no backward.
- The best measured real training input tok/s was `65.84581737073425` with fp16, fused AdamW, `batch_size=8`, `seq_len=64`, and `update_every=4`.
- Loss decreased in all measured synthetic training runs.
- The 50-step overfit-fixed-batch run decreased loss from `6.078369140625` to `3.42431640625`.
- Mono-forward update cadence `update_every=4` was faster than `update_every=1` on the same bf16 fused-AdamW batch 4 / seq 32 setup: `32.80` vs `13.40` input tok/s.
- `100M` real training input tok/s did not happen. The best measured value is about `1.52M` times below `100M`.
- `100M` layer-token-updates/sec did not happen in real training. The best measured value was `65,845.81737073426` layer-token-updates/sec.
- The run executed true fixed 1000 layers with `max_layers=1000`, `chunk_size=1000`, and `dynamic_halt_used=false`.
- Trainable parameter count stayed `216,320`.

## Next Bottleneck

The next bottleneck is backward/training support for the fused SSM path. The forward-only Triton kernel improves forward throughput, but real training still uses the PyTorch autograd loop. The next correction is a custom backward/rematerialized Triton or CUDA training path for the fixed 1000-layer SSM chunk.
