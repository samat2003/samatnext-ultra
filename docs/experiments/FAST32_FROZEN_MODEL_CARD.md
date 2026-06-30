# Fast32 Frozen Model Card

## What Was Frozen

Fast32 freezes the initialized Dynamic DNA-SSM inference artifacts after commit `1c8e488b3453f100e59c6b545f4ec646b3e27eca`.

- Trainable parameters: `216,320`
- `vocab_size`: `256`
- `d_model`: `256`
- Runtime depth for Fast32: `32`
- Tied output projection: yes
- Stored per-layer trainable parameters: no
- Separate LM head: no
- Trainable ACT head: no

Frozen files are stored under `checkpoints/fast32_frozen/`.

## Best Recorded Results

Speed-ablation result from Experiment 003G:

- Variant: `precomposed_stateless_32_fused_e2e`
- Full logits end-to-end: yes
- CUDA Graph: yes
- fp16 mean: `7.65 us`
- p99: `8.48 us`
- p99.9: `11.87 us`
- max: `18.82 us`
- input tok/s: `130,760`
- parameter count: `216,320`

Architecture-preserving result from Experiment 003H:

- Commit: `1c8e488b3453f100e59c6b545f4ec646b3e27eca`
- Variant: `original_stateful_32`
- Full logits end-to-end: yes
- CUDA Graph: yes
- fallback: no
- fp16 mean: `13.75 us`
- p99: `14.62 us`
- p99.9: `18.08 us`
- max: `25.86 us`
- input tok/s: `72,743`
- parameter count: `216,320`
- tests: `95 passed`

## Architecture Caveats

`precomposed_stateless_32_fused_e2e` is the fastest result, but it is an architectural speed ablation. It is not the original stateful SSM recurrence.

`original_stateful_32` is the architecture-preserving Fast32 result.

## Hardware And Software

The artifact metadata records the exact local Python, PyTorch, CUDA, Triton availability, device name, and git commit in `checkpoints/fast32_frozen/benchmark_metadata.json`.

## What Is Not Claimed

- This is not 1000-layer runtime inference.
- This is not trained model quality.
- This is not ChatGPT-like ability.
- This is not `100M` input tok/s.
- The `7.65 us` path is not the original stateful model.

## License

The frozen Fast32 artifacts are released under Apache-2.0 unless a dependency license says otherwise. Third-party dependencies keep their own licenses. See `LICENSE`, `NOTICE`, and `docs/experiments/FAST32_LICENSE_NOTES.md`.
