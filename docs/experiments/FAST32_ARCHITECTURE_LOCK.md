# Fast32 Architecture Lock

Architecture changes are frozen after commit `1c8e488b3453f100e59c6b545f4ec646b3e27eca`.

Future work must branch from a tag created from the frozen commit. Training must not modify architecture. Benchmark comparisons must cite this frozen version and must distinguish:

- `precomposed_stateless_32_fused_e2e`: fastest speed ablation.
- `original_stateful_32`: architecture-preserving Fast32 stateful recurrence.

No training was run as part of the freeze.
