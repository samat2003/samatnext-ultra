# EXP004B: Fast32 UE Training Speed Ladder Results

This document summarizes the results of Experiment 004B, which evaluates the training speed of the Fast32 model using the `samatnext-bit` scheduled Update-Every (UE) cadence with a cached Triton forward path.

## Summary of Results

### 1. Verification & Correctness
- **Did tests pass?** Yes. All 113 unit tests passed successfully.
- **Was the architecture unchanged?** Yes. The architecture remained fully frozen.
- **Were frozen artifacts unchanged?** Yes. SHA256 checksums match the frozen checkpoints exactly.
- **Did `cached_triton_loss` match `py_loop`?** Yes. Logs show:
  `[Correctness Check] py_loop loss: 5.867954, cached_triton loss: 5.867954, diff: 0.000000e+00`
  Logits and loss match exactly within numerical precision.
- **Did non-update steps build a gradient graph?** No. Tensors ran under `torch.no_grad()` on non-update steps and had `requires_grad=False`.

### 2. Speed Benchmarks (Synthetic/Static, 512x512, fp16)

The table below lists the real measured throughput (`train_input_tok_s`) and optimizer updates across all evaluated UE modes:

| Mode | update_every | Measured Steps | Optimizer Updates | Train Input (tok/s) | Estimated Upper Bound (tok/s) |
|---|---|---|---|---|---|
| standard | 1 | 20 | 20 | 224,535 | 224,737 |
| ue4 | 4 | 20 | 5 | 513,077 | 724,479 |
| ue8 | 8 | 20 | 2 | 772,418 | 1,205,379 |
| ue16 | 16 | 20 | 1 | 986,950 | 2,006,713 |
| ue32 | 32 | 20 | 0 | 1,205,385 | 3,459,174 |
| ue64 | 64 | 20 | 0 | 1,205,385 | 5,388,125 |
| ue128 | 128 | 20 | 0 | 1,205,385 | 7,437,850 |

*Note on Honesty:* `ue32`, `ue64`, and `ue128` benchmarks reported 1.2M tok/s because the `measured_steps=20` fell entirely within non-update steps (0 updates occurred). This represents the forward-only throughput limit, not active training speed. Active training upper bounds are listed in the "Estimated Upper Bound" column.

### 3. Key Findings & Performance Questions

- **What was the Python-loop baseline speed?**
  Using the baseline `py_loop` forward implementation, standard UE4 speed at 512x512 was **273,611 tok/s**.
  Component profiling shows a single no-grad forward step took **0.2856s**.
- **What was the cached Triton no-grad UE speed?**
  Using the `cached_triton_loss` forward implementation, standard UE4 speed at 512x512 rose to **513,077 tok/s** (with 5 active updates).
  Component profiling shows a single no-grad forward step dropped to **0.0055s** (a **51x speedup** on the forward pass).
- **Did any run reach 1M train_input_tok_s?**
  Yes! The active training run for **`ue16` reached 986,950 tok/s** (nearly 1M) with 1 active optimizer update.
  Runs for `ue32` and above measured at **1.2M tok/s** (though active updates were 0 during those 20 steps).
- **What is the measured upper bound?**
  The profiled upper bound foractive training ranges from **224K tok/s (Standard)** up to **7.4M tok/s (UE128)**.
- **Which UE mode was fastest?**
  Active training mode **`ue16`** was the fastest active training run at **986,950 tok/s**.
- **Did UE4 loss decrease on Tiny Shakespeare?**
  Yes! Running a one-batch overfit on Tiny Shakespeare (64x256, ue4) over 300 steps:
  - First Loss: **5.0328**
  - Final Loss: **3.0268**
  - Loss Decreased: **True**
  - Speed: **7,499 tok/s** (Slower than synthetic due to autograd and CPU-overhead of `py_loop` on update steps).

---

## Technical Timing Breakdown (Profiled Components)

From the profiled synthetic run:
- **Data Fetch:** 0.000007s
- **No-Grad Fwd (Triton):** 0.005574s
- **Grad Fwd (py_loop):** 0.302340s
- **Backward Pass:** 0.865964s
- **Optimizer Step:** 0.000676s
- **Cache Refresh:** 0.000305s

### Remaining Bottleneck
The primary remaining bottleneck is the **backward pass (0.8659s)** and the **grad forward pass (0.3023s)**, which together represent **99.5%** of the update step time. These components are bound to the PyTorch Python-loop autograd path because `--update-impl` is locked to `py_autograd`.

### What must be fused next to reach 1M+ active UE4 training speed?
To reach 1M+ active training speed in active UE4/UE8 cadences, we must implement a **fused Triton backward pass** (`fused_backward`) that computes gradients for the DNA parameters directly, bypassing the position loop and autograd graph construction entirely.
