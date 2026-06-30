# EXP010C: Binary Direction Only UE1 — Results

> **Honesty disclaimer**: These labels are future-return direction thresholds, NOT trading decisions.
> This is not a trading system. Accuracy above 50% baseline does not imply profit.

---

## Summary

All three horizons **failed** the decision gate. The model collapsed to single-class prediction on every horizon, producing non-discriminative results despite decreasing token-level cross-entropy loss.

| Horizon | Test Acc | Baseline | Margin | Invalid | Collapse Type | Gate |
|---------|----------|----------|--------|---------|---------------|------|
| **H=5m** | 51.35% | 50.00% | +1.35% | 0.00% | Always-UP | ❌ FAIL |
| **H=15m** | 49.00% | 50.00% | -1.00% | 0.00% | Always-DOWN | ❌ FAIL |
| **H=60m** | 50.05% | 50.00% | +0.05% | 0.00% | Always-UP | ❌ FAIL |

> [!IMPORTANT]
> The 0% invalid rate above was verified with a **corrected evaluation prompt** (`"A: "` with trailing space, matching training format). The training script itself used a buggy prompt (`"A:"` without space), which caused 100% invalid rate during training eval printouts — but **checkpoints were saved on val CE, not accuracy**, so the checkpoints are correct.

---

## Dataset Quality

All datasets passed leakage audit. Each split was exactly balanced UP=DOWN, giving majority baseline = 50.00%.

| Horizon | Threshold | Train | Val | Test | Near-flat Dropped | Embargo | Leakage |
|---------|-----------|-------|-----|------|-------------------|---------|---------|
| H=5m | 0.15% | 347,340 | 90,510 | 77,350 | 1,583,325 | 50 | ✅ PASS |
| H=15m | 0.20% | 478,640 | 123,390 | 105,546 | 1,390,499 | 170 | ✅ PASS |
| H=60m | 0.30% | 621,528 | 157,708 | 134,032 | 1,175,758 | 608 | ✅ PASS |

Data source: 12 months (Jun 2025 – May 2026) × 4 symbols (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT).
Split: chronological 70/15/15 by unique timestamps. Embargo = max(context_window=5, H) bars.

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Mode | UE1 (update_every=1) |
| Steps | 2,500 |
| Batch size | 64 |
| Seq len | 128 |
| Precision | bf16 |
| Optimizer | fused-adamw |
| Eval every | 250 steps |
| Seed | 1234 |
| Base checkpoint | EXP006B market-pretrained (step=20,000, val_loss=1.3307) |
| Architecture | Original stateful Fast32 |
| Parameter count | **216,320** (verified, unchanged) |

Each horizon was trained independently from the same EXP006B base checkpoint.

---

## Training Convergence

| Horizon | First Train CE | Final Train CE | Best Val CE | Best Val Step |
|---------|---------------|----------------|-------------|--------------|
| H=5m | ~0.28 | 0.2652 | 0.3038 | 1250 |
| H=15m | ~0.26 | 0.2537 | 0.2761 | 2250 |
| H=60m | ~0.25 | 0.2421 | 0.2851 | 1750 |

Token-level CE decreased consistently on all horizons — the model was learning the token distribution of the training stream. However, decreasing CE does not imply discriminative classification.

---

## Test Results (Corrected Prompt Evaluation)

### H=5m

- **Test accuracy**: 51.35% (2000 samples)
- **Margin over baseline**: +1.35%
- **Invalid rate**: 0.00%
- **Collapse**: Always predicts UP

**Confusion Matrix:**

| | Pred UP | Pred DOWN |
|--|---------|-----------|
| **True UP** | 1,027 | 0 |
| **True DOWN** | 973 | 0 |

**Per-symbol accuracy:**
| Symbol | Accuracy |
|--------|----------|
| BTCUSDT | 49.5% |
| ETHUSDT | 53.8% |
| SOLUSDT | 51.0% |
| BNBUSDT | 50.3% |

**P/R/F1:** UP: P=0.514, R=1.000, F1=0.679 | DOWN: P=0.000, R=0.000, F1=0.000

---

### H=15m

- **Test accuracy**: 49.00% (2000 samples)
- **Margin over baseline**: -1.00%
- **Invalid rate**: 0.00%
- **Collapse**: Always predicts DOWN

**Confusion Matrix:**

| | Pred UP | Pred DOWN |
|--|---------|-----------|
| **True UP** | 0 | 1,020 |
| **True DOWN** | 0 | 980 |

**Per-symbol accuracy:**
| Symbol | Accuracy |
|--------|----------|
| BTCUSDT | 50.2% |
| ETHUSDT | 50.1% |
| SOLUSDT | 47.6% |
| BNBUSDT | 48.6% |

**P/R/F1:** UP: P=0.000, R=0.000, F1=0.000 | DOWN: P=0.490, R=1.000, F1=0.657

---

### H=60m

- **Test accuracy**: 50.05% (2000 samples)
- **Margin over baseline**: +0.05%
- **Invalid rate**: 0.00%
- **Collapse**: Always predicts UP

**Confusion Matrix:**

| | Pred UP | Pred DOWN |
|--|---------|-----------|
| **True UP** | 994 | 0 |
| **True DOWN** | 1,006 | 0 |

**Per-symbol accuracy:**
| Symbol | Accuracy |
|--------|----------|
| BTCUSDT | 49.8% |
| ETHUSDT | 50.7% |
| SOLUSDT | 50.0% |
| BNBUSDT | 49.6% |

**P/R/F1:** UP: P=0.497, R=1.000, F1=0.664 | DOWN: P=0.000, R=0.000, F1=0.000

---

## Decision Gate

| Criterion | Requirement | H=5m | H=15m | H=60m |
|-----------|------------|------|-------|-------|
| Accuracy | ≥55% | ❌ 51.35% | ❌ 49.00% | ❌ 50.05% |
| Margin | ≥5% | ❌ +1.35% | ❌ -1.00% | ❌ +0.05% |
| Invalid rate | =0% | ✅ 0.00% | ✅ 0.00% | ✅ 0.00% |
| **Gate** | All pass | ❌ **FAIL** | ❌ **FAIL** | ❌ **FAIL** |

**No horizon passed the decision gate.**

---

## Answers to Required Questions

| Question | Answer |
|---|---|
| Was only `binary_dir` used? | ✅ Yes |
| Were all other tasks excluded? | ✅ Yes (three_class_dir, vol_regime, breakout, trend_range all excluded) |
| Did leakage/embargo audit pass? | ✅ Yes — all 3 horizons passed |
| Were UP/DOWN classes balanced? | ✅ Yes — exact balance, UP=DOWN in each split |
| How many examples were removed? | H5: 1,587,160 (flat+embargo+balance); H15: 1,394,744; H60: 1,188,872 |
| Did train CE decrease? | ✅ Yes — all horizons converged from ~1.33 to ~0.25 |
| Did validation CE improve? | ✅ Yes — val CE dropped consistently |
| What was test accuracy per horizon? | H5: 51.35% | H15: 49.00% | H60: 50.05% |
| What was majority baseline? | 50.00% (all horizons, exactly balanced) |
| What was margin over baseline? | H5: +1.35% | H15: -1.00% | H60: +0.05% |
| What was invalid/empty rate? | 0.00% (all horizons, corrected eval) |
| Which horizon worked best? | H=5m (+1.35% margin), but via single-class collapse |
| Did any horizon reach 55%+? | ❌ No |
| Is the model ready for binary quant-decision? | ❌ No — decision gate failed for all horizons |

---

## Root Cause Analysis

### Single-Class Collapse

All three models collapsed to predicting **one class for all inputs**:
- H5 → always `UP`
- H15 → always `DOWN`
- H60 → always `UP`

This is a canonical failure mode for learning from near-random binary labels:
1. The token-level cross-entropy loss incentivizes the model to match the answer token distribution
2. Since UP/DOWN are balanced 50/50, the model gets ~log(2) ≈ 0.693 CE from predicting either class always
3. With only 2500 steps and a short context window of 5 bars, the model never discovers per-input discrimination
4. Instead, it biases toward whichever class's token ('U'=85 vs 'D'=68) it encountered slightly more in early batches

### Why Val CE Decreased

Val CE decreased because the model learned the **format** (predicting plausible tokens after `"A: "`) but not the **label** (which token is correct for each context). A model that always outputs `U` with probability p=0.9 will have lower CE than a random model if ~51% of training answers are `UP`, even though it's not discriminating.

### Context Insufficiency

Five past 1-minute bar returns provide extremely weak signal for predicting direction H minutes ahead. Market microstructure noise dominates at 1-minute resolution. The model's SSM state carries information from prior examples in the batch stream, but this provides no benefit for per-example discrimination.

---

## What Should Change

> [!WARNING]
> **Do not continue training blindly.** The current dataset format and context encoding appear insufficient for discriminative learning. Changes to the task formulation are needed before further training.

### Dataset Changes (High Priority)
1. **Larger threshold** (e.g., 0.5%–1.0%) — removes more ambiguous near-flat examples, giving cleaner signal
2. **Longer horizon** (H=120m, H=240m, H=1440m) — more trend autocorrelation; still learnable from 1m bars
3. **Richer context** — include volume, OHLC, volatility, or more bars (e.g., 20–60 bars)
4. **Per-symbol normalization** — normalize returns by rolling volatility to make signals comparable across symbols

### Architecture/Training Changes (Medium Priority)
5. **More training steps** (5,000–10,000) — may help format learning but unlikely to fix signal absence
6. **Explicit sequence separation** — insert `<EOS>` tokens between examples so SSM state doesn't carry across examples
7. **Curriculum** — start with extreme directional moves (e.g., >2% threshold) where signal is stronger

### Alternative Task Formulations
8. **Volatility regime** — predict whether next-H volatility is above/below the rolling median (binary, more predictable)
9. **Extreme move detection** — predict whether abs(return) > 1% in next H minutes (rare but learnable)
10. **Trend continuation** — predict whether sign(return_H) == sign(return_last_5bars_sum) (autocorrelation-based)

---

## Notes on Evaluation Bug

The training runner evaluated accuracy using prompt `"A:"` (no space), causing 100% invalid rate during training `[Step N]` printouts. The final `results.json` from the runner also reports 0% accuracy due to this bug. The **corrected evaluation** above uses `"A: "` (with space, matching training format `"A: UP"`). All accuracy numbers in this report are from the corrected evaluator. Checkpoints were saved on val CE (not accuracy), so they are unaffected by the evaluation bug.
