# Dataset Card: Fast32 Quant Decision Classification Dataset

## Overview
- Compact answer-only SFT dataset for quant-decision classification.
- **Format:** `Q: <task description>
A: <label>`
- **No reasoning trace, no formatting overhead.**
- **Sequence limit:** strictly <= 128 bytes.

## Tasks included:
1. `binary_dir` (UP / DOWN)
2. `three_class_dir` (UP / DOWN / FLAT)
3. `vol_regime` (VOL_UP / VOL_DOWN / VOL_FLAT)
4. `breakout` (BREAKOUT / NO_BREAKOUT)
5. `trend_range` (TREND / RANGE)

---

## Disclaimers & Notes
- **No Trading Decision Signals:** Market direction classification is a simple programmatic check, NOT live signals.
- **No Profit Evaluation:** No trading simulations or profit metrics are evaluated.
