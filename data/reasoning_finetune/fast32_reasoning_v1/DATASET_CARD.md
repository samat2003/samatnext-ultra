# Dataset Card: Fast32 Reasoning Fine-Tuning Dataset

## Dataset Description
- **Dataset Name:** Fast32 Reasoning Fine-Tuning Dataset (v1.0.0)
- **Goal:** Supervised fine-tuning dataset to check if the Fast32 model can learn simple step-by-step reasoning behavior.
- **Encoding:** Byte-level UTF-8 (vocab size 256)
- **Formatting:**
  ```
  ### Task:
  <instruction>

  ### Reasoning:
  <step-by-step logic>

  ### Answer:
  <final answer>
  ```

## Categories & Example Distribution
- **Arithmetic:** 25183 examples (addition, subtraction, modulo, comparison)
- **Symbolic Reasoning:** 24861 examples (string reversal, sequence completion, list sorting, parenthesis checks)
- **Boolean Logic:** 25187 examples (AND/OR/NOT evaluation, Modus Ponens truth rules)
- **Market-bar Reasoning:** 24769 examples (candle shape analysis, direction calculations, high volume timestamps)

## Preprocessing & Validation
- Chronological global split of Binance source data.
- Duplicate checks verified.
- Byte boundaries constraint: token values are strictly inside `[0, 255]`.
- Special sequence separator byte: `2` (EOS).

---

## Disclaimers & Notes
- **No Trading Decision Labels:** Market-bar examples are simple reasoning checks (e.g. comparing volumes). No `LONG`, `SHORT`, or `HOLD` labels exist.
- **No Profit Evaluation:** No trading simulations or profit metrics are evaluated.
- **Not Financial Advice:** Research artifact only. Do not use for live trading.
