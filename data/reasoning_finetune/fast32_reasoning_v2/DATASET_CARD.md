# Dataset Card: Fast32 Answer-Only Reasoning SFT (v2)

## Overview
- Supervised fine-tuning dataset formatted as:
  ```
  Q: <short task>
  A: <answer>
  ```
- **No reasoning trace, no formatting overhead.**
- **Sequence limit:** strictly <= 128 bytes.

## Tasks included:
1. `boolean_only`
2. `arithmetic_compare_only`
3. `arithmetic_small_add_sub_only`
4. `market_direction_only`
5. `mixed_answer_only`
