#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP010C: Binary Direction Only UE1 Training.

Trains Fast32 on binary UP/DOWN direction classification using UE1 (update_every=1).
Runs one independent training run per horizon (H=5, H=15, H=60), each initialized
from the best EXP006B market-pretrained checkpoint.

Architecture is unchanged. Parameter count remains 216,320.
These are future-return direction labels, NOT trading decisions.
This is not a trading system. Do not claim profitability.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.exp004b_fast32_ue_train_speed as exp004b

HORIZONS = [5, 15, 60]
EXP006B_CHECKPOINT = ROOT / "results_market_pretraining_long" / "checkpoint_best_val.pt"
REQUIRED_PARAM_COUNT = 216_320

# Byte tokens for first character of each class (byte-level UTF-8 vocab)
TOK_U = ord("U")   # 85 — first byte of "UP"
TOK_D = ord("D")   # 68 — first byte of "DOWN"

# Decision gate thresholds
GATE_MIN_ACCURACY = 0.55
GATE_MIN_MARGIN = 0.05
GATE_MAX_INVALID_RATE = 0.0

# Evaluation sample caps
VAL_EVAL_SAMPLES = 500    # per eval checkpoint during training
TEST_EVAL_SAMPLES = 2000  # final test evaluation (satisfies >= 1000 requirement)


def load_base_checkpoint(model, chk_path: Path, device):
    """Load EXP006B market-pretrained weights into model."""
    if not chk_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {chk_path}")
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    step = chk.get("step", "?")
    val_loss = chk.get("best_val_loss", float("nan"))
    print(f"  Loaded EXP006B checkpoint: step={step}, best_val_loss={val_loss:.4f}")
    return model


def evaluate_tasks(model, jsonl_path: Path, device: torch.device,
                   max_samples: int = 2000) -> dict:
    """
    Evaluate exact accuracy on binary direction examples.

    Uses single forward pass per example: inspects next-token logits after the
    full prompt 'Q: ...\nA: ' (with trailing space, matching training format).
    Token 85 = 'U' -> predict UP. Token 68 = 'D' -> predict DOWN. Otherwise invalid.
    The space is critical: training data is 'A: UP' not 'A:UP', so the model
    predicts the token after the space. This is 1 forward pass per example vs
    1-4 for autoregressive generation, and is semantically correct since the
    format enforces UP or DOWN as the answer.
    """
    model.eval()
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    if len(examples) > max_samples:
        rng = np.random.default_rng(seed=777)
        idx = rng.choice(len(examples), size=max_samples, replace=False)
        examples = [examples[i] for i in sorted(idx)]

    total_correct = 0
    total_invalid = 0
    total_count = 0

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    sym_correct = {s: 0 for s in symbols}
    sym_total = {s: 0 for s in symbols}

    confusion = {
        "UP":   {"UP": 0, "DOWN": 0, "OTHER": 0},
        "DOWN": {"UP": 0, "DOWN": 0, "OTHER": 0},
    }

    sample_corrects = []
    sample_wrongs = []

    with torch.no_grad():
        for ex in examples:
            symbol = ex["symbol"]
            expected = ex["answer"]  # "UP" or "DOWN"

            # Feed prompt including trailing space after 'A:' — matches training format 'A: UP'
            prompt = f"Q: {ex['question']}\nA: "
            prompt_ids = torch.tensor(
                list(prompt.encode("utf-8")),
                dtype=torch.long, device=device,
            ).unsqueeze(0)

            out = model(prompt_ids, return_metadata=False)
            logits = out.logits[0, -1, :]  # [256]

            gen_tok = int(logits.argmax().item())

            if gen_tok == TOK_U:
                gen_ans = "UP"
            elif gen_tok == TOK_D:
                gen_ans = "DOWN"
            else:
                gen_ans = ""  # invalid: neither 'U' nor 'D' is top prediction

            is_correct = gen_ans == expected
            is_invalid = gen_ans == ""

            if is_correct:
                total_correct += 1
                sym_correct[symbol] = sym_correct.get(symbol, 0) + 1
                if len(sample_corrects) < 3:
                    sample_corrects.append({
                        "symbol": symbol,
                        "question": ex["question"],
                        "expected": expected,
                        "generated": gen_ans,
                    })
            else:
                if len(sample_wrongs) < 3:
                    sample_wrongs.append({
                        "symbol": symbol,
                        "question": ex["question"],
                        "expected": expected,
                        "generated": gen_ans,
                    })

            if is_invalid:
                total_invalid += 1

            pred_cls = gen_ans if gen_ans in ("UP", "DOWN") else "OTHER"
            confusion[expected][pred_cls] += 1

            sym_total[symbol] = sym_total.get(symbol, 0) + 1
            total_count += 1

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    invalid_rate = total_invalid / total_count if total_count > 0 else 0.0

    per_symbol_acc = {
        s: (sym_correct.get(s, 0) / sym_total[s]) if sym_total.get(s, 0) > 0 else 0.0
        for s in symbols
    }

    def prf1(label):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in confusion if other != label)
        fn = sum(confusion[label][c] for c in confusion[label] if c != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    prf_up = prf1("UP")
    prf_down = prf1("DOWN")

    model.train()

    return {
        "total_examples": total_count,
        "overall_accuracy": overall_acc,
        "invalid_rate": invalid_rate,
        "per_symbol_accuracy": per_symbol_acc,
        "per_symbol_counts": {
            s: {"correct": sym_correct.get(s, 0), "total": sym_total.get(s, 0)}
            for s in symbols
        },
        "confusion_matrix": confusion,
        "precision_recall_f1": {"UP": prf_up, "DOWN": prf_down},
        "sample_corrects": sample_corrects,
        "sample_wrongs": sample_wrongs,
        "majority_baseline": 0.50,
        "random_baseline": 0.50,
        "margin_over_baseline": overall_acc - 0.50,
    }


def train_one_horizon(horizon: int, device: torch.device, out_base: Path) -> dict:
    """Train UE1 on binary_dir for one horizon. Returns full results dict."""
    data_dir = ROOT / "data" / "quant_decision" / f"binary_dir_H{horizon}"
    out_dir = out_base / f"H{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXP010C: UE1 Training - Horizon H={horizon}m")
    print(f"{'='*60}")

    for name in ["train.bin", "val.bin", "test.bin",
                 "train.jsonl", "val.jsonl", "test.jsonl", "dataset_stats.json"]:
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"Dataset file missing: {data_dir / name}\n"
                "Run prepare_binary_dir_dataset.py first."
            )

    with open(data_dir / "dataset_stats.json") as f:
        dataset_stats = json.load(f)

    print(f"  Dataset: {data_dir}")
    print(f"  Train: {dataset_stats['splits']['train']['total']} examples")
    print(f"  Val:   {dataset_stats['splits']['val']['total']} examples")
    print(f"  Test:  {dataset_stats['splits']['test']['total']} examples")
    print(f"  Leakage audit: {'PASSED' if dataset_stats['leakage_audit']['passed'] else 'FAILED'}")

    # Training settings
    batch_size = 64
    seq_len = 128
    steps = 2500
    amp = "bf16"
    eval_every = 250
    seed = 1234

    args = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="",
        batch_size=batch_size,
        seq_len=seq_len,
        steps=steps,
        warmup_steps=20,
        amp=amp,
        mode="standard",  # UE1
        optimizer="fused-adamw",
        overfit_one_batch=False,
        forward_impl="cached_triton_loss",
        update_impl="py_autograd",
        profile_components=True,
        eval_every=eval_every,
        seed=seed,
    )

    # Build model and verify param count
    model = exp004b.make_model(device)
    n_params = model.trainable_parameter_count()
    assert n_params == REQUIRED_PARAM_COUNT, \
        f"Param count changed: {n_params} != {REQUIRED_PARAM_COUNT}"
    print(f"  Model parameters: {n_params:,} (verified)")

    # Load EXP006B market-pretrained base checkpoint
    print(f"  Loading base checkpoint: {EXP006B_CHECKPOINT}")
    model = load_base_checkpoint(model, EXP006B_CHECKPOINT, device)

    # Load token streams
    train_data = torch.from_numpy(np.fromfile(data_dir / "train.bin", dtype=np.uint8)).long()
    val_data   = torch.from_numpy(np.fromfile(data_dir / "val.bin",   dtype=np.uint8)).long()
    test_data  = torch.from_numpy(np.fromfile(data_dir / "test.bin",  dtype=np.uint8)).long()

    g = torch.Generator().manual_seed(seed)

    def get_batch():
        return exp004b.sample_batch(train_data, batch_size, seq_len, device, g)

    optimizer, opt_label, _ = exp004b.make_optimizer(model, args, device)

    def eval_val_ce():
        model.eval()
        val_losses = []
        g_val = torch.Generator().manual_seed(seed + 888)
        with torch.no_grad():
            for _ in range(10):
                vx, vy = exp004b.sample_batch(val_data, batch_size, seq_len, device, g_val)
                loss_v, _ = exp004b.cached_triton_forward_loss(model, None, vx, vy, amp)
                val_losses.append(loss_v.item())
        model.train()
        return float(np.mean(val_losses))

    best_val_loss = float("inf")
    best_chk_path = out_dir / "checkpoint_best_val.pt"

    history_train_ce = []
    history_val_ce = {}
    history_acc = {}

    first_loss = None
    final_loss = None
    optimizer_updates = 0
    forward_loss_calls = 0

    t_start = time.perf_counter()

    print(f"\n  Training UE1 for {steps} steps (eval every {eval_every})...")
    for step_idx in range(1, steps + 1):
        x, y = get_batch()

        with exp004b.autocast_ctx(device, amp):
            out = model(x, return_metadata=False)
            loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
        forward_loss_calls += 1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_updates += 1

        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        final_loss = lv

        if step_idx % eval_every == 0:
            val_ce = eval_val_ce()
            acc_metrics = evaluate_tasks(
                model, data_dir / "val.jsonl", device, max_samples=VAL_EVAL_SAMPLES
            )
            overall_acc = acc_metrics["overall_accuracy"]
            invalid_rate = acc_metrics["invalid_rate"]

            print(f"  [Step {step_idx:4d}] Train CE: {lv:.4f} | Val CE: {val_ce:.4f} | "
                  f"Acc: {overall_acc:.2%} | Invalid: {invalid_rate:.2%}")

            history_train_ce.append((step_idx, lv))
            history_val_ce[step_idx] = val_ce
            history_acc[step_idx] = {
                "overall_accuracy": overall_acc,
                "invalid_rate": invalid_rate,
            }

            if val_ce < best_val_loss:
                best_val_loss = val_ce
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step_idx,
                    "best_val_loss": val_ce,
                }, best_chk_path)

    elapsed = time.perf_counter() - t_start
    tokens_per_step = batch_size * seq_len
    throughput_tok_s = (tokens_per_step * steps) / elapsed

    assert optimizer_updates == steps, \
        f"UE1 violation: optimizer_updates={optimizer_updates} != steps={steps}"
    assert forward_loss_calls == steps, \
        f"forward_loss_calls={forward_loss_calls} != steps={steps}"

    # Load best checkpoint
    print(f"\n  Loading best checkpoint from {best_chk_path} ...")
    chk = torch.load(best_chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    print(f"  Best checkpoint: step={chk['step']}, val_loss={chk['best_val_loss']:.4f}")

    # Test CE (token-level)
    model.eval()
    test_losses = []
    g_test = torch.Generator().manual_seed(999)
    with torch.no_grad():
        for _ in range(10):
            tx, ty = exp004b.sample_batch(test_data, batch_size, seq_len, device, g_test)
            loss_t, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, amp)
            test_losses.append(loss_t.item())
    test_ce = float(np.mean(test_losses))

    # Full test evaluation (capped at TEST_EVAL_SAMPLES)
    print(f"  Evaluating on test set (up to {TEST_EVAL_SAMPLES} examples)...")
    test_acc_metrics = evaluate_tasks(
        model, data_dir / "test.jsonl", device, max_samples=TEST_EVAL_SAMPLES
    )

    overall_acc = test_acc_metrics["overall_accuracy"]
    invalid_rate = test_acc_metrics["invalid_rate"]
    margin = test_acc_metrics["margin_over_baseline"]

    print(f"\n  --- H={horizon}m Test Results ---")
    print(f"  Test CE: {test_ce:.4f}")
    print(f"  Overall Test Accuracy: {overall_acc:.2%} (baseline 50.00%, margin {margin:+.2%})")
    print(f"  Invalid/Empty Rate: {invalid_rate:.2%}")
    print(f"  Confusion Matrix:")
    for true_lbl, preds in test_acc_metrics["confusion_matrix"].items():
        print(f"    True {true_lbl}: {preds}")
    print(f"  Per-symbol accuracy:")
    for sym, acc in test_acc_metrics["per_symbol_accuracy"].items():
        cnt = test_acc_metrics["per_symbol_counts"][sym]
        print(f"    {sym}: {acc:.2%}  ({cnt['correct']}/{cnt['total']})")
    print(f"  Precision/Recall/F1:")
    for cls, prf in test_acc_metrics["precision_recall_f1"].items():
        print(f"    {cls}: P={prf['precision']:.3f}, R={prf['recall']:.3f}, F1={prf['f1']:.3f}")

    # Decision gate
    gate_passed = (
        overall_acc >= GATE_MIN_ACCURACY
        and margin >= GATE_MIN_MARGIN
        and invalid_rate <= GATE_MAX_INVALID_RATE
    )
    print(f"\n  Decision Gate:")
    print(f"    accuracy >= {GATE_MIN_ACCURACY:.0%}:   {'PASS' if overall_acc >= GATE_MIN_ACCURACY else 'FAIL'} ({overall_acc:.2%})")
    print(f"    margin >= {GATE_MIN_MARGIN:.0%}:     {'PASS' if margin >= GATE_MIN_MARGIN else 'FAIL'} ({margin:+.2%})")
    print(f"    invalid == 0%:      {'PASS' if invalid_rate <= GATE_MAX_INVALID_RATE else 'FAIL'} ({invalid_rate:.2%})")
    print(f"    GATE: {'PASSED' if gate_passed else 'FAILED'}")

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0

    results = {
        "horizon": horizon,
        "dataset_dir": str(data_dir),
        "dataset_stats": dataset_stats,
        "base_checkpoint": str(EXP006B_CHECKPOINT),
        "mode": "UE1",
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "optimizer": opt_label,
        "amp": amp,
        "seed": seed,
        "param_count": n_params,
        "elapsed_sec": elapsed,
        "throughput_tok_s": throughput_tok_s,
        "optimizer_updates": optimizer_updates,
        "forward_loss_calls": forward_loss_calls,
        "first_train_ce": first_loss,
        "final_train_ce": final_loss,
        "best_val_ce": best_val_loss,
        "best_val_step": chk["step"],
        "test_ce": test_ce,
        "test_accuracy": overall_acc,
        "majority_baseline": 0.50,
        "margin_over_baseline": margin,
        "invalid_rate": invalid_rate,
        "test_acc_metrics": test_acc_metrics,
        "history_train_ce": history_train_ce,
        "history_val_ce": history_val_ce,
        "history_acc": history_acc,
        "peak_cuda_mem_gb": peak_mem,
        "best_chk_path": str(best_chk_path),
        "gate_passed": gate_passed,
        "gate_criteria": {
            "min_accuracy": GATE_MIN_ACCURACY,
            "min_margin": GATE_MIN_MARGIN,
            "max_invalid_rate": GATE_MAX_INVALID_RATE,
        },
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_base = ROOT / "results_binary_dir"
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXP010C: Binary Direction Only UE1")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Base checkpoint: {EXP006B_CHECKPOINT}")
    print(f"Horizons: {HORIZONS}")
    print(f"Val eval cap: {VAL_EVAL_SAMPLES} examples")
    print(f"Test eval cap: {TEST_EVAL_SAMPLES} examples")
    print(f"Decision gate: accuracy>={GATE_MIN_ACCURACY:.0%}, "
          f"margin>={GATE_MIN_MARGIN:.0%}, invalid==0%")
    print("Honesty: labels are future-return thresholds, NOT trading decisions.")

    all_results = {}

    for horizon in HORIZONS:
        results = train_one_horizon(horizon, device, out_base)
        all_results[f"H{horizon}"] = results

    # Comparison table
    print(f"\n{'='*60}")
    print("EXP010C FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Test Acc':<12} {'Margin':<10} {'Invalid':<10} {'Gate'}")
    for H in HORIZONS:
        r = all_results[f"H{H}"]
        print(f"H={H:<8} {r['test_accuracy']:.2%}       {r['margin_over_baseline']:+.2%}      "
              f"{r['invalid_rate']:.2%}      {'PASS' if r['gate_passed'] else 'FAIL'}")

    passing = [H for H in HORIZONS if all_results[f"H{H}"]["gate_passed"]]
    if passing:
        best_H = max(passing, key=lambda H: all_results[f"H{H}"]["test_accuracy"])
        print(f"\n  Decision gate PASSED for: {[f'H={H}' for H in passing]}")
        print(f"  Best horizon by accuracy: H={best_H} "
              f"({all_results[f'H{best_H}']['test_accuracy']:.2%})")
        print(f"  Model is ready for binary quant-decision experiments at H={best_H}.")
    else:
        print(f"\n  No horizon passed the decision gate.")
        print(f"  Binary direction labels remain too noisy or not learnable at 2500 UE1 steps.")
        print(f"  Suggested next steps:")
        print(f"    - Increase threshold (larger, cleaner labels)")
        print(f"    - Use longer horizon (120m, 240m)")
        print(f"    - Try per-symbol volatility normalization")
        print(f"    - Try larger model or more training steps")
        print(f"    - Try richer context (volume, OHLC, more bars)")

    combined = {
        "experiment": "EXP010C",
        "mode": "UE1",
        "horizons_tested": HORIZONS,
        "gate_criteria": {
            "min_accuracy": GATE_MIN_ACCURACY,
            "min_margin": GATE_MIN_MARGIN,
            "max_invalid_rate": GATE_MAX_INVALID_RATE,
        },
        "passing_horizons": passing,
        "results_by_horizon": all_results,
    }

    with open(out_base / "combined_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Combined results saved: {out_base / 'combined_results.json'}")
    print("EXP010C pipeline completed.")


if __name__ == "__main__":
    main()
