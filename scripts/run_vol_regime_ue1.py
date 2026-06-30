#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXP012: Binary Volatility Regime UE1 Training.

Trains Fast32 on binary volatility-regime classification (HIGH_VOL vs LOW_VOL)
using standard scheduled updates (UE1 mode, update_every=1).
Initializes from the EXP006B market-pretrained checkpoint.
Tracks val exact accuracy on 5,000 examples and saves best checkpoints
based on CE and accuracy (rejecting collapsed/invalid models for best accuracy).
Performs final evaluation on the full test set of 83,856 examples.
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

EXP006B_CHECKPOINT = ROOT / "results_market_pretraining_long" / "checkpoint_best_val.pt"
REQUIRED_PARAM_COUNT = 216_320

# Byte tokens for first character of answer classes (byte-level UTF-8)
TOK_H = ord("H")  # 72 — first byte of "HIGH_VOL"
TOK_L = ord("L")  # 76 — first byte of "LOW_VOL"

# Decision gate thresholds
GATE_MIN_ACCURACY = 0.60
GATE_MIN_MARGIN = 0.10
GATE_MAX_INVALID_RATE = 0.0
GATE_MAX_CLASS_SHARE = 0.70
GATE_MIN_MACRO_F1 = 0.60

def load_base_checkpoint(model, chk_path: Path, device):
    if not chk_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {chk_path}")
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    step = chk.get("step", "?")
    val_loss = chk.get("best_val_loss", float("nan"))
    print(f"  Loaded EXP006B base checkpoint: step={step}, best_val_loss={val_loss:.4f}")
    return model

def evaluate_vol_regime(model, examples: list[dict], device: torch.device) -> dict:
    """Evaluate volatility regime accuracy and classification metrics.

    Uses a single forward pass per example: inspects logits at the next-token
    position after the prompt "Q: ...\nA: ". If argmax is H (72) -> HIGH_VOL.
    If argmax is L (76) -> LOW_VOL. Otherwise invalid.
    """
    model.eval()

    total_correct = 0
    total_invalid = 0
    total_count = len(examples)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    sym_correct = {s: 0 for s in symbols}
    sym_total = {s: 0 for s in symbols}
    sym_high_pred = {s: 0 for s in symbols}

    confusion = {
        "HIGH_VOL": {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0},
        "LOW_VOL":  {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0},
    }

    pred_counts = {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0}

    with torch.no_grad():
        for ex in examples:
            symbol = ex["symbol"]
            expected = ex["answer"]  # "HIGH_VOL" or "LOW_VOL"

            # Trailing space matches training prompt format
            prompt = f"Q: {ex['question']}\nA: "
            prompt_ids = torch.tensor(
                list(prompt.encode("utf-8")),
                dtype=torch.long, device=device
            ).unsqueeze(0)

            out = model(prompt_ids, return_metadata=False)
            logits = out.logits[0, -1, :]  # [256] next token distribution

            gen_tok = int(logits.argmax().item())

            if gen_tok == TOK_H:
                gen_ans = "HIGH_VOL"
            elif gen_tok == TOK_L:
                gen_ans = "LOW_VOL"
            else:
                gen_ans = ""

            is_correct = gen_ans == expected
            is_invalid = gen_ans == ""

            if is_correct:
                total_correct += 1
                sym_correct[symbol] = sym_correct.get(symbol, 0) + 1
            if is_invalid:
                total_invalid += 1

            pred_cls = gen_ans if gen_ans in ("HIGH_VOL", "LOW_VOL") else "OTHER"
            pred_counts[pred_cls] += 1
            confusion[expected][pred_cls] += 1

            if pred_cls == "HIGH_VOL":
                sym_high_pred[symbol] = sym_high_pred.get(symbol, 0) + 1

            sym_total[symbol] = sym_total.get(symbol, 0) + 1

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    invalid_rate = total_invalid / total_count if total_count > 0 else 0.0

    # Max predicted class share of valid classes
    valid_preds = pred_counts["HIGH_VOL"] + pred_counts["LOW_VOL"]
    if valid_preds > 0:
        max_share = max(pred_counts["HIGH_VOL"], pred_counts["LOW_VOL"]) / valid_preds
    else:
        max_share = 0.0
    collapsed = max_share > 0.90

    # P/R/F1 calculations
    def prf1(label):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in confusion if other != label)
        fn = sum(confusion[label][c] for c in confusion[label] if c != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    prf_high = prf1("HIGH_VOL")
    prf_low = prf1("LOW_VOL")
    macro_f1 = (prf_high["f1"] + prf_low["f1"]) / 2.0

    # Per-symbol metrics
    per_symbol_metrics = {}
    for s in symbols:
        total_s = sym_total.get(s, 0)
        correct_s = sym_correct.get(s, 0)
        high_pred_s = sym_high_pred.get(s, 0)
        per_symbol_metrics[s] = {
            "accuracy": correct_s / total_s if total_s > 0 else 0.0,
            "predicted_class_distribution": {
                "HIGH_VOL": high_pred_s / total_s if total_s > 0 else 0.0,
                "LOW_VOL": (total_s - high_pred_s) / total_s if total_s > 0 else 0.0,
            }
        }

    model.train()

    return {
        "overall_accuracy": overall_acc,
        "invalid_rate": invalid_rate,
        "predicted_class_distribution": {
            "HIGH_VOL": pred_counts["HIGH_VOL"] / total_count if total_count > 0 else 0.0,
            "LOW_VOL": pred_counts["LOW_VOL"] / total_count if total_count > 0 else 0.0,
            "OTHER": pred_counts["OTHER"] / total_count if total_count > 0 else 0.0,
        },
        "max_predicted_class_share": max_share,
        "single_class_collapse": collapsed,
        "confusion_matrix": confusion,
        "precision_recall_f1": {
            "HIGH_VOL": prf_high,
            "LOW_VOL": prf_low,
        },
        "macro_f1": macro_f1,
        "per_symbol": per_symbol_metrics
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_vol_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("EXP012: Fast32 Volatility Regime UE1 SFT Training")
    print("=================================================================")
    print(f"Device: {device}")
    print(f"Base checkpoint: {EXP006B_CHECKPOINT}")

    data_dir = ROOT / "data" / "quant_decision" / "vol_regime_H15_C60"
    for name in ["train.bin", "val.bin", "test.bin", "train.jsonl", "val.jsonl", "test.jsonl", "metadata.json"]:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing file in {data_dir}: {name}. Run prepare_vol_regime_dataset.py first.")

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Sequence length verification
    max_len = metadata["metrics"]["max_token_length"]
    print(f"  Dataset max sequence length: {max_len} bytes")
    assert max_len <= 128, f"Sequence length error: max_len={max_len} exceeds 128 bytes limit!"

    batch_size = 64
    seq_len = 128
    steps = 2500
    amp = "bf16"
    eval_every = 250
    seed = 1234

    args = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="vol_regime_H15_C60",
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

    # Load 5,000 deterministic validation examples for training evaluations
    print("Loading validation examples...")
    val_examples = []
    with open(data_dir / "val.jsonl", encoding="utf-8") as f:
        for line in f:
            val_examples.append(json.loads(line))
            if len(val_examples) >= 5000:
                break
    print(f"  Loaded {len(val_examples)} validation examples for periodic checks")

    # Load full test set for final test evaluation
    print("Loading test set...")
    test_examples = []
    with open(data_dir / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))
    print(f"  Loaded {len(test_examples)} total test examples")

    model = exp004b.make_model(device)
    n_params = model.trainable_parameter_count()
    assert n_params == REQUIRED_PARAM_COUNT, f"Param count mismatch: {n_params} != {REQUIRED_PARAM_COUNT}"
    print(f"  Model parameter count: {n_params:,} (verified)")

    model = load_base_checkpoint(model, EXP006B_CHECKPOINT, device)

    train_data = torch.from_numpy(np.fromfile(data_dir / "train.bin", dtype=np.uint8)).long()
    val_data   = torch.from_numpy(np.fromfile(data_dir / "val.bin",   dtype=np.uint8)).long()

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
    best_val_accuracy = 0.0

    chk_ce_path = out_dir / "best_val_ce.pt"
    chk_acc_path = out_dir / "best_val_accuracy.pt"

    history = []
    optimizer_updates = 0
    forward_loss_calls = 0

    t_start = time.perf_counter()

    print(f"\nTraining UE1 for {steps} steps...")
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

        if step_idx % eval_every == 0:
            val_ce = eval_val_ce()
            metrics = evaluate_vol_regime(model, val_examples, device)

            acc = metrics["overall_accuracy"]
            max_share = metrics["max_predicted_class_share"]
            inv_rate = metrics["invalid_rate"]
            collapsed = metrics["single_class_collapse"]

            print(f"  [Step {step_idx:4d}] Train CE: {lv:.4f} | Val CE: {val_ce:.4f} | Val Acc: {acc:.2%} "
                  f"| Max Share: {max_share:.2%} | Collapsed: {collapsed}")

            history.append({
                "step": step_idx,
                "train_ce": lv,
                "val_ce": val_ce,
                "val_accuracy": acc,
                "max_predicted_class_share": max_share,
                "invalid_rate": inv_rate,
                "single_class_collapse": collapsed
            })

            # Checkpoint: Best Val CE
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                torch.save({
                    "model_state": model.state_dict(),
                    "step": step_idx,
                    "val_ce": val_ce,
                    "val_accuracy": acc,
                    "metrics": metrics
                }, chk_ce_path)
                print(f"    --> Saved best_val_ce.pt (Val CE = {val_ce:.4f})")

            # Checkpoint: Best Val Accuracy
            # Rule:
            # 1. highest validation exact accuracy
            # 2. tie-breaker: lower validation CE
            # 3. reject if max predicted class share > 90%
            # 4. reject if invalid rate > 1%
            if not collapsed and inv_rate <= 0.01:
                is_better = False
                if acc > best_val_accuracy:
                    is_better = True
                elif acc == best_val_accuracy and val_ce < best_val_loss:
                    is_better = True

                if is_better:
                    best_val_accuracy = acc
                    torch.save({
                        "model_state": model.state_dict(),
                        "step": step_idx,
                        "val_ce": val_ce,
                        "val_accuracy": acc,
                        "metrics": metrics
                    }, chk_acc_path)
                    print(f"    --> Saved best_val_accuracy.pt (Val Acc = {acc:.2%}, CE = {val_ce:.4f})")

    elapsed_time = time.perf_counter() - t_start
    tokens_per_step = batch_size * seq_len
    throughput = (tokens_per_step * steps) / elapsed_time

    assert optimizer_updates == steps, f"UE1 constraint violated: updates={optimizer_updates} != steps={steps}"
    assert forward_loss_calls == steps, f"forward_loss_calls={forward_loss_calls} != steps={steps}"

    # Final Test Set Evaluation for both checkpoints
    test_results = {}
    
    # 1. Evaluate best_val_accuracy.pt (Primary Checkpoint)
    print("\n=================================================================")
    print("EVALUATING PRIMARY CHECKPOINT: best_val_accuracy.pt")
    print("=================================================================")
    if chk_acc_path.exists():
        chk = torch.load(chk_acc_path, map_location=device)
        model.load_state_dict(chk["model_state"])
        print(f"  Loaded checkpoint from step {chk['step']} (Val Acc: {chk['val_accuracy']:.2%})")
        
        # Test CE
        model.eval()
        test_losses = []
        g_test = torch.Generator().manual_seed(999)
        test_data = torch.from_numpy(np.fromfile(data_dir / "test.bin", dtype=np.uint8)).long()
        with torch.no_grad():
            for _ in range(10):
                tx, ty = exp004b.sample_batch(test_data, batch_size, seq_len, device, g_test)
                loss_t, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, amp)
                test_losses.append(loss_t.item())
        test_ce = float(np.mean(test_losses))
        
        print("  Evaluating on full test set...")
        metrics = evaluate_vol_regime(model, test_examples, device)
        metrics["test_ce"] = test_ce
        metrics["throughput_tok_s"] = throughput
        metrics["elapsed_sec"] = elapsed_time
        metrics["peak_cuda_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        metrics["forward_loss_calls"] = forward_loss_calls
        metrics["optimizer_updates"] = optimizer_updates
        metrics["checkpoint_path"] = str(chk_acc_path)
        metrics["step"] = chk["step"]
        
        test_results["best_val_accuracy"] = metrics
        
        # Decision Gate Check
        acc = metrics["overall_accuracy"]
        margin = acc - 0.50
        inv_rate = metrics["invalid_rate"]
        max_share = metrics["max_predicted_class_share"]
        macro_f1 = metrics["macro_f1"]
        collapsed = metrics["single_class_collapse"]
        
        gate_passed = (
            acc >= GATE_MIN_ACCURACY
            and margin >= GATE_MIN_MARGIN
            and inv_rate <= GATE_MAX_INVALID_RATE
            and max_share <= GATE_MAX_SHARE if "GATE_MAX_SHARE" in globals() else max_share <= GATE_MAX_CLASS_SHARE
            and macro_f1 >= GATE_MIN_MACRO_F1
            and not collapsed
        )
        metrics["gate_passed"] = gate_passed
        print(f"  Test Accuracy: {acc:.2%} (Margin: {margin:+.2%})")
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Max Share:     {max_share:.2%} (Collapsed: {collapsed})")
        print(f"  Gate Status:   {'PASSED' if gate_passed else 'FAILED'}")
    else:
        print("  [WARN] No best_val_accuracy.pt checkpoint saved (all steps collapsed or invalid!).")
        test_results["best_val_accuracy"] = None

    # 2. Evaluate best_val_ce.pt
    print("\n=================================================================")
    print("EVALUATING COMPARISON CHECKPOINT: best_val_ce.pt")
    print("=================================================================")
    if chk_ce_path.exists():
        chk = torch.load(chk_ce_path, map_location=device)
        model.load_state_dict(chk["model_state"])
        print(f"  Loaded checkpoint from step {chk['step']} (Val CE: {chk['val_ce']:.4f})")
        
        # Test CE
        model.eval()
        test_losses = []
        g_test = torch.Generator().manual_seed(999)
        test_data = torch.from_numpy(np.fromfile(data_dir / "test.bin", dtype=np.uint8)).long()
        with torch.no_grad():
            for _ in range(10):
                tx, ty = exp004b.sample_batch(test_data, batch_size, seq_len, device, g_test)
                loss_t, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, amp)
                test_losses.append(loss_t.item())
        test_ce = float(np.mean(test_losses))
        
        print("  Evaluating on full test set...")
        metrics = evaluate_vol_regime(model, test_examples, device)
        metrics["test_ce"] = test_ce
        metrics["throughput_tok_s"] = throughput
        metrics["elapsed_sec"] = elapsed_time
        metrics["peak_cuda_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        metrics["forward_loss_calls"] = forward_loss_calls
        metrics["optimizer_updates"] = optimizer_updates
        metrics["checkpoint_path"] = str(chk_ce_path)
        metrics["step"] = chk["step"]
        
        test_results["best_val_ce"] = metrics
        print(f"  Test Accuracy: {metrics['overall_accuracy']:.2%} (Val CE = {chk['val_ce']:.4f})")
    else:
        print("  [WARN] No best_val_ce.pt checkpoint found.")
        test_results["best_val_ce"] = None

    # Save outputs
    combined = {
        "dataset_name": "vol_regime_H15_C60",
        "train_elapsed_sec": elapsed_time,
        "throughput_tok_s": throughput,
        "history": history,
        "test_results": test_results
    }
    with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\nEXP012 UE1 training runner execution completed successfully.")

if __name__ == "__main__":
    main()
