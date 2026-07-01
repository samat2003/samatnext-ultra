#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the final test accuracy of Fast32 on vol_regime_H15_C60.

Evaluates results_vol_regime/best_val_accuracy.pt on data/quant_decision/vol_regime_H15_C60/test.jsonl.
"""

import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.exp004b_fast32_ue_train_speed as exp004b

CHECKPOINT_PATH = ROOT / "results_vol_regime" / "best_val_accuracy.pt"
DATA_DIR        = ROOT / "data" / "quant_decision" / "vol_regime_H15_C60"

TOK_H = ord("H")
TOK_L = ord("L")


def evaluate_test_set(model, examples: list[dict], device: torch.device) -> dict:
    model.eval()
    total_correct = 0
    total_invalid = 0
    total_count = len(examples)

    confusion = {
        "HIGH_VOL": {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0},
        "LOW_VOL":  {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0},
    }
    pred_counts = {"HIGH_VOL": 0, "LOW_VOL": 0, "OTHER": 0}

    # Evaluate in batches for efficiency (logits are identical to sequential)
    batch_size = 256
    with torch.no_grad():
        for i in range(0, total_count, batch_size):
            batch_exs = examples[i : i + batch_size]
            batch_ids = []
            
            for ex in batch_exs:
                prompt = f"Q: {ex['question']}\nA: "
                ids = list(prompt.encode("utf-8"))
                if len(ids) < 128:
                    ids = ids + [32] * (128 - len(ids))
                else:
                    ids = ids[:128]
                batch_ids.append(ids)
                
            x = torch.tensor(batch_ids, dtype=torch.long, device=device)
            out = model(x, return_metadata=False)
            
            for idx, ex in enumerate(batch_exs):
                expected = ex["answer"]
                prompt = f"Q: {ex['question']}\nA: "
                prompt_len = len(prompt.encode("utf-8"))
                
                logits = out.logits[idx, prompt_len - 1, :]
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
                if is_invalid:
                    total_invalid += 1

                pred_cls = gen_ans if gen_ans in ("HIGH_VOL", "LOW_VOL") else "OTHER"
                pred_counts[pred_cls] += 1
                confusion[expected][pred_cls] += 1

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    invalid_rate = total_invalid / total_count if total_count > 0 else 0.0
    valid_preds = pred_counts["HIGH_VOL"] + pred_counts["LOW_VOL"]
    max_share = max(pred_counts["HIGH_VOL"], pred_counts["LOW_VOL"]) / valid_preds if valid_preds > 0 else 0.0

    def prf1(label):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in confusion if other != label)
        fn = sum(confusion[label][c] for c in confusion[label] if c != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    prf_high = prf1("HIGH_VOL")
    prf_low = prf1("LOW_VOL")
    macro_f1 = (prf_high["f1"] + prf_low["f1"]) / 2.0

    return {
        "overall_accuracy": overall_acc,
        "invalid_rate": invalid_rate,
        "max_predicted_class_share": max_share,
        "macro_f1": macro_f1,
        "predicted_class_distribution": {
            "HIGH_VOL": pred_counts["HIGH_VOL"] / total_count if total_count > 0 else 0.0,
            "LOW_VOL": pred_counts["LOW_VOL"] / total_count if total_count > 0 else 0.0,
            "OTHER": pred_counts["OTHER"] / total_count if total_count > 0 else 0.0,
        },
        "confusion_matrix": confusion,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("FAST32 VOLATILITY REGIME ACCURACY REPRODUCTION")
    print("=================================================================")
    print(f"Device: {device}")
    
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
        
    if not (DATA_DIR / "test.jsonl").exists():
        print(f"Error: Test dataset not found at {DATA_DIR / 'test.jsonl'}")
        print("Please prepare the dataset first using prepare_vol_regime_dataset.py.")
        sys.exit(1)

    print("Loading test set...")
    examples = []
    with open(DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} test examples")

    print("Initializing model and loading checkpoint...")
    model = exp004b.make_model(device)
    chk = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(chk["model_state"])
    print(f"  Loaded step {chk.get('step')} checkpoint.")

    print("\nRunning test evaluation...")
    metrics = evaluate_test_set(model, examples, device)

    print("\n----------------- RESULTS -----------------")
    print(f"Test Accuracy:             {metrics['overall_accuracy']:.4%} (Expected: ~71.38%)")
    print(f"Macro F1 Score:            {metrics['macro_f1']:.4f} (Expected: ~0.706)")
    print(f"Invalid Rate:              {metrics['invalid_rate']:.4%} (Expected: 0.00%)")
    print(f"Max Predicted Class Share: {metrics['max_predicted_class_share']:.4%} (Expected: ~66.40%)")
    
    dist = metrics["predicted_class_distribution"]
    print(f"Prediction Distribution:   HIGH_VOL={dist['HIGH_VOL']:.2%}, LOW_VOL={dist['LOW_VOL']:.2%}, OTHER={dist['OTHER']:.2%}")
    
    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print("                Pred HIGH_VOL   Pred LOW_VOL   Pred OTHER")
    print(f"True HIGH_VOL   {cm['HIGH_VOL']['HIGH_VOL']:<13d}   {cm['HIGH_VOL']['LOW_VOL']:<12d}   {cm['HIGH_VOL']['OTHER']:<10d}")
    print(f"True LOW_VOL    {cm['LOW_VOL']['HIGH_VOL']:<13d}   {cm['LOW_VOL']['LOW_VOL']:<12d}   {cm['LOW_VOL']['OTHER']:<10d}")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()
