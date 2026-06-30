import argparse
import gc
import json
import os
import random
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

def generate_answer_v2(model, question: str, device: torch.device, max_new_tokens: int = 15) -> str:
    prompt = f"Q: {question}\nA: "
    prompt_bytes = prompt.encode("utf-8")
    input_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    
    generated_bytes = bytearray()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids, return_metadata=False)
            logits = out.logits[0, -1]
            next_token = logits.argmax().item()
            
            # Stop at newline (10) or EOS (2)
            if next_token in [10, 2]:
                break
            generated_bytes.append(next_token)
            
            # Append token to input_ids for next step
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=-1)
            
    return generated_bytes.decode("utf-8", errors="ignore").strip()

def evaluate_accuracy_v2(model, dataset_path: Path, device: torch.device, samples_per_cat: int = 100) -> dict:
    model.eval()
    
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            
    # Sample up to samples_per_cat
    random.Random(42).shuffle(examples)
    examples = examples[:samples_per_cat]
    
    total_correct = 0
    total_count = 0
    invalid_count = 0
    
    sample_corrects = []
    sample_wrongs = []
    
    # Track confusion matrix
    all_classes = sorted(list(set(ex["answer"] for ex in examples)))
    confusion = {true_c: {pred_c: 0 for pred_c in all_classes + ["OTHER"]} for true_c in all_classes}
    
    for ex in examples:
        gen_ans = generate_answer_v2(model, ex["question"], device)
        expected = ex["answer"].strip()
        
        is_correct = (gen_ans == expected)
        if is_correct:
            correct_flag = True
            total_correct += 1
            if len(sample_corrects) < 3:
                sample_corrects.append({"question": ex["question"], "expected": expected, "generated": gen_ans})
        else:
            correct_flag = False
            if len(sample_wrongs) < 3:
                sample_wrongs.append({"question": ex["question"], "expected": expected, "generated": gen_ans})
                
        if not gen_ans:
            invalid_count += 1
            
        # Update confusion matrix
        pred_class = gen_ans if gen_ans in confusion[expected] else "OTHER"
        confusion[expected][pred_class] += 1
        
        total_count += 1
        
    model.train()
    
    return {
        "accuracy": total_correct / total_count if total_count > 0 else 0.0,
        "count": total_count,
        "invalid_rate": invalid_count / total_count if total_count > 0 else 0.0,
        "confusion_matrix": confusion,
        "sample_corrects": sample_corrects,
        "sample_wrongs": sample_wrongs
    }

def run_single_task(task_name: str, device: torch.device) -> dict:
    print(f"\n=================================================================")
    print(f"TRAINING TASK: {task_name}")
    print(f"=================================================================")
    
    out_dir = ROOT / "results_reasoning_sft_v2" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    args = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="",
        batch_size=64,
        seq_len=128,  # Answer-only examples are short (<= 128 bytes)
        steps=2500,
        warmup_steps=20,
        amp="bf16",
        mode="standard",  # UE1
        optimizer="fused-adamw",
        overfit_one_batch=False,
        forward_impl="cached_triton_loss",
        update_impl="py_autograd",
        profile_components=True,
        eval_every=250,
        seed=1234,
    )
    
    model = exp004b.make_model(device)
    
    # Load task binary files
    sft_dir = ROOT / "data" / "reasoning_finetune" / "fast32_reasoning_v2" / task_name
    train_path = sft_dir / "train.bin"
    val_path = sft_dir / "val.bin"
    test_path = sft_dir / "test.bin"
    
    train_data = torch.from_numpy(np.fromfile(train_path, dtype=np.uint8)).long()
    val_data = torch.from_numpy(np.fromfile(val_path, dtype=np.uint8)).long()
    test_data = torch.from_numpy(np.fromfile(test_path, dtype=np.uint8)).long()
    
    g = torch.Generator().manual_seed(args.seed)
    def get_batch():
        return exp004b.sample_batch(train_data, args.batch_size, args.seq_len, device, g)
        
    optimizer, opt_label, opt_fallback = exp004b.make_optimizer(model, args, device)
    amp = args.amp
    
    # Helper to calculate cross entropy on validation
    def eval_val_ce():
        model.eval()
        val_losses = []
        g_val = torch.Generator().manual_seed(args.seed + 888)
        with torch.no_grad():
            for _ in range(10):
                vx, vy = exp004b.sample_batch(val_data, args.batch_size, args.seq_len, device, g_val)
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
    t_start = time.perf_counter()
    
    # Training Loop
    for step_idx in range(1, args.steps + 1):
        x, y = get_batch()
        with exp004b.autocast_ctx(device, amp):
            out = model(x, return_metadata=False)
            loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        final_loss = lv
        
        # Periodic validation
        if step_idx % args.eval_every == 0:
            val_ce = eval_val_ce()
            print(f"[Step {step_idx:4d}] Train CE: {lv:.4f} | Val CE: {val_ce:.4f}")
            
            # Evaluate SFT exact accuracy on validation JSONL (100 samples)
            acc_metrics = evaluate_accuracy_v2(model, sft_dir / "val.jsonl", device, samples_per_cat=100)
            print(f"Validation Accuracy: {acc_metrics['accuracy']:.2%} | Invalid Rate: {acc_metrics['invalid_rate']:.2%}")
            
            history_val_ce[step_idx] = val_ce
            history_train_ce.append((step_idx, lv))
            history_acc[step_idx] = acc_metrics
            
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step_idx,
                    "best_val_loss": val_ce
                }, best_chk_path)
                
    elapsed = time.perf_counter() - t_start
    tokens_per_step = args.batch_size * args.seq_len
    train_input_tok_s = (tokens_per_step * step_idx) / elapsed
    
    # Final Test set evaluation
    print(f"Loading best checkpoint from {best_chk_path} ...")
    chk = torch.load(best_chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    
    model.eval()
    test_losses = []
    g_test = torch.Generator().manual_seed(999)
    with torch.no_grad():
        for _ in range(10):
            tx, ty = exp004b.sample_batch(test_data, args.batch_size, args.seq_len, device, g_test)
            loss_t, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, amp)
            test_losses.append(loss_t.item())
    test_ce = float(np.mean(test_losses))
    
    # Test accuracy (100 samples)
    test_acc_metrics = evaluate_accuracy_v2(model, sft_dir / "test.jsonl", device, samples_per_cat=100)
    print(f"Final Test CE: {test_ce:.4f} | Test Accuracy: {test_acc_metrics['accuracy']:.2%}")
    
    results = {
        "task_name": task_name,
        "train_elapsed_sec": elapsed,
        "train_input_tok_s": train_input_tok_s,
        "first_train_ce": first_loss,
        "final_train_ce": final_loss,
        "best_val_ce": best_val_loss,
        "test_ce": test_ce,
        "test_accuracy_metrics": test_acc_metrics,
        "history_train_ce": history_train_ce,
        "history_val_ce": history_val_ce,
        "history_acc": history_acc,
        "best_chk_path": str(best_chk_path)
    }
    
    # Save results to json
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_reasoning_sft_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run boolean_only
    res_boolean = run_single_task("boolean_only", device)
    bool_acc = res_boolean["test_accuracy_metrics"]["accuracy"]
    
    # Gate 1: If boolean_only accuracy stays below 70%, stop and debug
    if bool_acc < 0.70:
        print(f"\nDECISION GATE: boolean_only accuracy is {bool_acc:.2%}, which is below the 70% threshold. Stopping.")
        # Write combined final report of what completed
        combined = {"boolean_only": res_boolean, "status": "stopped_at_boolean_gate"}
        with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        sys.exit(0)
        
    print("\nDECISION GATE: boolean_only accuracy passed 70%. Continuing to arithmetic_compare_only...")
    res_compare = run_single_task("arithmetic_compare_only", device)
    compare_acc = res_compare["test_accuracy_metrics"]["accuracy"]
    
    # Gate 2: If arithmetic_compare_only exceeds 60%, continue to market_direction_only
    if compare_acc < 0.60:
        print(f"\nDECISION GATE: arithmetic_compare_only accuracy is {compare_acc:.2%}, which is below the 60% threshold. Stopping.")
        combined = {
            "boolean_only": res_boolean,
            "arithmetic_compare_only": res_compare,
            "status": "stopped_at_compare_gate"
        }
        with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        sys.exit(0)
        
    print("\nDECISION GATE: arithmetic_compare_only accuracy passed 60%. Continuing to market_direction_only...")
    res_market = run_single_task("market_direction_only", device)
    
    # Since single-task results worked, run mixed_answer_only
    print("\nDECISION GATE: Single tasks succeeded. Running mixed_answer_only...")
    res_mixed = run_single_task("mixed_answer_only", device)
    
    combined = {
        "boolean_only": res_boolean,
        "arithmetic_compare_only": res_compare,
        "market_direction_only": res_market,
        "mixed_answer_only": res_mixed,
        "status": "all_succeeded"
    }
    with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
        
    print("\nAll answer-only reasoning experiments completed successfully!")

if __name__ == "__main__":
    main()
