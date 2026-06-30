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

def generate_answer(model, question: str, device: torch.device, max_new_tokens: int = 15) -> str:
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
            
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=-1)
            
    return generated_bytes.decode("utf-8", errors="ignore").strip()

def evaluate_tasks(model, dataset_path: Path, device: torch.device, samples_per_task: int = 100) -> dict:
    model.eval()
    
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            
    # Group by task
    grouped = {}
    for ex in examples:
        grouped.setdefault(ex["task"], []).append(ex)
        
    results_by_task = {}
    total_correct = 0
    total_count = 0
    invalid_count = 0
    
    allowed_tasks = ["binary_dir", "three_class_dir", "vol_regime", "breakout"]
    
    for task in allowed_tasks:
        task_exs = grouped.get(task, [])
        if not task_exs:
            continue
            
        random.Random(42).shuffle(task_exs)
        sampled = task_exs[:samples_per_task]
        
        task_correct = 0
        task_total = 0
        
        # Track confusion matrix
        all_classes = sorted(list(set(ex["answer"] for ex in task_exs)))
        confusion = {true_c: {pred_c: 0 for pred_c in all_classes + ["OTHER"]} for true_c in all_classes}
        
        sample_corrects = []
        sample_wrongs = []
        
        for ex in sampled:
            gen_ans = generate_answer(model, ex["question"], device)
            expected = ex["answer"].strip()
            
            is_correct = (gen_ans == expected)
            if is_correct:
                task_correct += 1
                total_correct += 1
                if len(sample_corrects) < 2:
                    sample_corrects.append({"question": ex["question"], "expected": expected, "generated": gen_ans})
            else:
                if len(sample_wrongs) < 2:
                    sample_wrongs.append({"question": ex["question"], "expected": expected, "generated": gen_ans})
                    
            if not gen_ans:
                invalid_count += 1
                
            pred_class = gen_ans if gen_ans in confusion[expected] else "OTHER"
            confusion[expected][pred_class] += 1
            
            task_total += 1
            total_count += 1
            
        # Class Balance & Baseline for this split subset
        all_answers = [ex["answer"] for ex in task_exs]
        unique_ans = list(set(all_answers))
        ans_counts = {ans: all_answers.count(ans) for ans in unique_ans}
        majority_count = max(ans_counts.values()) if ans_counts else 0
        baseline = majority_count / len(all_answers) if all_answers else 0.0
        
        acc = task_correct / task_total if task_total > 0 else 0.0
        results_by_task[task] = {
            "accuracy": acc,
            "baseline": baseline,
            "diff_from_baseline": acc - baseline,
            "confusion_matrix": confusion,
            "sample_corrects": sample_corrects,
            "sample_wrongs": sample_wrongs
        }
        
    model.train()
    
    return {
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
        "invalid_rate": invalid_count / total_count if total_count > 0 else 0.0,
        "tasks": results_by_task
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_quant_decision_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=================================================================")
    print("EXP010: Fast32 Quant Decision Classification UE1 Baseline")
    print("=================================================================")
    
    args = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="",
        batch_size=64,
        seq_len=128,
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
    
    # Load audited SFT datasets
    sft_dir = ROOT / "data" / "quant_decision" / "fast32_quant_decision_v1_audited"
    train_data = torch.from_numpy(np.fromfile(sft_dir / "train.bin", dtype=np.uint8)).long()
    val_data = torch.from_numpy(np.fromfile(sft_dir / "val.bin", dtype=np.uint8)).long()
    test_data = torch.from_numpy(np.fromfile(sft_dir / "test.bin", dtype=np.uint8)).long()
    
    g = torch.Generator().manual_seed(args.seed)
    def get_batch():
        return exp004b.sample_batch(train_data, args.batch_size, args.seq_len, device, g)
        
    optimizer, opt_label, opt_fallback = exp004b.make_optimizer(model, args, device)
    amp = args.amp
    
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
    
    optimizer_updates = 0
    forward_loss_calls = 0
    
    for step_idx in range(1, args.steps + 1):
        x, y = get_batch()
        
        # Every step: forward + loss
        with exp004b.autocast_ctx(device, amp):
            out = model(x, return_metadata=False)
            loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            forward_loss_calls += 1
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Every step: optimizer update
        optimizer.step()
        optimizer_updates += 1
        
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        final_loss = lv
        
        # Periodic validation
        if step_idx % args.eval_every == 0:
            val_ce = eval_val_ce()
            print(f"[Step {step_idx:4d}] Train CE: {lv:.4f} | Val CE: {val_ce:.4f}")
            
            # Evaluate SFT accuracy on validation subset (100 samples per task)
            acc_metrics = evaluate_tasks(model, sft_dir / "val.jsonl", device, samples_per_task=100)
            print(f"Validation Overall Accuracy: {acc_metrics['overall_accuracy']:.2%} | Invalid Rate: {acc_metrics['invalid_rate']:.2%}")
            
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
    
    # Final Test evaluation
    print(f"\n=================================================================")
    print(f"LOADING BEST VALIDATION CHECKPOINT")
    print(f"=================================================================")
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
    
    # Evaluate SFT accuracy on test set (100 samples per task)
    test_acc_metrics = evaluate_tasks(model, sft_dir / "test.jsonl", device, samples_per_task=100)
    print(f"Final Test CE: {test_ce:.4f} | Overall Test Accuracy: {test_acc_metrics['overall_accuracy']:.2%}")
    
    # Print task breakdown
    for task, res in test_acc_metrics["tasks"].items():
        print(f"  {task:18s} Accuracy: {res['accuracy']:.2%} (Baseline: {res['baseline']:.2%}, Diff: {res['diff_from_baseline']:+.2%})")
        
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    results = {
        "dataset_name": "fast32_quant_decision_v1_audited",
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
        "optimizer_updates": optimizer_updates,
        "forward_loss_calls": forward_loss_calls,
        "peak_cuda_mem_gb": peak_mem,
        "best_chk_path": str(best_chk_path)
    }
    
    with open(out_dir / "combined_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nQuant Decision UE1 Baseline SFT Completed successfully.")

if __name__ == "__main__":
    main()
