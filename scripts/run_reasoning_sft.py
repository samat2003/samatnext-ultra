import argparse
import gc
import json
import os
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

def generate_answer(model, task_str: str, device: torch.device, max_new_tokens: int = 50) -> str:
    prompt = f"### Task:\n{task_str}\n\n### Answer:\n"
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

def evaluate_accuracy(model, dataset_path: Path, device: torch.device, samples_per_cat: int = 50) -> dict:
    # Load SFT examples from jsonl
    categories = ["arithmetic", "symbolic_reasoning", "boolean_logic", "market_bar_reasoning"]
    cat_samples = {cat: [] for cat in categories}
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            cat = ex["category"]
            if len(cat_samples[cat]) < samples_per_cat:
                cat_samples[cat].append(ex)
            if all(len(cat_samples[c]) >= samples_per_cat for c in categories):
                break
                
    model.eval()
    results = {}
    total_correct = 0
    total_count = 0
    invalid_count = 0
    
    sample_corrects = []
    sample_wrongs = []
    
    for cat in categories:
        correct = 0
        count = 0
        for ex in cat_samples[cat]:
            gen_ans = generate_answer(model, ex["task"], device)
            expected = ex["answer"].strip()
            
            is_correct = (gen_ans == expected)
            if is_correct:
                correct += 1
                total_correct += 1
                if len(sample_corrects) < 3:
                    sample_corrects.append({"task": ex["task"], "expected": expected, "generated": gen_ans})
            else:
                if len(sample_wrongs) < 3:
                    sample_wrongs.append({"task": ex["task"], "expected": expected, "generated": gen_ans})
                    
            if not gen_ans:
                invalid_count += 1
                
            count += 1
            total_count += 1
            
        results[f"{cat}_accuracy"] = correct / count if count > 0 else 0.0
        results[f"{cat}_count"] = count
        
    results["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0.0
    results["invalid_rate"] = invalid_count / total_count if total_count > 0 else 0.0
    results["sample_corrects"] = sample_corrects
    results["sample_wrongs"] = sample_wrongs
    
    model.train()
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_reasoning_sft"
    out_dir.mkdir(exist_ok=True)
    
    print("=================================================================")
    print("EXP008: Fast32 SFT Reasoning Fine-Tuning (UE1)")
    print("=================================================================")
    
    # Configuration matches prompt specs
    args = argparse.Namespace(
        device="cuda",
        data="binance",  # We will load SFT reasoning binary dataset manually below
        dataset_name="",
        batch_size=32,
        seq_len=512,  # seq_len=512 is used, max length is 522 so a few examples might be slightly truncated
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
    
    # Load reasoning dataset packaged in EXP007
    sft_dir = ROOT / "data" / "reasoning_finetune" / "fast32_reasoning_v1"
    train_path = sft_dir / "train.bin"
    val_path = sft_dir / "val.bin"
    test_path = sft_dir / "test.bin"
    
    train_data = torch.from_numpy(np.fromfile(train_path, dtype=np.uint8)).long()
    val_data = torch.from_numpy(np.fromfile(val_path, dtype=np.uint8)).long()
    test_data = torch.from_numpy(np.fromfile(test_path, dtype=np.uint8)).long()
    
    train_token_count = train_data.numel()
    
    g = torch.Generator().manual_seed(args.seed)
    def get_batch():
        return exp004b.sample_batch(train_data, args.batch_size, args.seq_len, device, g)
        
    optimizer, opt_label, opt_fallback = exp004b.make_optimizer(model, args, device)
    use_scaler = (device.type == "cuda" and args.amp == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    
    amp = args.amp
    
    # Helper to calculate cross entropy on validation binary set
    def eval_val_ce():
        model.eval()
        val_losses = []
        g_val = torch.Generator().manual_seed(args.seed + 888)
        with torch.no_grad():
            for _ in range(20):
                vx, vy = exp004b.sample_batch(val_data, args.batch_size, args.seq_len, device, g_val)
                loss_v, _ = exp004b.cached_triton_forward_loss(model, None, vx, vy, amp)
                val_losses.append(loss_v.item())
        model.train()
        return float(np.mean(val_losses))
        
    # --- Warmup ---
    print("Running warmup...")
    for _ in range(args.warmup_steps):
        x, y = get_batch()
        with exp004b.autocast_ctx(device, amp):
            out = model(x, return_metadata=False)
            loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    exp004b.synchronize(device)
    exp004b.reset_peak(device)
    
    # -----------------------------------------------------------------------
    # SFT Training Loop
    # -----------------------------------------------------------------------
    print("Starting SFT training loop...")
    best_val_loss = float("inf")
    best_chk_path = out_dir / "checkpoint_best_val.pt"
    
    history_train_ce = []
    history_val_ce = {}
    history_acc = {}
    
    first_loss = None
    final_loss = None
    
    t_start = time.perf_counter()
    
    for step_idx in range(1, args.steps + 1):
        x, y = get_batch()
        
        with exp004b.autocast_ctx(device, amp):
            out = model(x, return_metadata=False)
            loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        final_loss = lv
        
        # Stop condition 1: NaN/Inf Loss
        if not np.isfinite(lv):
            print(f"ERROR: NaN/Inf loss encountered at step {step_idx}: {lv}. Aborting.")
            sys.exit(1)
            
        # Periodic validation & evaluation
        if step_idx % args.eval_every == 0:
            val_ce = eval_val_ce()
            print(f"\n[Step {step_idx:4d}] Train CE: {lv:.4f} | Val CE: {val_ce:.4f}")
            
            # Save best checkpoint
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                try:
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": step_idx,
                        "best_val_loss": best_val_loss,
                    }, best_chk_path)
                except Exception as exc:
                    print(f"ERROR: Checkpoint save failed: {exc}. Aborting.")
                    sys.exit(1)
                    
            # Check accuracy on validation subset (50 samples per category)
            print("Evaluating exact answer accuracy on validation subset...")
            t_eval_start = time.perf_counter()
            acc_metrics = evaluate_accuracy(model, sft_dir / "val.jsonl", device, samples_per_cat=50)
            t_eval_dur = time.perf_counter() - t_eval_start
            
            print(f"Validation Accuracy: {acc_metrics['overall_accuracy']:.2%} (arithmetic: {acc_metrics['arithmetic_accuracy']:.2%}, symbolic: {acc_metrics['symbolic_reasoning_accuracy']:.2%}, boolean: {acc_metrics['boolean_logic_accuracy']:.2%}, market: {acc_metrics['market_bar_reasoning_accuracy']:.2%}) in {t_eval_dur:.1f}s")
            
            history_val_ce[step_idx] = val_ce
            history_train_ce.append((step_idx, lv))
            history_acc[step_idx] = acc_metrics
            
            # Stop condition 2: Validation CE worsens sharply (divergence)
            if val_ce > best_val_loss + 1.0:
                print(f"Early Stopping: Validation CE worsened sharply ({val_ce:.4f} vs best {best_val_loss:.4f}). Divergence detected.")
                break
                
    exp004b.synchronize(device)
    total_elapsed = time.perf_counter() - t_start
    peak_mem = exp004b.peak_memory(device)
    
    tokens_per_step = args.batch_size * args.seq_len
    train_input_tok_s = (tokens_per_step * step_idx) / total_elapsed
    
    # -----------------------------------------------------------------------
    # Final Test Set Evaluation
    # -----------------------------------------------------------------------
    print("\n=================================================================")
    print("FINAL TEST SET EVALUATION")
    print("=================================================================")
    print(f"Loading best validation checkpoint from {best_chk_path} ...")
    chk = torch.load(best_chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    
    # Calculate test CE
    model.eval()
    test_losses = []
    g_test = torch.Generator().manual_seed(999)
    with torch.no_grad():
        for _ in range(20):
            tx, ty = exp004b.sample_batch(test_data, args.batch_size, args.seq_len, device, g_test)
            loss_t, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, amp)
            test_losses.append(loss_t.item())
    test_ce = float(np.mean(test_losses))
    
    # Exact Answer Accuracy on Test Set (100 samples per category)
    print("Evaluating exact answer accuracy on test subset (100 samples per category)...")
    test_acc_metrics = evaluate_accuracy(model, sft_dir / "test.jsonl", device, samples_per_cat=100)
    
    print(f"Final Test CE: {test_ce:.4f}")
    print(f"Final Test Accuracy: {test_acc_metrics['overall_accuracy']:.2%}")
    print(f"Arithmetic Accuracy: {test_acc_metrics['arithmetic_accuracy']:.2%}")
    print(f"Symbolic Accuracy:   {test_acc_metrics['symbolic_reasoning_accuracy']:.2%}")
    print(f"Boolean Accuracy:    {test_acc_metrics['boolean_logic_accuracy']:.2%}")
    print(f"Market Accuracy:     {test_acc_metrics['market_bar_reasoning_accuracy']:.2%}")
    
    # Write combined SFT results
    combined_results = {
        "dataset_name": "fast32_reasoning_v1",
        "mode": "standard",
        "steps_run": step_idx,
        "first_train_ce": first_loss,
        "final_train_ce": final_loss,
        "best_val_ce": best_val_loss,
        "test_ce": test_ce,
        "test_accuracy_metrics": test_acc_metrics,
        "train_elapsed_sec": total_elapsed,
        "train_input_tok_s": train_input_tok_s,
        "peak_cuda_memory_bytes": peak_mem,
        "history_train_ce": history_train_ce,
        "history_val_ce": history_val_ce,
        "history_acc": history_acc,
        "training_ready": True,
        "best_chk_path": str(best_chk_path),
    }
    
    with open(out_dir / "combined_results_sft.json", "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
        
    print("\nSFT Reasoning Fine-Tuning Completed successfully.")

if __name__ == "__main__":
    main()
