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

def run_long_pretraining():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_market_pretraining_long"
    out_dir.mkdir(exist_ok=True)
    
    # Standard output checkpoints path
    chk_dir = ROOT / "results_1000_steps"
    chk_dir.mkdir(exist_ok=True)
    
    print("=================================================================")
    print("EXP006: Long 20,000-Step UE32 Market Pretraining")
    print("=================================================================")
    
    args = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="",
        batch_size=64,
        seq_len=256,
        steps=20000,
        warmup_steps=20,
        amp="fp16",
        mode="ue32",
        optimizer="fused-adamw",
        overfit_one_batch=False,
        forward_impl="cached_triton_loss",
        update_impl="py_autograd",
        profile_components=True,
        eval_every=1000,  # Periodic logs
        seed=1234,
    )
    
    # 1. Make model and load Binance dataset
    model = exp004b.make_model(device)
    
    data_dir = ROOT / "data" / "market_pretrain" / "binance_um_futures_1m"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    
    train_data = torch.from_numpy(np.fromfile(train_path, dtype=np.uint8)).long()
    val_data = torch.from_numpy(np.fromfile(val_path, dtype=np.uint8)).long()
    
    train_token_count = train_data.numel()
    
    g = torch.Generator().manual_seed(args.seed)
    def get_batch():
        return exp004b.sample_batch(train_data, args.batch_size, args.seq_len, device, g)
        
    optimizer, opt_label, opt_fallback = exp004b.make_optimizer(model, args, device)
    use_scaler = (device.type == "cuda" and args.amp == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    
    update_every = 32
    amp = args.amp
    
    # Validation helper
    def eval_val():
        model.eval()
        val_losses = []
        g_val = torch.Generator().manual_seed(args.seed + 999)
        with torch.no_grad():
            for _ in range(8):
                vx, vy = exp004b.sample_batch(val_data, args.batch_size, args.seq_len, device, g_val)
                loss_v, _ = exp004b.cached_triton_forward_loss(model, None, vx, vy, amp)
                val_losses.append(loss_v.item())
        model.train()
        return float(np.mean(val_losses))
        
    # --- WARMUP PASS ---
    print("Running warmup steps...")
    cached_params = None
    global_step = 0
    for _ in range(args.warmup_steps):
        x, y = get_batch()
        global_step += 1
        is_update = (global_step % update_every == 0)
        if is_update:
            with exp004b.autocast_ctx(device, amp):
                out = model(x, return_metadata=False)
                loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                a, b, c, g_param = model.generate_chunk(0, model.config.max_layers, device)
                cached_params = (torch.sigmoid(a).contiguous(), b.contiguous(), c.contiguous(), F.silu(g_param).contiguous())
        else:
            with torch.no_grad():
                loss, cached_params = exp004b.cached_triton_forward_loss(model, cached_params, x, y, amp)

    exp004b.synchronize(device)
    exp004b.reset_peak(device)
    
    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    print("Starting 20,000-step training loop...")
    best_val_loss = float("inf")
    patience = 5
    no_improvement_count = 0
    
    history_train_ce = []
    history_val_ce = {}
    
    first_loss = None
    final_loss = None
    measured_optimizer_updates = 0
    measured_forward_loss_calls = 0
    
    t_start = time.perf_counter()
    
    for step_idx in range(1, args.steps + 1):
        x, y = get_batch()
        global_step += 1
        is_update = (global_step % update_every == 0)
        measured_forward_loss_calls += 1
        
        if is_update:
            with exp004b.autocast_ctx(device, amp):
                out = model(x, return_metadata=False)
                loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
                
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                a, b, c, g_param = model.generate_chunk(0, model.config.max_layers, device)
                cached_params = (torch.sigmoid(a).contiguous(), b.contiguous(), c.contiguous(), F.silu(g_param).contiguous())
                
            measured_optimizer_updates += 1
        else:
            with torch.no_grad():
                loss, cached_params = exp004b.cached_triton_forward_loss(model, cached_params, x, y, amp)
                
        lv = float(loss.item())
        if first_loss is None:
            first_loss = lv
        final_loss = lv
        
        # Stop condition 3: NaN/Inf Loss
        if not np.isfinite(lv):
            print(f"ERROR: NaN or Inf loss encountered at step {step_idx}: {lv}. Aborting.")
            sys.exit(1)
            
        # Periodic evaluation every 1000 steps
        if step_idx % 1000 == 0:
            val_loss = eval_val()
            history_val_ce[step_idx] = val_loss
            history_train_ce.append((step_idx, lv))
            
            elapsed_m = (time.perf_counter() - t_start) / 60
            ce_imp_m = (first_loss - lv) / elapsed_m if elapsed_m > 0 else 0.0
            print(f"[Step {step_idx:5d}/20000] Train CE: {lv:.4f} | Val CE: {val_loss:.4f} | CE/min: {ce_imp_m:.3f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                # Stop condition 4: Save validation checkpoint
                try:
                    best_path = out_dir / "checkpoint_best_val.pt"
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": step_idx,
                        "best_val_loss": best_val_loss,
                    }, best_path)
                except Exception as exc:
                    print(f"ERROR: Checkpoint save failed: {exc}. Aborting.")
                    sys.exit(1)
            else:
                no_improvement_count += 1
                
            # Stop condition 1: Validation CE stops improving for 5 consecutive validations
            if no_improvement_count >= patience:
                print(f"Early Stopping: No validation improvement for {patience} evaluations. Best Val CE: {best_val_loss:.4f}")
                break
                
            # Stop condition 2: Validation CE worsens sharply while train CE keeps falling
            # Define "sharp worsening" as Val CE increasing by more than 0.5 from the best Val CE while train CE is below 2.5
            if val_loss > best_val_loss + 0.5 and lv < 2.5:
                print(f"Early Stopping: Validation CE worsened sharply ({val_loss:.4f} vs best {best_val_loss:.4f}) while train CE is falling ({lv:.4f}). Divergence detected.")
                break
                
        # Save checkpoints every 2500 steps
        if step_idx % 2500 == 0:
            chk_path = out_dir / f"checkpoint_step_{step_idx}.pt"
            print(f"Saving periodic checkpoint to {chk_path} ...")
            try:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step_idx,
                    "val_loss": val_loss if step_idx % 1000 == 0 else None,
                }, chk_path)
            except Exception as exc:
                print(f"ERROR: Checkpoint save failed: {exc}. Aborting.")
                sys.exit(1)
                
    exp004b.synchronize(device)
    total_elapsed = time.perf_counter() - t_start
    peak_mem = exp004b.peak_memory(device)
    
    tokens_per_step = args.batch_size * args.seq_len
    train_input_tok_s = (tokens_per_step * measured_forward_loss_calls) / total_elapsed
    effective_train_passes = (tokens_per_step * step_idx) / train_token_count
    
    # Save combined long results
    metrics = {
        "dataset_name": "Binance Futures UM 1m",
        "mode": "ue32",
        "update_every": update_every,
        "steps_run": step_idx,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "final_val_loss": list(history_val_ce.values())[-1] if history_val_ce else None,
        "forward_loss_calls": measured_forward_loss_calls,
        "optimizer_updates": measured_optimizer_updates,
        "train_elapsed_sec": total_elapsed,
        "train_input_tok_s": train_input_tok_s,
        "effective_train_passes": effective_train_passes,
        "peak_cuda_memory_bytes": peak_mem,
        "history_train_ce": history_train_ce,
        "history_val_ce": history_val_ce,
        "training_ready": True,
        "best_chk_path": str(out_dir / "checkpoint_best_val.pt"),
    }
    
    with open(out_dir / "combined_results_long.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print("\nLong training run completed successfully.")
    print(f"Elapsed Time: {total_elapsed:.2f} seconds.")
    print(f"Throughput: {train_input_tok_s:.2f} tok/s")
    print(f"Best Val CE: {best_val_loss:.4f}")

if __name__ == "__main__":
    run_long_pretraining()
