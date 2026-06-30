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

def evaluate_test_ce(checkpoint_path: Path, test_data: torch.Tensor, device: torch.device) -> float:
    print(f"Evaluating test CE using checkpoint: {checkpoint_path}")
    # Load model and state dict
    model = exp004b.make_model(device)
    chk = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    model.eval()
    
    # Run test evaluation over 100 deterministic batches
    g_test = torch.Generator().manual_seed(9999)
    test_losses = []
    with torch.no_grad():
        for _ in range(100):
            tx, ty = exp004b.sample_batch(test_data, 64, 256, device, g_test)
            # Use cached triton forward path or py_loop forward pass
            loss, _ = exp004b.cached_triton_forward_loss(model, None, tx, ty, "fp16")
            test_losses.append(loss.item())
            
    return float(np.mean(test_losses))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results_market_pretraining"
    out_dir.mkdir(exist_ok=True)
    
    # -----------------------------------------------------------------------
    # Phase 1: UE32 Sanity Check
    # -----------------------------------------------------------------------
    print("=================================================================")
    print("PHASE 1: UE32 Sanity Check (500 Steps)")
    print("=================================================================")
    
    args_sanity = argparse.Namespace(
        device="cuda",
        data="binance",
        dataset_name="",
        batch_size=64,
        seq_len=256,
        steps=500,
        warmup_steps=20,
        amp="fp16",
        mode="ue32",
        optimizer="fused-adamw",
        overfit_one_batch=False,
        forward_impl="cached_triton_loss",
        update_impl="py_autograd",
        profile_components=True,
        eval_every=100,
        seed=1234,
    )
    
    # Clean checkpoints from results_1000_steps to prevent mixing
    chk_sanity_path = ROOT / "results_1000_steps" / "checkpoint_ue32.pt"
    if chk_sanity_path.exists():
        chk_sanity_path.unlink()
        
    res_sanity = exp004b.run_once(args_sanity)
    
    print(f"Sanity Check Complete. Train CE: {res_sanity['first_loss']:.4f} -> {res_sanity['final_loss']:.4f}")
    if not res_sanity["loss_decreased"]:
        print("ERROR: UE32 Sanity check failed! Training CE did not decrease.")
        sys.exit(1)
    print("UE32 Sanity check passed successfully! Proceeding to Phase 2...")
    
    # -----------------------------------------------------------------------
    # Phase 2: Full 2500-Step UE1 vs UE32 Comparison
    # -----------------------------------------------------------------------
    print("\n=================================================================")
    print("PHASE 2: Full 2500-Step UE1 vs UE32 Comparison")
    print("=================================================================")
    
    modes = ["standard", "ue32"]
    results = {}
    
    for mode in modes:
        print(f"\nRunning training for MODE: {mode} ...")
        args_train = argparse.Namespace(
            device="cuda",
            data="binance",
            dataset_name="",
            batch_size=64,
            seq_len=256,
            steps=2500,
            warmup_steps=20,
            amp="fp16",
            mode=mode,
            optimizer="fused-adamw",
            overfit_one_batch=False,
            forward_impl="cached_triton_loss",
            update_impl="py_autograd",
            profile_components=True,
            eval_every=250,
            seed=1234,
        )
        
        # Clean specific checkpoint
        chk_path = ROOT / "results_1000_steps" / f"checkpoint_{mode}.pt"
        if chk_path.exists():
            chk_path.unlink()
            
        res = exp004b.run_once(args_train)
        results[mode] = res
        
        gc.collect()
        torch.cuda.empty_cache()
        
    # -----------------------------------------------------------------------
    # Phase 3: Test CE Evaluation & Comparative Report
    # -----------------------------------------------------------------------
    print("\n=================================================================")
    print("PHASE 3: Test CE Evaluation")
    print("=================================================================")
    
    # Load test data
    test_path = ROOT / "data" / "market_pretrain" / "binance_um_futures_1m" / "test.bin"
    test_data = torch.from_numpy(np.fromfile(test_path, dtype=np.uint8)).long()
    
    test_ce_standard = evaluate_test_ce(ROOT / "results_1000_steps" / "checkpoint_standard.pt", test_data, device)
    test_ce_ue32 = evaluate_test_ce(ROOT / "results_1000_steps" / "checkpoint_ue32.pt", test_data, device)
    
    results["standard"]["test_ce"] = test_ce_standard
    results["ue32"]["test_ce"] = test_ce_ue32
    
    # Write combined results
    combined_file = out_dir / "combined_results.json"
    combined_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # Print quick report
    print("\n=================================================================")
    print("COMPARATIVE REPORT:")
    print("=================================================================")
    print(f"Standard (UE1) Test CE: {test_ce_standard:.4f}")
    print(f"UE32 Test CE:           {test_ce_ue32:.4f}")
    print(f"Test CE Gap (UE32-UE1):  {test_ce_ue32 - test_ce_standard:.4f}")
    
    speed_std = results["standard"]["train_input_tok_s"]
    speed_ue32 = results["ue32"]["train_input_tok_s"]
    print(f"Standard Throughput:    {speed_std:.1f} tok/s")
    print(f"UE32 Throughput:        {speed_ue32:.1f} tok/s")
    print(f"Speedup:                {speed_ue32 / speed_std:.2f}x")

if __name__ == "__main__":
    main()
