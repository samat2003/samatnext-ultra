import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.exp004b_fast32_ue_train_speed as exp004b

def run_benchmarks():
    modes = ["standard", "ue32"]
    results = {}
    
    out_dir = ROOT / "results_2500_steps"
    out_dir.mkdir(exist_ok=True)

    print("Starting sequential 2500-step Wikipedia benchmarks...")
    
    for mode in modes:
        print(f"\n=========================================")
        print(f"Running MODE: {mode}")
        print(f"=========================================")
        t_start = time.perf_counter()
        
        args = argparse.Namespace(
            device="cuda",
            data="wikipedia",
            dataset_name="wikimedia/wikipedia",
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
        
        try:
            result = exp004b.run_once(args)
            
            # Save individual mode results
            mode_file = out_dir / f"result_{mode}.json"
            mode_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            
            results[mode] = result
            elapsed = time.perf_counter() - t_start
            print(f"\n--> Finished {mode} in {elapsed:.2f} seconds.")
            print(f"Throughput: {result['train_input_tok_s']:.2f} tok/s")
            print(f"First Loss: {result['first_loss']:.4f} | Final Loss: {result['final_loss']:.4f}")
            print(f"Best Val CE: {result['best_val_loss']:.4f}")
            
        except Exception as e:
            print(f"Error running {mode}: {e}")
            import traceback
            traceback.print_exc()
            
        # Clean up CUDA memory between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    # Save combined results
    combined_file = out_dir / "combined_results.json"
    combined_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nAll Wikipedia benchmarks completed successfully.")

if __name__ == "__main__":
    run_benchmarks()
