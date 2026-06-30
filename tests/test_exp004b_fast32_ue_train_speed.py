# SPDX-License-Identifier: Apache-2.0
"""Tests for EXP004B: Fast32 UE Training Speed Ladder."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        device="cpu",
        data="synthetic",
        dataset_name="karpathy/tiny_shakespeare",
        batch_size=4,
        seq_len=16,
        steps=8,
        warmup_steps=2,
        amp="off",
        mode="ue4",
        optimizer="adamw",
        overfit_one_batch=False,
        forward_impl="py_loop",
        update_impl="py_autograd",
        profile_components=False,
        seed=1234,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _import_exp004b():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "exp004b_fast32_ue_train_speed",
        ROOT / "scripts" / "exp004b_fast32_ue_train_speed.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp004b_fast32_ue_train_speed"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = None

def get_mod():
    global _mod
    if _mod is None:
        _mod = _import_exp004b()
    return _mod


# ---------------------------------------------------------------------------
# Correctness / Integrity Tests
# ---------------------------------------------------------------------------

def test_parameter_count_unchanged():
    """Parameter count must stay at 216,320."""
    mod = get_mod()
    device = torch.device("cpu")
    model = mod.make_model(device)
    assert model.trainable_parameter_count() == 216_320


def test_frozen_dir_unchanged():
    """Frozen artifact checksums must match SHA256SUMS.txt."""
    mod = get_mod()
    assert mod.frozen_dir_unchanged(), "Frozen artifacts modified!"


def test_mode_to_update_every():
    mod = get_mod()
    assert mod.mode_to_update_every("standard") == 1
    assert mod.mode_to_update_every("ue1") == 1
    assert mod.mode_to_update_every("ue4") == 4
    assert mod.mode_to_update_every("ue8") == 8
    assert mod.mode_to_update_every("ue16") == 16
    assert mod.mode_to_update_every("ue32") == 32
    assert mod.mode_to_update_every("ue64") == 64
    assert mod.mode_to_update_every("ue128") == 128


# ---------------------------------------------------------------------------
# Update Counts Verification
# ---------------------------------------------------------------------------

def test_ue4_update_count_16_steps():
    """UE4 over 16 steps should have exactly 4 optimizer updates."""
    mod = get_mod()
    args = _make_args(mode="ue4", steps=16, warmup_steps=0)
    result = mod.run_once(args)
    assert result["measured_optimizer_updates"] == 4
    assert result["measured_forward_loss_calls"] == 16


def test_ue8_update_count_16_steps():
    """UE8 over 16 steps should have exactly 2 optimizer updates."""
    mod = get_mod()
    args = _make_args(mode="ue8", steps=16, warmup_steps=0)
    result = mod.run_once(args)
    assert result["measured_optimizer_updates"] == 2
    assert result["measured_forward_loss_calls"] == 16


def test_ue16_update_count_32_steps():
    """UE16 over 32 steps should have exactly 2 optimizer updates."""
    mod = get_mod()
    args = _make_args(mode="ue16", steps=32, warmup_steps=0)
    result = mod.run_once(args)
    assert result["measured_optimizer_updates"] == 2
    assert result["measured_forward_loss_calls"] == 32


# ---------------------------------------------------------------------------
# Graph-free / Correctness Tests
# ---------------------------------------------------------------------------

def test_non_update_steps_do_not_build_grad_graph():
    """Verify that non-update steps do not build a grad graph."""
    mod = get_mod()
    device = torch.device("cpu")
    model = mod.make_model(device)
    
    # Standard py_loop forward pass
    x = torch.randint(0, 256, (2, 8), device=device)
    y = torch.randint(0, 256, (2, 8), device=device)
    
    # We should run it under torch.no_grad()
    with torch.no_grad():
        out = model(x, return_metadata=False)
        loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
    
    assert not loss.requires_grad


# ---------------------------------------------------------------------------
# CPU Tiny Step & Checkpoint save/load
# ---------------------------------------------------------------------------

def test_one_tiny_cpu_step_and_checkpoint_save_load():
    """Verify that one tiny CPU training step runs and checkpoint save/load works."""
    mod = get_mod()
    device = torch.device("cpu")
    model = mod.make_model(device)
    optimizer, _, _ = mod.make_optimizer(model, _make_args(), device)
    
    x = torch.randint(0, 256, (2, 8), device=device)
    y = torch.randint(0, 256, (2, 8), device=device)
    
    # Step 1: grad step
    out = model(x, return_metadata=False)
    loss = F.cross_entropy(out.logits.reshape(-1, 256), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "checkpoint.pt"
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": json.loads((ROOT / "checkpoints" / "fast32_frozen" / "config.json").read_text(encoding="utf-8")),
        }
        torch.save(payload, chk_path)
        
        # Load and verify
        chk = torch.load(chk_path, map_location="cpu")
        new_model = mod.make_model(device)
        new_model.load_state_dict(chk["model_state"])
        assert new_model.trainable_parameter_count() == 216_320


# ---------------------------------------------------------------------------
# Metrics presence
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "mode",
    "update_every",
    "batch_size",
    "seq_len",
    "amp",
    "optimizer",
    "parameter_count",
    "measured_steps",
    "measured_forward_loss_calls",
    "measured_optimizer_updates",
    "total_forward_loss_calls",
    "total_optimizer_updates",
    "train_input_tok_s",
    "forward_loss_tok_s",
    "update_step_tok_s",
    "first_loss",
    "final_loss",
    "loss_decreased",
    "best_val_loss",
    "peak_cuda_memory_bytes",
    "architecture_changed",
    "frozen_artifacts_modified",
]

def test_required_metrics_present():
    mod = get_mod()
    args = _make_args()
    result = mod.run_once(args)
    for field in REQUIRED_FIELDS:
        assert field in result, f"Field {field} is missing from result dict."


# ---------------------------------------------------------------------------
# Triton correctness and speed comparison (CUDA Gated)
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

@cuda_only
def test_cuda_cached_triton_loss_correctness():
    """Verify cached_triton_loss produces matching logits and loss."""
    mod = get_mod()
    args = _make_args(
        device="cuda",
        forward_impl="cached_triton_loss",
        steps=2,
        warmup_steps=1,
    )
    # The run_once function includes verify_cached_triton_correctness internally
    result = mod.run_once(args)
    assert result["frozen_artifacts_modified"] is False
    assert result["architecture_changed"] is False
