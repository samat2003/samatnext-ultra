import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    compare_tensors,
    fixed_ssm_reference,
    fixed_ssm_triton,
)


def _inputs(batch: int, seq_len: int, d_model: int, layers: int, dtype=torch.float32):
    torch.manual_seed(123)
    x = torch.randn(batch, seq_len, d_model, device="cuda", dtype=dtype) * 0.02
    a = torch.randn(layers, d_model, device="cuda", dtype=torch.float32) * 0.01
    b = torch.randn(layers, d_model, device="cuda", dtype=torch.float32) * 0.01
    c = torch.randn(layers, d_model, device="cuda", dtype=torch.float32) * 0.01
    g = torch.randn(layers, d_model, device="cuda", dtype=torch.float32) * 0.01
    return x, torch.sigmoid(a), b, c, F.silu(g)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required for Experiment 003B Triton tests",
)


@pytest.mark.parametrize("layers", [4, 16, 64])
@pytest.mark.parametrize("d_model", [16, 32])
def test_triton_matches_reference_small_shapes(layers, d_model):
    x, a, b, c, g = _inputs(batch=1, seq_len=8, d_model=d_model, layers=layers)
    expected = fixed_ssm_reference(x, a, b, c, g)
    actual = fixed_ssm_triton(x, a, b, c, g, block_seq=8, block_d=16)
    result = compare_tensors(actual, expected)

    assert result.max_abs_error <= 1e-5
    assert result.mean_abs_error <= 1e-6


def test_triton_matches_reference_1000_layers_d_model_256_small_sequence():
    x, a, b, c, g = _inputs(batch=1, seq_len=1, d_model=256, layers=1000)
    expected = fixed_ssm_reference(x, a, b, c, g)
    actual = fixed_ssm_triton(x, a, b, c, g, block_seq=1, block_d=32)
    result = compare_tensors(actual, expected)

    assert result.max_abs_error <= 1e-5
    assert result.mean_abs_error <= 1e-6


def test_triton_prefix_invariance():
    x, a, b, c, g = _inputs(batch=1, seq_len=8, d_model=32, layers=16)
    mutated = x.clone()
    mutated[:, 5:, :] = mutated[:, 5:, :] + 1.0

    original = fixed_ssm_triton(x, a, b, c, g, block_seq=8, block_d=16)
    changed = fixed_ssm_triton(mutated, a, b, c, g, block_seq=8, block_d=16)

    prefix_diff = (original[:, :5, :] - changed[:, :5, :]).abs().max()
    assert prefix_diff.item() <= 1e-6
