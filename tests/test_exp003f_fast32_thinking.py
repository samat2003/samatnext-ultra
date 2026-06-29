import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    compressed_state_reference,
    compressed_state_triton_,
    precompose_stateless_master_coeff,
    precomposed_stateless_reference,
    precomposed_stateless_triton_,
    shared_state_d_reference,
    shared_state_d_triton_,
    stateful_ssm_token_reference,
    stateful_ssm_token_triton_,
    stateless_x_only_reference,
    stateless_x_only_triton_,
)


def make_params(layers, dtype, device):
    torch.manual_seed(3200 + layers)
    d_model = 256
    x = torch.randn(d_model, device=device, dtype=dtype) * 0.02
    a = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    b = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    c = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    g = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    a_sig = torch.sigmoid(a).to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    g_silu = F.silu(g).to(dtype)
    coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).to(dtype)
    return x, a_sig, b, c, g_silu, coeff


def test_parameter_count_unchanged_for_fast32():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=32, chunk_size=32))
    assert model.trainable_parameter_count() == 216_320


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("layers", [1, 2, 8, 16, 32])
def test_original_stateful_32_matches_reference(layers):
    x, a, b, c, g, _ = make_params(layers, torch.float16, torch.device("cuda"))
    h = torch.zeros(layers, 256, device="cuda", dtype=torch.float16)
    expected, expected_h = stateful_ssm_token_reference(x, h, a, b, c, g)
    actual_h = h.clone()
    actual = torch.empty_like(x)
    stateful_ssm_token_triton_(x, actual_h, a, b, c, g, actual, block_d=256)
    torch.cuda.synchronize()
    assert actual.shape == (256,)
    assert actual_h.shape == (layers, 256)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("layers", [1, 2, 8, 16, 32])
def test_stateless_32_matches_reference(layers):
    x, a, b, c, g, coeff = make_params(layers, torch.float16, torch.device("cuda"))
    expected = stateless_x_only_reference(x, a, b, c, g)
    actual = torch.empty_like(x)
    stateless_x_only_triton_(x, a, b, c, g, actual, block_d=256, coeff=coeff)
    torch.cuda.synchronize()
    assert actual.shape == (256,)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_precomposed_stateless_32_matches_running_updates():
    x, a, b, c, g, coeff = make_params(32, torch.float16, torch.device("cuda"))
    running = stateless_x_only_reference(x, a, b, c, g)
    master = precompose_stateless_master_coeff(coeff)
    expected = precomposed_stateless_reference(x, master)
    actual = torch.empty_like(x)
    precomposed_stateless_triton_(x, master, actual, block_d=256)
    torch.cuda.synchronize()
    assert actual.shape == (256,)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (expected.float() - running.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_shared_state_32_matches_reference():
    x, a, b, c, g, _ = make_params(32, torch.float16, torch.device("cuda"))
    h = torch.zeros(256, device="cuda", dtype=torch.float16)
    expected, expected_h = shared_state_d_reference(x, h, a, b, c, g)
    actual_h = h.clone()
    actual = torch.empty_like(x)
    shared_state_d_triton_(x, actual_h, a, b, c, g, actual, block_d=256)
    torch.cuda.synchronize()
    assert actual_h.shape == (256,)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("rank", [1, 8])
def test_compressed_state_32_matches_reference(rank):
    x, a, b, c, g, _ = make_params(32, torch.float16, torch.device("cuda"))
    h = torch.zeros(rank, 256, device="cuda", dtype=torch.float16)
    expected, expected_h = compressed_state_reference(x, h, a, b, c, g)
    actual_h = h.clone()
    actual = torch.empty_like(x)
    compressed_state_triton_(x, actual_h, a, b, c, g, actual, block_d=min(256, 1024 // rank))
    torch.cuda.synchronize()
    assert actual_h.shape == (rank, 256)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3
