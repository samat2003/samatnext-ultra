import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    compressed_state_reference,
    compressed_state_triton_,
    shared_state_d_reference,
    shared_state_d_triton_,
    stateless_x_only_reference,
    stateless_x_only_triton_,
    vector_only_triton_,
)


def make_params(layers, d_model, dtype, device):
    torch.manual_seed(9000 + layers)
    x = torch.randn(d_model, device=device, dtype=dtype) * 0.02
    a = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    b = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    c = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    g = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    return x, torch.sigmoid(a).to(dtype), b.to(dtype), c.to(dtype), F.silu(g).to(dtype)


def test_parameter_count_report_still_works():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=1000, chunk_size=1000))
    assert model.trainable_parameter_count() == 216_320


def test_references_return_expected_shapes():
    d_model = 256
    layers = 8
    x, a, b, c, g = make_params(layers, d_model, torch.float32, torch.device("cpu"))
    stateless = stateless_x_only_reference(x, a, b, c, g)
    shared, shared_h = shared_state_d_reference(x, torch.zeros(d_model), a, b, c, g)
    compressed, compressed_h = compressed_state_reference(x, torch.zeros(4, d_model), a, b, c, g)
    assert stateless.shape == (d_model,)
    assert shared.shape == (d_model,)
    assert shared_h.shape == (d_model,)
    assert compressed.shape == (d_model,)
    assert compressed_h.shape == (4, d_model)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("layers", [1, 2, 8, 32, 1000])
@pytest.mark.parametrize("amp", ["fp32", "fp16", "bf16"])
def test_stateless_triton_matches_reference(layers, amp):
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    x, a, b, c, g = make_params(layers, 256, dtype, torch.device("cuda"))
    expected = stateless_x_only_reference(x, a, b, c, g)
    actual = torch.empty_like(x)
    stateless_x_only_triton_(x, a, b, c, g, actual, block_d=256)
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    tol = 2e-5 if amp == "fp32" else 2e-3
    assert diff.max().item() < tol
    assert diff.mean().item() < tol / 4


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_stateless_packed_layout_matches_reference():
    x, a, b, c, g = make_params(32, 256, torch.float16, torch.device("cuda"))
    expected = stateless_x_only_reference(x, a, b, c, g)
    actual = torch.empty_like(x)
    packed = torch.stack([a, b, c, g], dim=-1).contiguous()
    stateless_x_only_triton_(x, a, b, c, g, actual, block_d=128, packed=packed)
    torch.cuda.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_stateless_combined_coeff_layout_matches_reference():
    x, a, b, c, g = make_params(32, 256, torch.float16, torch.device("cuda"))
    expected = stateless_x_only_reference(x, a, b, c, g)
    actual = torch.empty_like(x)
    coeff = (g.float() * c.float() * (a.float() + b.float())).to(torch.float16).contiguous()
    stateless_x_only_triton_(x, a, b, c, g, actual, block_d=128, coeff=coeff)
    torch.cuda.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("layers", [1, 8, 32, 1000])
def test_shared_state_triton_matches_reference(layers):
    x, a, b, c, g = make_params(layers, 256, torch.float16, torch.device("cuda"))
    h = torch.zeros(256, device="cuda", dtype=torch.float16)
    expected, expected_h = shared_state_d_reference(x, h, a, b, c, g)
    actual_h = h.clone()
    actual = torch.empty_like(x)
    shared_state_d_triton_(x, actual_h, a, b, c, g, actual, block_d=256)
    torch.cuda.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("rank", [1, 4, 8, 16, 32])
def test_compressed_state_triton_matches_reference(rank):
    x, a, b, c, g = make_params(32, 256, torch.float16, torch.device("cuda"))
    h = torch.zeros(rank, 256, device="cuda", dtype=torch.float16)
    expected, expected_h = compressed_state_reference(x, h, a, b, c, g)
    actual_h = h.clone()
    actual = torch.empty_like(x)
    block_d = min(256, max(1, 1024 // rank))
    compressed_state_triton_(x, actual_h, a, b, c, g, actual, block_d=block_d)
    torch.cuda.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_vector_only_triton_is_deterministic():
    x = torch.randn(256, device="cuda", dtype=torch.float16)
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    vector_only_triton_(x, out1, block_d=256)
    vector_only_triton_(x, out2, block_d=256)
    torch.cuda.synchronize()
    assert torch.equal(out1, out2)
