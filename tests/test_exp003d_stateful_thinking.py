import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    stateful_ssm_token_reference,
    stateful_ssm_token_triton_,
)


def test_stateful_reference_shapes_and_parameter_count():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=1000, chunk_size=1000))
    assert model.trainable_parameter_count() == 216_320

    d_model = 256
    layers = 8
    x = torch.randn(d_model)
    h = torch.zeros(layers, d_model)
    a = torch.randn(layers, d_model)
    b = torch.randn(layers, d_model) * 0.01
    c = torch.randn(layers, d_model) * 0.01
    g = torch.randn(layers, d_model) * 0.01
    out, next_h = stateful_ssm_token_reference(x, h, torch.sigmoid(a), b, c, F.silu(g))
    assert out.shape == (d_model,)
    assert next_h.shape == (layers, d_model)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("layers", [1, 2, 8, 32, 1000])
@pytest.mark.parametrize("amp", ["fp32", "fp16", "bf16"])
def test_stateful_triton_matches_reference(layers, amp):
    torch.manual_seed(1234 + layers)
    d_model = 256
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    device = torch.device("cuda")

    x = torch.randn(d_model, device=device, dtype=dtype) * 0.02
    h = torch.randn(layers, d_model, device=device, dtype=dtype) * 0.02
    a = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    b = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    c = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    g = torch.randn(layers, d_model, device=device, dtype=torch.float32) * 0.01
    a_sig = torch.sigmoid(a).to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    g_silu = F.silu(g).to(dtype)

    expected_x, expected_h = stateful_ssm_token_reference(x, h, a_sig, b, c, g_silu)
    actual_h = h.clone()
    actual_x = torch.empty_like(x)
    stateful_ssm_token_triton_(x, actual_h, a_sig, b, c, g_silu, actual_x, block_d=256)
    torch.cuda.synchronize()

    max_x = (actual_x.float() - expected_x.float()).abs().max().item()
    mean_x = (actual_x.float() - expected_x.float()).abs().mean().item()
    max_h = (actual_h.float() - expected_h.float()).abs().max().item()
    tol = 2e-5 if amp == "fp32" else 2e-3
    assert max_x < tol
    assert mean_x < tol / 4
    assert max_h < tol
