import argparse

import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import (
    TRITON_AVAILABLE,
    fused_precomposed_logits_triton_,
    fused_precomposed_top1_triton_,
    precompose_stateless_master_coeff,
    precomposed_stateless_triton_,
    stateful_ssm_token_reference,
    stateful_ssm_token_triton_,
    stateless_x_only_reference,
)
from scripts.exp003g_e2e_fast32_cached import make_tensors, run_once


def make_args(**overrides):
    base = dict(
        device="cuda",
        variant="precomposed_stateless_32_fused_e2e",
        layers=32,
        d_model=256,
        vocab_size=256,
        amp="fp16",
        warmup_iters=2,
        measure_iters=3,
        force_triton=True,
        use_cuda_graph=False,
        profile_components=False,
        audit_allocations=False,
        projection="fused",
        output_mode="full_logits",
        compare_reference=False,
        seed=1234,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parameter_count_remains_216320():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=32, chunk_size=32))
    assert model.trainable_parameter_count() == 216_320


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_cached_params_match_generated_params(amp):
    args = make_args(amp=amp)
    tensors = make_tensors(args, torch.device("cuda"))
    a, b, c, g = tensors["model"].generate_chunk(0, args.layers, torch.device("cuda"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    assert torch.equal(torch.sigmoid(a).to(dtype), tensors["a_sig"])
    assert torch.equal(b.to(dtype), tensors["b"])
    assert torch.equal(c.to(dtype), tensors["c"])
    assert torch.equal(F.silu(g).to(dtype), tensors["g_silu"])


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_cached_master_matches_stateless_updates(amp):
    args = make_args(amp=amp)
    tensors = make_tensors(args, torch.device("cuda"))
    running = stateless_x_only_reference(
        tensors["weight"][tensors["token"][0]],
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
    )
    actual = torch.empty_like(tensors["weight"][0])
    precomposed_stateless_triton_(
        tensors["weight"][tensors["token"][0]],
        tensors["master_coeff"],
        actual,
    )
    torch.cuda.synchronize()
    assert (actual.float() - running.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_fused_logits_match_unfused_reference(amp):
    args = make_args(amp=amp)
    tensors = make_tensors(args, torch.device("cuda"))
    fused_precomposed_logits_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["logits"],
    )
    x = tensors["weight"][tensors["token"][0]] * tensors["master_coeff"]
    x = x.float() / torch.sqrt(torch.mean(x.float() * x.float()) + tensors["model"].config.output_norm_eps)
    expected = F.linear(x.unsqueeze(0), tensors["weight"].float()).squeeze(0)
    torch.cuda.synchronize()
    assert tensors["logits"].shape == (256,)
    assert (tensors["logits"].float() - expected.float()).abs().max().item() < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_fused_top1_matches_full_logits_argmax():
    args = make_args()
    tensors = make_tensors(args, torch.device("cuda"))
    fused_precomposed_logits_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["logits"],
    )
    fused_precomposed_top1_triton_(
        tensors["token"],
        tensors["weight"],
        tensors["master_coeff"],
        tensors["top1"],
        tensors["top1_score"],
    )
    torch.cuda.synchronize()
    assert int(tensors["top1"].item()) == int(torch.argmax(tensors["logits"]).item())


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_original_stateful_cached_matches_reference():
    args = make_args(variant="original_stateful_32_cached_params", projection="triton")
    tensors = make_tensors(args, torch.device("cuda"))
    x = tensors["weight"][tensors["token"][0]]
    expected, expected_h = stateful_ssm_token_reference(
        x,
        tensors["stateful_h"],
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
    )
    actual_h = tensors["stateful_h"].clone()
    actual = torch.empty_like(x)
    stateful_ssm_token_triton_(
        x,
        actual_h,
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
        actual,
    )
    torch.cuda.synchronize()
    assert (actual.float() - expected.float()).abs().max().item() < 2e-3
    assert (actual_h.float() - expected_h.float()).abs().max().item() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_script_reports_no_fallback_for_force_triton():
    args = make_args(measure_iters=2, warmup_iters=1)
    result = run_once(args)
    assert result["fallback_used"] is False
    assert result["triton_kernel_used"] is True
    assert result["parameter_count"] == 216_320


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_fused_output_is_deterministic():
    args = make_args()
    tensors = make_tensors(args, torch.device("cuda"))
    out1 = torch.empty_like(tensors["logits"])
    out2 = torch.empty_like(tensors["logits"])
    fused_precomposed_logits_triton_(tensors["token"], tensors["weight"], tensors["master_coeff"], out1)
    fused_precomposed_logits_triton_(tensors["token"], tensors["weight"], tensors["master_coeff"], out2)
    torch.cuda.synchronize()
    assert torch.equal(out1, out2)
