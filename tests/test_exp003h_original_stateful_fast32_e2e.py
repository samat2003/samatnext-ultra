import argparse

import pytest
import torch
import torch.nn.functional as F

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE, rms_project_triton_, stateful_ssm_token_reference, stateful_ssm_token_triton_
from scripts.exp003h_original_stateful_fast32_e2e import make_tensors, run_e2e, run_once


def make_args(**overrides):
    base = dict(
        device="cuda",
        layers=32,
        d_model=256,
        vocab_size=256,
        amp="fp16",
        warmup_iters=1,
        measure_iters=2,
        force_triton=True,
        use_cuda_graph=False,
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
def test_cached_params_match_generated(amp):
    args = make_args(amp=amp)
    tensors = make_tensors(args, torch.device("cuda"))
    a, b, c, g = tensors["model"].generate_chunk(0, 32, torch.device("cuda"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    assert torch.equal(torch.sigmoid(a).to(dtype), tensors["a_sig"])
    assert torch.equal(b.to(dtype), tensors["b"])
    assert torch.equal(c.to(dtype), tensors["c"])
    assert torch.equal(F.silu(g).to(dtype), tensors["g_silu"])


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_original_stateful_e2e_logits_match_reference(amp):
    args = make_args(amp=amp)
    tensors = make_tensors(args, torch.device("cuda"))
    run_e2e(args, tensors)
    x = tensors["weight"][tensors["token"][0]]
    expected_hidden, _ = stateful_ssm_token_reference(
        x,
        torch.zeros_like(tensors["stateful_h"]),
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
    )
    expected_hidden = expected_hidden.float()
    expected_hidden = expected_hidden / torch.sqrt(
        torch.mean(expected_hidden * expected_hidden) + tensors["model"].config.output_norm_eps
    )
    expected = F.linear(expected_hidden.unsqueeze(0), tensors["weight"].float()).squeeze(0)
    torch.cuda.synchronize()
    assert tensors["logits"].shape == (256,)
    assert (tensors["logits"].float() - expected.float()).abs().max().item() < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_stateful_kernel_and_projection_shapes():
    args = make_args()
    tensors = make_tensors(args, torch.device("cuda"))
    stateful_ssm_token_triton_(
        tensors["weight"][tensors["token"][0]],
        tensors["stateful_h"],
        tensors["a_sig"],
        tensors["b"],
        tensors["c"],
        tensors["g_silu"],
        tensors["hidden"],
    )
    rms_project_triton_(tensors["hidden"], tensors["weight"], tensors["logits"])
    torch.cuda.synchronize()
    assert tensors["hidden"].shape == (256,)
    assert tensors["stateful_h"].shape == (32, 256)
    assert tensors["logits"].shape == (256,)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_script_reports_no_fallback_and_full_logits():
    result = run_once(make_args(compare_reference=True))
    assert result["fallback_used"] is False
    assert result["triton_kernel_used"] is True
    assert result["full_logits_produced"] is True
    assert result["parameter_count"] == 216_320
    assert result["max_abs_error"] < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_cuda_graph_smoke_reports_usage():
    result = run_once(make_args(use_cuda_graph=True, measure_iters=2, warmup_iters=1))
    assert result["cuda_graph_requested"] is True
    assert result["fallback_used"] is False
    assert result["graph_capture_failure_reason"] is None
