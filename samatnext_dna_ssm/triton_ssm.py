from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - depends on local optional dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class SsmCompareResult:
    max_abs_error: float
    mean_abs_error: float


def fixed_ssm_reference(
    x: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
) -> Tensor:
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, seq, d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if a_sig.shape[1] != x.shape[2]:
        raise ValueError("SSM vectors must match x d_model")

    out = x.clone()
    batch, seq_len, d_model = out.shape
    for layer in range(a_sig.shape[0]):
        h = torch.zeros(batch, d_model, device=out.device, dtype=out.dtype)
        a_l = a_sig[layer].to(out.dtype).unsqueeze(0)
        b_l = b[layer].to(out.dtype).unsqueeze(0)
        c_l = c[layer].to(out.dtype).unsqueeze(0)
        g_l = g_silu[layer].to(out.dtype).unsqueeze(0)
        for token in range(seq_len):
            x_t = out[:, token, :]
            h = a_l * h + b_l * x_t
            y_t = c_l * h
            out[:, token, :] = x_t + residual_scale * g_l * y_t
    return out


def compare_tensors(actual: Tensor, expected: Tensor) -> SsmCompareResult:
    diff = (actual.float() - expected.float()).abs()
    return SsmCompareResult(
        max_abs_error=float(diff.max().detach().cpu()),
        mean_abs_error=float(diff.mean().detach().cpu()),
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _fixed_ssm_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        g_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        d_model: tl.constexpr,
        layers: tl.constexpr,
        stride_x_b: tl.constexpr,
        stride_x_t: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        d_block = tl.program_id(1)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < d_model
        token_offsets = tl.arange(0, BLOCK_SEQ)
        token_mask = token_offsets < seq_len

        x_vals = tl.load(
            x_ptr
            + batch_id * stride_x_b
            + token_offsets[:, None] * stride_x_t
            + d_offsets[None, :] * stride_x_d,
            mask=token_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        h = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for layer in tl.range(0, layers):
            a_l = tl.load(a_ptr + layer * stride_v_l + d_offsets * stride_v_d, mask=d_mask, other=0.0).to(tl.float32)
            b_l = tl.load(b_ptr + layer * stride_v_l + d_offsets * stride_v_d, mask=d_mask, other=0.0).to(tl.float32)
            c_l = tl.load(c_ptr + layer * stride_v_l + d_offsets * stride_v_d, mask=d_mask, other=0.0).to(tl.float32)
            g_l = tl.load(g_ptr + layer * stride_v_l + d_offsets * stride_v_d, mask=d_mask, other=0.0).to(tl.float32)
            for token in tl.range(0, BLOCK_SEQ):
                row_mask = token_offsets == token
                x_t = tl.sum(tl.where(row_mask[:, None], x_vals, 0.0), axis=0)
                h = a_l * h + b_l * x_t
                y_t = c_l * h
                x_t = x_t + residual_scale * g_l * y_t
                x_vals = tl.where(row_mask[:, None], x_t[None, :], x_vals)

        tl.store(
            out_ptr
            + batch_id * stride_x_b
            + token_offsets[:, None] * stride_x_t
            + d_offsets[None, :] * stride_x_d,
            x_vals,
            mask=token_mask[:, None] & d_mask[None, :],
        )


def fixed_ssm_triton(
    x: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
    block_seq: Optional[int] = None,
    block_d: int = 32,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise ValueError("Triton SSM requires CUDA tensors")
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, seq, d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")

    batch, seq_len, d_model = x.shape
    layers = a_sig.shape[0]
    if a_sig.shape[1] != d_model:
        raise ValueError("SSM vectors must match x d_model")
    if block_seq is None:
        block_seq = triton.next_power_of_2(seq_len)
    if seq_len > block_seq:
        raise ValueError("V0 Triton kernel requires seq_len <= block_seq")
    if block_d <= 0:
        raise ValueError("block_d must be positive")

    x_contig = x.contiguous()
    a_contig = a_sig.contiguous()
    b_contig = b.contiguous()
    c_contig = c.contiguous()
    g_contig = g_silu.contiguous()
    out = torch.empty_like(x_contig)
    grid = (batch, triton.cdiv(d_model, block_d))
    _fixed_ssm_kernel[grid](
        x_contig,
        a_contig,
        b_contig,
        c_contig,
        g_contig,
        out,
        seq_len,
        d_model,
        layers,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        a_contig.stride(0),
        a_contig.stride(1),
        residual_scale,
        block_seq,
        block_d,
    )
    return out
