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


def stateful_ssm_token_reference(
    x: Tensor,
    h: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
) -> tuple[Tensor, Tensor]:
    if x.ndim != 1:
        raise ValueError("x must have shape [d_model]")
    if h.ndim != 2:
        raise ValueError("h must have shape [layers, d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if h.shape != a_sig.shape:
        raise ValueError("h and A/B/C/G tensors must have matching shape [layers, d_model]")
    if x.shape[0] != h.shape[1]:
        raise ValueError("x d_model must match h d_model")

    out_x = x.clone()
    out_h = h.clone()
    for layer in range(a_sig.shape[0]):
        h_l = a_sig[layer].to(out_x.dtype) * out_h[layer].to(out_x.dtype) + b[layer].to(out_x.dtype) * out_x
        y_l = c[layer].to(out_x.dtype) * h_l
        out_x = out_x + residual_scale * g_silu[layer].to(out_x.dtype) * y_l
        out_h[layer] = h_l.to(out_h.dtype)
    return out_x, out_h


def stateless_x_only_reference(
    x: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
) -> Tensor:
    if x.ndim != 1:
        raise ValueError("x must have shape [d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if x.shape[0] != a_sig.shape[1]:
        raise ValueError("x d_model must match A/B/C/G d_model")

    out = x.clone()
    for layer in range(a_sig.shape[0]):
        update = g_silu[layer].to(out.dtype) * c[layer].to(out.dtype) * (
            a_sig[layer].to(out.dtype) + b[layer].to(out.dtype)
        )
        out = out + residual_scale * update * out
    return out


def precompose_stateless_master_coeff(coeff: Tensor, *, residual_scale: float = 0.01) -> Tensor:
    if coeff.ndim != 2:
        raise ValueError("coeff must have shape [layers, d_model]")
    return torch.prod(1.0 + residual_scale * coeff.float(), dim=0).to(coeff.dtype)


def precomposed_stateless_reference(x: Tensor, master_coeff: Tensor) -> Tensor:
    if x.ndim != 1 or master_coeff.shape != x.shape:
        raise ValueError("x and master_coeff must have shape [d_model]")
    return x * master_coeff.to(x.dtype)


def shared_state_d_reference(
    x: Tensor,
    h: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
) -> tuple[Tensor, Tensor]:
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x and h must have shape [d_model]")
    if x.shape != h.shape:
        raise ValueError("x and h must have matching shape")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if x.shape[0] != a_sig.shape[1]:
        raise ValueError("x d_model must match A/B/C/G d_model")

    out = x.clone()
    state = h.clone()
    for layer in range(a_sig.shape[0]):
        state = a_sig[layer].to(out.dtype) * state + b[layer].to(out.dtype) * out
        out = out + residual_scale * g_silu[layer].to(out.dtype) * c[layer].to(out.dtype) * state
    return out, state


def compressed_state_reference(
    x: Tensor,
    state: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    *,
    residual_scale: float = 0.01,
) -> tuple[Tensor, Tensor]:
    if x.ndim != 1:
        raise ValueError("x must have shape [d_model]")
    if state.ndim != 2:
        raise ValueError("state must have shape [rank, d_model]")
    if state.shape[1] != x.shape[0]:
        raise ValueError("state d_model must match x d_model")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if x.shape[0] != a_sig.shape[1]:
        raise ValueError("x d_model must match A/B/C/G d_model")

    out = x.clone()
    next_state = state.clone()
    rank = state.shape[0]
    for layer in range(a_sig.shape[0]):
        slot = layer % rank
        h_l = a_sig[layer].to(out.dtype) * next_state[slot] + b[layer].to(out.dtype) * out
        next_state[slot] = h_l
        out = out + residual_scale * g_silu[layer].to(out.dtype) * c[layer].to(out.dtype) * h_l
    return out, next_state


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

    @triton.jit
    def _stateful_ssm_token_kernel(
        x_ptr,
        h_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        g_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        stride_h_l: tl.constexpr,
        stride_h_d: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        x_t = tl.load(x_ptr + d_offsets).to(tl.float32)

        for layer in tl.range(0, layers):
            h_offsets = layer * stride_h_l + d_offsets * stride_h_d
            v_offsets = layer * stride_v_l + d_offsets * stride_v_d
            h_l = tl.load(h_ptr + h_offsets).to(tl.float32)
            a_l = tl.load(a_ptr + v_offsets).to(tl.float32)
            b_l = tl.load(b_ptr + v_offsets).to(tl.float32)
            c_l = tl.load(c_ptr + v_offsets).to(tl.float32)
            g_l = tl.load(g_ptr + v_offsets).to(tl.float32)
            h_l = a_l * h_l + b_l * x_t
            y_t = c_l * h_l
            x_t = x_t + residual_scale * g_l * y_t
            tl.store(h_ptr + h_offsets, h_l)

        tl.store(out_ptr + d_offsets, x_t)

    @triton.jit
    def _empty_kernel(out_ptr):
        tl.store(out_ptr, tl.load(out_ptr))

    @triton.jit
    def _vector_only_kernel(x_ptr, out_ptr, d_model: tl.constexpr, BLOCK_D: tl.constexpr):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + d_offsets, x_t, mask=mask)

    @triton.jit
    def _stateless_x_only_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        g_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        for layer in tl.range(0, layers):
            offsets = layer * stride_v_l + d_offsets * stride_v_d
            a_l = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            b_l = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            c_l = tl.load(c_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            g_l = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            x_t = x_t + residual_scale * g_l * c_l * (a_l + b_l) * x_t
        tl.store(out_ptr + d_offsets, x_t, mask=mask)

    @triton.jit
    def _stateless_x_only_packed_kernel(
        x_ptr,
        packed_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        stride_p_l: tl.constexpr,
        stride_p_d: tl.constexpr,
        stride_p_p: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        for layer in tl.range(0, layers):
            base = layer * stride_p_l + d_offsets * stride_p_d
            a_l = tl.load(packed_ptr + base + 0 * stride_p_p, mask=mask, other=0.0).to(tl.float32)
            b_l = tl.load(packed_ptr + base + 1 * stride_p_p, mask=mask, other=0.0).to(tl.float32)
            c_l = tl.load(packed_ptr + base + 2 * stride_p_p, mask=mask, other=0.0).to(tl.float32)
            g_l = tl.load(packed_ptr + base + 3 * stride_p_p, mask=mask, other=0.0).to(tl.float32)
            x_t = x_t + residual_scale * g_l * c_l * (a_l + b_l) * x_t
        tl.store(out_ptr + d_offsets, x_t, mask=mask)

    @triton.jit
    def _stateless_x_only_coeff_kernel(
        x_ptr,
        coeff_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        for layer in tl.range(0, layers):
            coeff = tl.load(
                coeff_ptr + layer * stride_v_l + d_offsets * stride_v_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_t = x_t + residual_scale * coeff * x_t
        tl.store(out_ptr + d_offsets, x_t, mask=mask)

    @triton.jit
    def _precomposed_stateless_kernel(
        x_ptr,
        master_ptr,
        out_ptr,
        d_model: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        m_t = tl.load(master_ptr + d_offsets, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_ptr + d_offsets, x_t * m_t, mask=mask)

    @triton.jit
    def _rms_project_kernel(
        x_ptr,
        weight_ptr,
        logits_ptr,
        d_model: tl.constexpr,
        vocab_size: tl.constexpr,
        stride_w_v: tl.constexpr,
        stride_w_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < d_model
        v_mask = v_offsets < vocab_size
        x = tl.load(x_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x, axis=0) / d_model + eps)
        x = x / rms
        w = tl.load(
            weight_ptr + v_offsets[:, None] * stride_w_v + d_offsets[None, :] * stride_w_d,
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(w * x[None, :], axis=1)
        tl.store(logits_ptr + v_offsets, logits, mask=v_mask)

    @triton.jit
    def _fused_precomposed_logits_kernel(
        token_ptr,
        weight_ptr,
        master_ptr,
        logits_ptr,
        d_model: tl.constexpr,
        vocab_size: tl.constexpr,
        stride_w_v: tl.constexpr,
        stride_w_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.load(token_ptr)
        v_block = tl.program_id(0)
        v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < d_model
        v_mask = v_offsets < vocab_size
        emb = tl.load(
            weight_ptr + token * stride_w_v + d_offsets * stride_w_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        master = tl.load(master_ptr + d_offsets, mask=d_mask, other=1.0).to(tl.float32)
        x = emb * master
        rms = tl.sqrt(tl.sum(x * x, axis=0) / d_model + eps)
        x = x / rms
        w = tl.load(
            weight_ptr + v_offsets[:, None] * stride_w_v + d_offsets[None, :] * stride_w_d,
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(w * x[None, :], axis=1)
        tl.store(logits_ptr + v_offsets, logits, mask=v_mask)

    @triton.jit
    def _fused_precomposed_top1_kernel(
        token_ptr,
        weight_ptr,
        master_ptr,
        top1_ptr,
        score_ptr,
        d_model: tl.constexpr,
        vocab_size: tl.constexpr,
        stride_w_v: tl.constexpr,
        stride_w_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.load(token_ptr)
        v_offsets = tl.arange(0, BLOCK_V)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < d_model
        v_mask = v_offsets < vocab_size
        emb = tl.load(
            weight_ptr + token * stride_w_v + d_offsets * stride_w_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        master = tl.load(master_ptr + d_offsets, mask=d_mask, other=1.0).to(tl.float32)
        x = emb * master
        rms = tl.sqrt(tl.sum(x * x, axis=0) / d_model + eps)
        x = x / rms
        w = tl.load(
            weight_ptr + v_offsets[:, None] * stride_w_v + d_offsets[None, :] * stride_w_d,
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(w * x[None, :], axis=1)
        logits = tl.where(v_mask, logits, -3.4028234663852886e38)
        max_score = tl.max(logits, axis=0)
        idx_candidates = tl.where(logits == max_score, v_offsets, vocab_size)
        top1 = tl.min(idx_candidates, axis=0)
        tl.store(top1_ptr, top1)
        tl.store(score_ptr, max_score)

    @triton.jit
    def _shared_state_d_kernel(
        x_ptr,
        h_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        g_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < d_model
        x_t = tl.load(x_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        h_t = tl.load(h_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
        for layer in tl.range(0, layers):
            offsets = layer * stride_v_l + d_offsets * stride_v_d
            a_l = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            b_l = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            c_l = tl.load(c_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            g_l = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            h_t = a_l * h_t + b_l * x_t
            x_t = x_t + residual_scale * g_l * c_l * h_t
        tl.store(h_ptr + d_offsets, h_t, mask=mask)
        tl.store(out_ptr + d_offsets, x_t, mask=mask)

    @triton.jit
    def _compressed_state_kernel(
        x_ptr,
        state_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        g_ptr,
        out_ptr,
        layers: tl.constexpr,
        d_model: tl.constexpr,
        state_rank: tl.constexpr,
        stride_s_r: tl.constexpr,
        stride_s_d: tl.constexpr,
        stride_v_l: tl.constexpr,
        stride_v_d: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        r_offsets = tl.arange(0, BLOCK_R)
        d_mask = d_offsets < d_model
        r_mask = r_offsets < state_rank
        x_t = tl.load(x_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        h_vals = tl.load(
            state_ptr + r_offsets[:, None] * stride_s_r + d_offsets[None, :] * stride_s_d,
            mask=r_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        for layer in tl.range(0, layers):
            slot = layer % state_rank
            offsets = layer * stride_v_l + d_offsets * stride_v_d
            a_l = tl.load(a_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
            b_l = tl.load(b_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
            c_l = tl.load(c_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
            g_l = tl.load(g_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
            slot_mask = r_offsets == slot
            h_t = tl.sum(tl.where(slot_mask[:, None], h_vals, 0.0), axis=0)
            h_t = a_l * h_t + b_l * x_t
            h_vals = tl.where(slot_mask[:, None], h_t[None, :], h_vals)
            x_t = x_t + residual_scale * g_l * c_l * h_t
        tl.store(
            state_ptr + r_offsets[:, None] * stride_s_r + d_offsets[None, :] * stride_s_d,
            h_vals,
            mask=r_mask[:, None] & d_mask[None, :],
        )
        tl.store(out_ptr + d_offsets, x_t, mask=d_mask)


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


def stateful_ssm_token_triton_(
    x: Tensor,
    h: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    out: Tensor,
    *,
    residual_scale: float = 0.01,
    block_d: int = 256,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not h.is_cuda or not out.is_cuda:
        raise ValueError("Stateful Triton SSM requires CUDA tensors")
    if x.ndim != 1:
        raise ValueError("x must have shape [d_model]")
    if h.ndim != 2:
        raise ValueError("h must have shape [layers, d_model]")
    if out.shape != x.shape:
        raise ValueError("out must have shape [d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    if h.shape != a_sig.shape:
        raise ValueError("h and A/B/C/G tensors must have matching shape [layers, d_model]")
    if x.shape[0] != h.shape[1]:
        raise ValueError("x d_model must match h d_model")

    layers, d_model = h.shape
    if d_model % block_d != 0:
        raise ValueError("d_model must be divisible by block_d for the stateful exact-shape kernel")
    if block_d <= 0:
        raise ValueError("block_d must be positive")

    x_contig = x.contiguous()
    h_contig = h if h.is_contiguous() else h.contiguous()
    a_contig = a_sig.contiguous()
    b_contig = b.contiguous()
    c_contig = c.contiguous()
    g_contig = g_silu.contiguous()
    out_contig = out.contiguous()
    grid = (triton.cdiv(d_model, block_d),)
    _stateful_ssm_token_kernel[grid](
        x_contig,
        h_contig,
        a_contig,
        b_contig,
        c_contig,
        g_contig,
        out_contig,
        layers,
        d_model,
        h_contig.stride(0),
        h_contig.stride(1),
        a_contig.stride(0),
        a_contig.stride(1),
        residual_scale,
        block_d,
    )
    if h_contig.data_ptr() != h.data_ptr():
        h.copy_(h_contig)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def empty_triton_(out: Tensor) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not out.is_cuda:
        raise ValueError("empty_triton_ requires a CUDA tensor")
    _empty_kernel[(1,)](out)
    return out


def vector_only_triton_(x: Tensor, out: Tensor, *, block_d: int = 256) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not out.is_cuda:
        raise ValueError("vector_only_triton_ requires CUDA tensors")
    if x.ndim != 1 or out.shape != x.shape:
        raise ValueError("x and out must have shape [d_model]")
    d_model = x.shape[0]
    x_contig = x.contiguous()
    out_contig = out.contiguous()
    _vector_only_kernel[(triton.cdiv(d_model, block_d),)](x_contig, out_contig, d_model, block_d)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def stateless_x_only_triton_(
    x: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    out: Tensor,
    *,
    residual_scale: float = 0.01,
    block_d: int = 256,
    packed: Optional[Tensor] = None,
    coeff: Optional[Tensor] = None,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not out.is_cuda:
        raise ValueError("stateless_x_only_triton_ requires CUDA tensors")
    if x.ndim != 1 or out.shape != x.shape:
        raise ValueError("x and out must have shape [d_model]")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    layers, d_model = a_sig.shape
    if x.shape[0] != d_model:
        raise ValueError("x d_model must match A/B/C/G d_model")
    if block_d <= 0:
        raise ValueError("block_d must be positive")

    x_contig = x.contiguous()
    out_contig = out.contiguous()
    grid = (triton.cdiv(d_model, block_d),)
    if coeff is not None:
        coeff_contig = coeff.contiguous()
        _stateless_x_only_coeff_kernel[grid](
            x_contig,
            coeff_contig,
            out_contig,
            layers,
            d_model,
            coeff_contig.stride(0),
            coeff_contig.stride(1),
            residual_scale,
            block_d,
        )
    elif packed is not None:
        packed_contig = packed.contiguous()
        _stateless_x_only_packed_kernel[grid](
            x_contig,
            packed_contig,
            out_contig,
            layers,
            d_model,
            packed_contig.stride(0),
            packed_contig.stride(1),
            packed_contig.stride(2),
            residual_scale,
            block_d,
        )
    else:
        a_contig = a_sig.contiguous()
        b_contig = b.contiguous()
        c_contig = c.contiguous()
        g_contig = g_silu.contiguous()
        _stateless_x_only_kernel[grid](
            x_contig,
            a_contig,
            b_contig,
            c_contig,
            g_contig,
            out_contig,
            layers,
            d_model,
            a_contig.stride(0),
            a_contig.stride(1),
            residual_scale,
            block_d,
        )
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def precomposed_stateless_triton_(
    x: Tensor,
    master_coeff: Tensor,
    out: Tensor,
    *,
    block_d: int = 256,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not master_coeff.is_cuda or not out.is_cuda:
        raise ValueError("precomposed_stateless_triton_ requires CUDA tensors")
    if x.ndim != 1 or master_coeff.shape != x.shape or out.shape != x.shape:
        raise ValueError("x, master_coeff, and out must have shape [d_model]")
    d_model = x.shape[0]
    x_contig = x.contiguous()
    master_contig = master_coeff.contiguous()
    out_contig = out.contiguous()
    _precomposed_stateless_kernel[(triton.cdiv(d_model, block_d),)](
        x_contig,
        master_contig,
        out_contig,
        d_model,
        block_d,
    )
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def rms_project_triton_(
    x: Tensor,
    weight: Tensor,
    logits: Tensor,
    *,
    eps: float = 1e-6,
    block_v: int = 16,
    block_d: int = 256,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda or not weight.is_cuda or not logits.is_cuda:
        raise ValueError("rms_project_triton_ requires CUDA tensors")
    if x.ndim != 1 or weight.ndim != 2 or logits.ndim != 1:
        raise ValueError("x/logits must be 1D and weight must be [vocab, d_model]")
    vocab_size, d_model = weight.shape
    if x.shape[0] != d_model or logits.shape[0] != vocab_size:
        raise ValueError("shape mismatch for RMS projection")
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    logits_contig = logits.contiguous()
    _rms_project_kernel[(triton.cdiv(vocab_size, block_v),)](
        x_contig,
        weight_contig,
        logits_contig,
        d_model,
        vocab_size,
        weight_contig.stride(0),
        weight_contig.stride(1),
        eps,
        block_v,
        block_d,
    )
    if logits_contig.data_ptr() != logits.data_ptr():
        logits.copy_(logits_contig)
    return logits


def fused_precomposed_logits_triton_(
    token: Tensor,
    weight: Tensor,
    master_coeff: Tensor,
    logits: Tensor,
    *,
    eps: float = 1e-6,
    block_v: int = 16,
    block_d: int = 256,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not token.is_cuda or not weight.is_cuda or not master_coeff.is_cuda or not logits.is_cuda:
        raise ValueError("fused_precomposed_logits_triton_ requires CUDA tensors")
    if token.numel() != 1:
        raise ValueError("token must contain one token id")
    vocab_size, d_model = weight.shape
    if master_coeff.shape != (d_model,) or logits.shape != (vocab_size,):
        raise ValueError("shape mismatch for fused precomposed logits")
    token_contig = token.contiguous()
    weight_contig = weight.contiguous()
    master_contig = master_coeff.contiguous()
    logits_contig = logits.contiguous()
    _fused_precomposed_logits_kernel[(triton.cdiv(vocab_size, block_v),)](
        token_contig,
        weight_contig,
        master_contig,
        logits_contig,
        d_model,
        vocab_size,
        weight_contig.stride(0),
        weight_contig.stride(1),
        eps,
        block_v,
        block_d,
    )
    if logits_contig.data_ptr() != logits.data_ptr():
        logits.copy_(logits_contig)
    return logits


def fused_precomposed_top1_triton_(
    token: Tensor,
    weight: Tensor,
    master_coeff: Tensor,
    top1: Tensor,
    score: Tensor,
    *,
    eps: float = 1e-6,
    block_v: int = 256,
    block_d: int = 256,
) -> tuple[Tensor, Tensor]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not token.is_cuda or not weight.is_cuda or not master_coeff.is_cuda or not top1.is_cuda or not score.is_cuda:
        raise ValueError("fused_precomposed_top1_triton_ requires CUDA tensors")
    if token.numel() != 1 or top1.numel() != 1 or score.numel() != 1:
        raise ValueError("token, top1, and score must be scalar tensors")
    vocab_size, d_model = weight.shape
    if vocab_size > block_v:
        raise ValueError("top1 fused kernel currently requires vocab_size <= block_v")
    if master_coeff.shape != (d_model,):
        raise ValueError("master_coeff shape mismatch")
    token_contig = token.contiguous()
    weight_contig = weight.contiguous()
    master_contig = master_coeff.contiguous()
    _fused_precomposed_top1_kernel[(1,)](
        token_contig,
        weight_contig,
        master_contig,
        top1,
        score,
        d_model,
        vocab_size,
        weight_contig.stride(0),
        weight_contig.stride(1),
        eps,
        block_v,
        block_d,
    )
    return top1, score


def shared_state_d_triton_(
    x: Tensor,
    h: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    out: Tensor,
    *,
    residual_scale: float = 0.01,
    block_d: int = 256,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if x.ndim != 1 or h.ndim != 1 or out.shape != x.shape:
        raise ValueError("x, h, and out must have shape [d_model]")
    if not x.is_cuda or not h.is_cuda or not out.is_cuda:
        raise ValueError("shared_state_d_triton_ requires CUDA tensors")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    layers, d_model = a_sig.shape
    if x.shape[0] != d_model or h.shape[0] != d_model:
        raise ValueError("x/h d_model must match A/B/C/G d_model")

    x_contig = x.contiguous()
    h_contig = h if h.is_contiguous() else h.contiguous()
    out_contig = out.contiguous()
    a_contig = a_sig.contiguous()
    b_contig = b.contiguous()
    c_contig = c.contiguous()
    g_contig = g_silu.contiguous()
    _shared_state_d_kernel[(triton.cdiv(d_model, block_d),)](
        x_contig,
        h_contig,
        a_contig,
        b_contig,
        c_contig,
        g_contig,
        out_contig,
        layers,
        d_model,
        a_contig.stride(0),
        a_contig.stride(1),
        residual_scale,
        block_d,
    )
    if h_contig.data_ptr() != h.data_ptr():
        h.copy_(h_contig)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def compressed_state_triton_(
    x: Tensor,
    state: Tensor,
    a_sig: Tensor,
    b: Tensor,
    c: Tensor,
    g_silu: Tensor,
    out: Tensor,
    *,
    residual_scale: float = 0.01,
    block_d: int = 32,
) -> Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if x.ndim != 1 or state.ndim != 2 or out.shape != x.shape:
        raise ValueError("x/out must have shape [d_model] and state must have shape [rank, d_model]")
    if not x.is_cuda or not state.is_cuda or not out.is_cuda:
        raise ValueError("compressed_state_triton_ requires CUDA tensors")
    if a_sig.shape != b.shape or b.shape != c.shape or c.shape != g_silu.shape:
        raise ValueError("A/B/C/G tensors must have matching shape [layers, d_model]")
    layers, d_model = a_sig.shape
    state_rank = state.shape[0]
    if x.shape[0] != d_model or state.shape[1] != d_model:
        raise ValueError("x/state d_model must match A/B/C/G d_model")
    if state_rank <= 0:
        raise ValueError("state rank must be positive")

    x_contig = x.contiguous()
    state_contig = state if state.is_contiguous() else state.contiguous()
    out_contig = out.contiguous()
    a_contig = a_sig.contiguous()
    b_contig = b.contiguous()
    c_contig = c.contiguous()
    g_contig = g_silu.contiguous()
    _compressed_state_kernel[(triton.cdiv(d_model, block_d),)](
        x_contig,
        state_contig,
        a_contig,
        b_contig,
        c_contig,
        g_contig,
        out_contig,
        layers,
        d_model,
        state_rank,
        state_contig.stride(0),
        state_contig.stride(1),
        a_contig.stride(0),
        a_contig.stride(1),
        residual_scale,
        block_d,
        state_rank,
    )
    if state_contig.data_ptr() != state.data_ptr():
        state.copy_(state_contig)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out
