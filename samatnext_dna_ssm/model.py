from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicDnaSsmConfig:
    vocab_size: int = 256
    d_model: int = 256
    max_layers: int = 1_000_000
    chunk_size: int = 1_000
    layer_embed_dim: int = 16
    dna_hidden_dim: int = 128
    halt_threshold: float = 0.999
    min_chunks: int = 1
    residual_scale: float = 0.01
    output_norm_eps: float = 1e-6


@dataclass(frozen=True)
class DynamicDnaSsmOutput:
    logits: Tensor
    layers_used: int
    chunks_used: int
    halt_score: float
    halted: bool
    hidden_rms: float
    logits_rms: float


class DynamicDnaSsmLM(nn.Module):
    def __init__(self, config: Optional[DynamicDnaSsmConfig] = None) -> None:
        super().__init__()
        self.config = config or DynamicDnaSsmConfig()
        self._validate_config()

        cfg = self.config
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.dna = nn.Sequential(
            nn.Linear(cfg.layer_embed_dim, cfg.dna_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.dna_hidden_dim, cfg.dna_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.dna_hidden_dim, 4 * cfg.d_model),
        )
        self._init_dna_output()

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if cfg.d_model <= 0:
            raise ValueError("d_model must be positive")
        if cfg.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if cfg.layer_embed_dim <= 0 or cfg.layer_embed_dim % 2 != 0:
            raise ValueError("layer_embed_dim must be a positive even integer")
        if cfg.dna_hidden_dim <= 0:
            raise ValueError("dna_hidden_dim must be positive")
        if cfg.min_chunks <= 0:
            raise ValueError("min_chunks must be positive")
        if cfg.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if cfg.output_norm_eps <= 0:
            raise ValueError("output_norm_eps must be positive")

    def _init_dna_output(self) -> None:
        final = self.dna[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("expected final DNA module to be nn.Linear")
        nn.init.normal_(final.weight, mean=0.0, std=1e-5)
        with torch.no_grad():
            final.bias.zero_()
            d = self.config.d_model
            final.bias[d : 2 * d].fill_(0.001)
            final.bias[2 * d : 3 * d].fill_(0.001)
            final.bias[3 * d : 4 * d].fill_(0.001)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_index_embedding(self, layer_indices: Tensor) -> Tensor:
        cfg = self.config
        normalized = layer_indices.to(dtype=torch.float32) / float(max(cfg.max_layers - 1, 1))
        half_dim = cfg.layer_embed_dim // 2
        freqs = torch.pow(
            torch.tensor(2.0, device=layer_indices.device, dtype=torch.float32),
            torch.arange(half_dim, device=layer_indices.device, dtype=torch.float32),
        )
        angles = normalized.unsqueeze(-1) * freqs.unsqueeze(0) * (2.0 * torch.pi)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def generate_chunk(self, start_layer: int, chunk_layers: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        layer_indices = torch.arange(start_layer, start_layer + chunk_layers, device=device)
        layer_emb = self.layer_index_embedding(layer_indices)
        generated = self.dna(layer_emb)
        return generated.chunk(4, dim=-1)

    def forward(
        self,
        input_ids: Tensor,
        *,
        halt_threshold: Optional[float] = None,
        min_chunks: Optional[int] = None,
        max_chunks: Optional[int] = None,
        return_metadata: bool = True,
    ) -> DynamicDnaSsmOutput:
        cfg = self.config
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive when provided")

        threshold = cfg.halt_threshold if halt_threshold is None else halt_threshold
        min_chunks_value = cfg.min_chunks if min_chunks is None else min_chunks
        if min_chunks_value <= 0:
            raise ValueError("min_chunks must be positive")

        x = self.token_embed(input_ids)
        chunks_used = 0
        layers_used = 0
        halt_score = 0.0
        halted = False

        for start_layer in range(0, cfg.max_layers, cfg.chunk_size):
            if max_chunks is not None and chunks_used >= max_chunks:
                break

            chunk_layers = min(cfg.chunk_size, cfg.max_layers - start_layer)
            x_before = x
            a_chunk, b_chunk, c_chunk, g_chunk = self.generate_chunk(
                start_layer,
                chunk_layers,
                input_ids.device,
            )

            for layer_offset in range(chunk_layers):
                x = self._apply_ssm_layer(
                    x,
                    a_chunk[layer_offset],
                    b_chunk[layer_offset],
                    c_chunk[layer_offset],
                    g_chunk[layer_offset],
                    cfg.residual_scale,
                )

            chunks_used += 1
            layers_used += chunk_layers
            delta = (x - x_before).abs().mean()
            halt_score_tensor = torch.reciprocal(1.0 + delta)
            if return_metadata:
                halt_score = float(halt_score_tensor.detach().cpu())

            del a_chunk, b_chunk, c_chunk, g_chunk

            if return_metadata:
                if chunks_used >= min_chunks_value and halt_score >= threshold:
                    halted = start_layer + chunk_layers < cfg.max_layers
                    break

        hidden_rms_tensor = torch.sqrt(torch.mean(x.detach() * x.detach()))
        x = self._output_normalize(x, cfg.output_norm_eps)
        logits = F.linear(x, self.token_embed.weight)
        if return_metadata:
            logits_rms_tensor = torch.sqrt(torch.mean(logits.detach() * logits.detach()))
            hidden_rms = float(hidden_rms_tensor.cpu())
            logits_rms = float(logits_rms_tensor.cpu())
        else:
            hidden_rms = 0.0
            logits_rms = 0.0
        return DynamicDnaSsmOutput(
            logits=logits,
            layers_used=layers_used,
            chunks_used=chunks_used,
            halt_score=halt_score,
            halted=halted,
            hidden_rms=hidden_rms,
            logits_rms=logits_rms,
        )

    @staticmethod
    def _output_normalize(x: Tensor, eps: float) -> Tensor:
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _apply_ssm_layer(x: Tensor, a: Tensor, b: Tensor, c: Tensor, g: Tensor, residual_scale: float) -> Tensor:
        retention = torch.sigmoid(a).unsqueeze(0)
        b = b.unsqueeze(0)
        c = c.unsqueeze(0)
        gate = F.silu(g).unsqueeze(0)
        h = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        outputs = []

        for pos in range(x.shape[1]):
            x_t = x[:, pos, :]
            h = retention * h + b * x_t
            y_t = c * h
            outputs.append(x_t + residual_scale * gate * y_t)

        return torch.stack(outputs, dim=1)
