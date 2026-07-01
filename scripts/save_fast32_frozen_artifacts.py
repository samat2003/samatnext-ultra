#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE, precompose_stateless_master_coeff

OUT_DIR = ROOT / "checkpoints" / "fast32_frozen"
SEED = 1234


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def cuda_device_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(out_dir: Path) -> None:
    files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256(path)}  {path.name}" for path in files]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    config = DynamicDnaSsmConfig(
        vocab_size=256,
        d_model=256,
        max_layers=32,
        chunk_size=32,
        halt_threshold=1.1,
    )
    model = DynamicDnaSsmLM(config)
    model.eval()
    parameter_count = model.trainable_parameter_count()
    if parameter_count != 216_320:
        raise RuntimeError(f"unexpected parameter_count={parameter_count}")

    with torch.no_grad():
        a, b, c, g = model.generate_chunk(0, 32, torch.device("cpu"))
    a_sig = torch.sigmoid(a).contiguous()
    b = b.contiguous()
    c = c.contiguous()
    g_silu = F.silu(g).contiguous()
    coeff = (g_silu.float() * c.float() * (a_sig.float() + b.float())).contiguous()
    master_coeff = precompose_stateless_master_coeff(coeff, residual_scale=config.residual_scale).contiguous()

    torch.save(model.state_dict(), OUT_DIR / "model_state.pt")
    torch.save(
        {
            "variant": "precomposed_stateless_32_fused_e2e",
            "dtype": "fp32_saved_cpu",
            "layers": 32,
            "d_model": 256,
            "formula": "x_out = x_in * product_i(1 + residual_scale * silu(G_i) * C_i * (sigmoid(A_i) + B_i))",
            "a_sig": a_sig,
            "b": b,
            "c": c,
            "g_silu": g_silu,
            "coeff": coeff,
            "master_coeff": master_coeff,
            "residual_scale": config.residual_scale,
        },
        OUT_DIR / "cached_precomposed_stateless_32.pt",
    )
    torch.save(
        {
            "variant": "original_stateful_32",
            "dtype": "fp32_saved_cpu",
            "layers": 32,
            "d_model": 256,
            "state_shape": [32, 256],
            "recurrence": "h = A_sig[layer,d] * h + B[layer,d] * x; x = x + residual_scale * G_silu[layer,d] * C[layer,d] * h",
            "a_sig": a_sig,
            "b": b,
            "c": c,
            "g_silu": g_silu,
            "initial_h": torch.zeros(32, 256),
            "residual_scale": config.residual_scale,
        },
        OUT_DIR / "cached_original_stateful_32.pt",
    )

    config_payload = asdict(config)
    config_payload.update({"seed": SEED, "parameter_count": parameter_count})
    write_json(OUT_DIR / "config.json", config_payload)

    write_json(
        OUT_DIR / "architecture.json",
        {
            "name": "Fast32 frozen Dynamic DNA-SSM inference checkpoint",
            "architecture_frozen_after_commit": "1c8e488b3453f100e59c6b545f4ec646b3e27eca",
            "parameter_count": parameter_count,
            "vocab_size": 256,
            "d_model": 256,
            "layers": 32,
            "chunk_size": 32,
            "trainable_parameter_sources": ["token_embed.weight", "dna hypernetwork"],
            "stored_per_layer_trainable_parameters": False,
            "separate_lm_head": False,
            "trainable_act_head": False,
            "tied_output_projection": True,
            "variants": {
                "precomposed_stateless_32_fused_e2e": "fastest speed ablation; not original stateful SSM",
                "original_stateful_32": "architecture-preserving Fast32 stateful recurrence",
            },
            "no_training_after_freeze": True,
        },
    )

    write_json(
        OUT_DIR / "benchmark_metadata.json",
        {
            "git_commit": git_commit(),
            "frozen_after_commit": "1c8e488b3453f100e59c6b545f4ec646b3e27eca",
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "triton_available": TRITON_AVAILABLE,
            "cuda_available": torch.cuda.is_available(),
            "device_name": cuda_device_name(),
            "parameter_count": parameter_count,
            "d_model": 256,
            "vocab_size": 256,
            "layers": 32,
            "dtype": "fp16 benchmark, fp32 CPU artifact storage",
            "variant_names": ["precomposed_stateless_32_fused_e2e", "original_stateful_32"],
            "benchmarks": {
                "003g_precomposed_stateless_32_fused_e2e": {
                    "full_logits_end_to_end": True,
                    "cuda_graph": True,
                    "amp": "fp16",
                    "mean_us": 7.65,
                    "p99_us": 8.48,
                    "p999_us": 11.87,
                    "max_us": 18.82,
                    "input_tok_s": 130_760,
                    "parameter_count": 216_320,
                    "caveat": "architectural speed ablation, not original stateful SSM",
                },
                "003h_original_stateful_32": {
                    "commit": "1c8e488b3453f100e59c6b545f4ec646b3e27eca",
                    "full_logits_end_to_end": True,
                    "cuda_graph": True,
                    "fallback": False,
                    "amp": "fp16",
                    "mean_us": 13.75,
                    "p99_us": 14.62,
                    "p999_us": 18.08,
                    "max_us": 25.86,
                    "input_tok_s": 72_743,
                    "parameter_count": 216_320,
                    "tests": "95 passed",
                    "caveat": "architecture-preserving Fast32 stateful recurrence",
                },
            },
            "architecture_caveats": [
                "precomposed_stateless_32_fused_e2e is the fastest result but is an architectural speed ablation",
                "original_stateful_32 is the architecture-preserving result",
                "no training has been done after freezing",
            ],
            "license": "Apache-2.0 unless a dependency license says otherwise",
        },
    )

    write_checksums(OUT_DIR)
    print(f"wrote frozen Fast32 artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
