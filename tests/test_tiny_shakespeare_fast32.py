# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from scripts.train_tiny_shakespeare_fast32 import bytes_decode, bytes_encode, sample_batch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "checkpoints" / "fast32_frozen"


def file_hashes() -> dict[str, str]:
    hashes = {}
    for path in sorted(FROZEN_DIR.iterdir()):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path.name] = digest
    return hashes


def test_byte_roundtrip() -> None:
    text = "ROMEO:\nBut soft, what light?\n"
    assert bytes_decode(bytes_encode(text)) == text


def test_batch_shapes_and_shifted_labels() -> None:
    data = torch.arange(512, dtype=torch.long) % 256
    generator = torch.Generator().manual_seed(0)
    x, y = sample_batch(data, batch_size=4, seq_len=16, device=torch.device("cpu"), generator=generator)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_fast32_parameter_count() -> None:
    config_payload = json.loads((FROZEN_DIR / "config.json").read_text(encoding="utf-8"))
    config = DynamicDnaSsmConfig(
        vocab_size=config_payload["vocab_size"],
        d_model=config_payload["d_model"],
        max_layers=config_payload["max_layers"],
        chunk_size=config_payload["chunk_size"],
        layer_embed_dim=config_payload["layer_embed_dim"],
        dna_hidden_dim=config_payload["dna_hidden_dim"],
        halt_threshold=config_payload["halt_threshold"],
        min_chunks=config_payload["min_chunks"],
        residual_scale=config_payload["residual_scale"],
        output_norm_eps=config_payload["output_norm_eps"],
    )
    model = DynamicDnaSsmLM(config)
    assert model.trainable_parameter_count() == 216_320


def test_one_tiny_training_step_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    before = file_hashes()
    out_dir = tmp_path / "run"
    subprocess.run(
        [
            "python",
            "scripts/train_tiny_shakespeare_fast32.py",
            "--device",
            "cpu",
            "--dataset-name",
            "tiny_shakespeare",
            "--seq-len",
            "16",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--lr",
            "1e-4",
            "--amp",
            "off",
            "--overfit-one-batch",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(**{k: checkpoint["config"][k] for k in DynamicDnaSsmConfig.__dataclass_fields__}))
    model.load_state_dict(checkpoint["model_state"])
    assert model.trainable_parameter_count() == 216_320
    assert (out_dir / "metrics.json").exists()
    assert file_hashes() == before
