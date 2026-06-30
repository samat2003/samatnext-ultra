# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM
from samatnext_dna_ssm.triton_ssm import TRITON_AVAILABLE

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "checkpoints" / "fast32_frozen"


def ensure_artifacts() -> None:
    required = ARTIFACT_DIR / "model_state.pt"
    if not required.exists():
        subprocess.run(["python", "scripts/save_fast32_frozen_artifacts.py"], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fast32_frozen_files_and_checksums() -> None:
    ensure_artifacts()
    expected = {
        "model_state.pt",
        "config.json",
        "architecture.json",
        "benchmark_metadata.json",
        "cached_precomposed_stateless_32.pt",
        "cached_original_stateful_32.pt",
        "SHA256SUMS.txt",
        "README.md",
    }
    assert expected.issubset({path.name for path in ARTIFACT_DIR.iterdir()})
    checksum_lines = (ARTIFACT_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").strip().splitlines()
    checksums = {}
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert expected - {"SHA256SUMS.txt"} == set(checksums)
    for name, digest in checksums.items():
        assert sha256(ARTIFACT_DIR / name) == digest


def test_fast32_frozen_config_and_parameter_count() -> None:
    ensure_artifacts()
    config = json.loads((ARTIFACT_DIR / "config.json").read_text(encoding="utf-8"))
    assert config["vocab_size"] == 256
    assert config["d_model"] == 256
    assert config["max_layers"] == 32
    assert config["chunk_size"] == 32
    assert config["parameter_count"] == 216_320

    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(**{k: config[k] for k in DynamicDnaSsmConfig.__dataclass_fields__}))
    state = torch.load(ARTIFACT_DIR / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    assert model.trainable_parameter_count() == 216_320


def test_fast32_frozen_cached_artifacts_load() -> None:
    ensure_artifacts()
    precomposed = torch.load(ARTIFACT_DIR / "cached_precomposed_stateless_32.pt", map_location="cpu", weights_only=True)
    stateful = torch.load(ARTIFACT_DIR / "cached_original_stateful_32.pt", map_location="cpu", weights_only=True)
    assert precomposed["variant"] == "precomposed_stateless_32_fused_e2e"
    assert stateful["variant"] == "original_stateful_32"
    assert precomposed["master_coeff"].shape == (256,)
    assert stateful["initial_h"].shape == (32, 256)
    for key in ("a_sig", "b", "c", "g_silu"):
        assert precomposed[key].shape == (32, 256)
        assert stateful[key].shape == (32, 256)


def test_fast32_frozen_metadata_records_no_fallback_requirement() -> None:
    ensure_artifacts()
    metadata = json.loads((ARTIFACT_DIR / "benchmark_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parameter_count"] == 216_320
    assert metadata["benchmarks"]["003h_original_stateful_32"]["fallback"] is False
    assert metadata["architecture_caveats"][0].startswith("precomposed_stateless_32_fused_e2e")
    assert isinstance(TRITON_AVAILABLE, bool)
