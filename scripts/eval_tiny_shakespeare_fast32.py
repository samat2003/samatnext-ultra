#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate from an Experiment 004 Fast32 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def bytes_decode(values: list[int]) -> str:
    return bytes(int(x) % 256 for x in values).decode("utf-8", errors="replace")


def config_from_payload(payload: dict) -> DynamicDnaSsmConfig:
    cfg_payload = payload["config"]
    return DynamicDnaSsmConfig(**{field.name: cfg_payload[field.name] for field in fields(DynamicDnaSsmConfig)})


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DynamicDnaSsmLM(config_from_payload(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if model.trainable_parameter_count() != 216_320:
        raise RuntimeError(f"parameter count changed: {model.trainable_parameter_count()}")
    model.eval()

    tokens = list(args.prompt.encode("utf-8"))
    seq_len = checkpoint.get("args", {}).get("seq_len", 128)
    with torch.no_grad():
        for _ in range(args.max_new_bytes):
            context = tokens[-seq_len:]
            x = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
            logits = model(x, return_metadata=False).logits[0, -1].float() / args.temperature
            probs = F.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(next_token)
    generated = bytes_decode(tokens)
    payload = {
        "checkpoint": args.checkpoint,
        "parameter_count": model.trainable_parameter_count(),
        "prompt": args.prompt,
        "max_new_bytes": args.max_new_bytes,
        "temperature": args.temperature,
        "generated_text": generated,
        "quality_claim": "generation smoke test only",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
