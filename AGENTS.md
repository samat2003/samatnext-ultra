# AGENTS.md

## Role

You are Codex, the coding agent for SamatNext Ultra.

You must implement small, reproducible ML architecture experiments. Be honest about failures, limits, speed, VRAM, and benchmark results.

## Workflow

Before implementation, write a plan first.

The plan must include:
- files to create/edit
- architecture choices
- parameter count estimate
- VRAM strategy
- causality test
- commands to run
- risks/limitations

Do not edit implementation files until the user approves the plan.

After implementation, write a walkthrough with:
- files changed
- commands run
- parameter count
- causality result
- VRAM result
- training/speed result
- limitations

## Active Experiment

Experiment 002: Dynamic 1M-Layer Causal DNA-SSM.

## Non-Negotiable Requirements

Defaults:
- vocab_size = 256
- d_model = 256
- max_layers up to 1,000,000
- chunk_size default = 1,000
- total trainable parameters < 1,000,000

Architecture:
- token embedding
- LM head
- DNA hypernetwork
- causal ephemeral SSM
- ACT / dynamic halting

The generated SSM layers must have zero stored parameters.

DNA hypernetwork:
- input: continuous layer index embedding
- output per layer: A, B, C, G
- each output is shape [d_model]
- never output [d_model, d_model] matrices

Chunked MonoForward:
- do not generate all max_layers at once
- generate one chunk
- run sequence through the chunk
- discard chunk tensors
- decide whether to halt
- continue only if needed

Causal SSM update:
h_t = sigmoid(A_i) * h_{t-1} + B_i * x_t
y_t = C_i * h_t
x_t = x_t + silu(G_i) * y_t

Strict causality:
- output at position t must not depend on future tokens
- include a future-token mutation test

ACT:
- V0 may halt once per chunk
- return layers_used, chunks_used, halt_score, halted

VRAM:
- inference/no_grad should avoid max_layers-sized tensors
- normal PyTorch training through huge depth is not O(1) VRAM unless checkpointing/rematerialization is implemented
- do not claim 1M-layer training O(1) VRAM unless tested

UE4 speed rule:
- UE1: update every step
- UE4: forward/loss every step, backward/optimizer only when step % 4 == 0
- report update_every and optimizer_updates

Forbidden:
- no ModuleList with thousands/millions of layers
- no saved per-layer parameters
- no all-at-once 1M layer generation
- no dense generated SSM matrices
- no future-token leakage
