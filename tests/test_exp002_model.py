import torch

from samatnext_dna_ssm import DynamicDnaSsmConfig, DynamicDnaSsmLM


def test_default_parameter_count_is_expected_and_under_budget():
    model = DynamicDnaSsmLM()
    assert model.trainable_parameter_count() == 216_320
    assert model.trainable_parameter_count() < 1_000_000


def test_no_lm_head_or_trainable_act_head_is_present():
    model = DynamicDnaSsmLM()
    module_names = set(dict(model.named_modules()))
    parameter_names = set(dict(model.named_parameters()))

    assert "lm_head" not in module_names
    assert "halt_head" not in module_names
    assert "act_head" not in module_names
    assert all("lm_head" not in name for name in parameter_names)
    assert all("halt" not in name and "act" not in name for name in parameter_names)


def test_forward_shapes_and_chunk_metadata():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=8, chunk_size=4, halt_threshold=1.1))
    tokens = torch.randint(0, 256, (2, 6))

    output = model(tokens)

    assert output.logits.shape == (2, 6, 256)
    assert output.layers_used == 8
    assert output.chunks_used == 2
    assert isinstance(output.halt_score, float)
    assert output.halted is False


def test_max_chunks_caps_execution_without_max_layers_allocation():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=1_000_000, chunk_size=1_000, halt_threshold=1.1))
    tokens = torch.randint(0, 256, (1, 4))

    with torch.no_grad():
        output = model(tokens, max_chunks=3)

    assert output.layers_used == 3_000
    assert output.chunks_used == 3
    assert output.halted is False


def test_generated_ssm_shapes_are_vectors_not_dense_matrices():
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=8, chunk_size=4))
    generated = model.generate_chunk(0, 4, torch.device("cpu"))

    assert len(generated) == 4
    assert all(tensor.shape == (4, 256) for tensor in generated)
    assert all(tensor.ndim == 2 for tensor in generated)


def test_future_token_mutation_does_not_change_prefix_logits():
    torch.manual_seed(7)
    model = DynamicDnaSsmLM(DynamicDnaSsmConfig(max_layers=8, chunk_size=4, halt_threshold=1.1))
    model.eval()
    tokens = torch.randint(0, 256, (2, 10))
    mutated = tokens.clone()
    cutoff = 4
    mutated[:, cutoff + 1 :] = (mutated[:, cutoff + 1 :] + 31) % 256

    with torch.no_grad():
        original_output = model(tokens)
        mutated_output = model(mutated)

    max_diff = (
        original_output.logits[:, : cutoff + 1] - mutated_output.logits[:, : cutoff + 1]
    ).abs().max()
    assert max_diff.item() <= 1e-6
