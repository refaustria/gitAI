"""Interpretability tools.

The tests that matter here are the ones asserting a decomposition actually
reconstructs what it claims to explain. An attribution method whose parts do not
add back to the whole is telling you a story, not a fact.
"""

from __future__ import annotations

import math

import pytest
import torch

from gitai.interpret import (
    ablate_heads,
    ablation_table,
    attention_heatmap,
    attribute_heads,
    attribute_logit,
    attribution_table,
    bits_per_byte,
    colour_by_surprisal,
    lens_table,
    lens_trajectory,
    logit_difference,
    logit_lens,
    loss_metric,
    patch_residual,
    predict_tokens,
    surprisal_bits,
    visible,
)
from gitai.model import ModelConfig, Transformer
from gitai.tokenizer import CharTokenizer

VOCAB = 96


class _IdentityTokenizer:
    """Decodes an id to its own number, so tests read clearly."""

    def decode(self, ids):
        return f"<{ids[0]}>"


def build(**overrides) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(
        **{
            "vocab_size": VOCAB,
            "seq_len": 32,
            "d_model": 48,
            "n_layer": 3,
            "n_head": 4,
            **overrides,
        }
    )
    model = Transformer(config)
    model.eval()
    return model


@pytest.fixture
def model() -> Transformer:
    return build()


@pytest.fixture
def ids() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randint(0, VOCAB, (1, 12))


# ------------------------------------------------------------------ surprisal


def test_uniform_model_has_log2_vocab_surprisal():
    """A model predicting uniformly is surprised by exactly log2(V) bits."""
    logits = torch.zeros(1, 5, VOCAB)
    targets = torch.randint(0, VOCAB, (1, 5))
    assert torch.allclose(
        surprisal_bits(logits, targets), torch.tensor(math.log2(VOCAB)), atol=1e-5
    )


def test_mean_surprisal_equals_the_loss_in_bits(model, ids):
    """Surprisal is the loss, decomposed by position — the same number in a
    different unit. If these disagree, one of them is wrong."""
    inputs, targets = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        logits, loss = model(inputs, targets)
    assert surprisal_bits(logits, targets).mean().item() == pytest.approx(
        loss.item() / math.log(2), abs=1e-4
    )


def test_confident_prediction_has_near_zero_surprisal():
    logits = torch.full((1, 1, VOCAB), -20.0)
    logits[0, 0, 7] = 20.0
    assert surprisal_bits(logits, torch.tensor([[7]])).item() < 1e-5


def test_bits_per_byte():
    assert bits_per_byte(800.0, 400) == 2.0
    with pytest.raises(ValueError, match="must be positive"):
        bits_per_byte(1.0, 0)


def test_predict_tokens_reports_every_position(model, ids):
    predictions = predict_tokens(model, _IdentityTokenizer(), ids[0], top_k=3)
    assert len(predictions) == ids.shape[1] - 1
    first = predictions[0]
    assert first.position == 0
    assert first.actual_id == int(ids[0, 1])
    assert len(first.top) == 3
    assert 1 <= first.rank <= VOCAB
    assert first.surprisal >= 0.0


def test_top_predictions_are_sorted_descending(model, ids):
    for prediction in predict_tokens(model, _IdentityTokenizer(), ids[0]):
        probabilities = [p for _, p in prediction.top]
        assert probabilities == sorted(probabilities, reverse=True)


def test_rank_one_means_lowest_surprisal_of_all_tokens(model, ids):
    predictions = predict_tokens(model, _IdentityTokenizer(), ids[0])
    ranked_first = [p for p in predictions if p.rank == 1]
    for prediction in ranked_first:
        assert prediction.surprisal <= math.log2(VOCAB)


def test_predict_tokens_rejects_bad_input(model):
    with pytest.raises(ValueError, match="1-D"):
        predict_tokens(model, _IdentityTokenizer(), torch.randint(0, VOCAB, (2, 4)))
    with pytest.raises(ValueError, match="at least two tokens"):
        predict_tokens(model, _IdentityTokenizer(), torch.tensor([3]))


def test_predict_tokens_restores_training_mode(ids):
    m = build()
    m.train()
    predict_tokens(m, _IdentityTokenizer(), ids[0])
    assert m.training


# ---------------------------------------------------------------- logit lens


def test_lens_final_layer_reproduces_the_model_output_exactly(model, ids):
    """The strongest check on the lens: decoding the *last* residual state is
    not an approximation — it is literally what the model does. If this fails,
    the lens is reading the wrong tensor."""
    with torch.no_grad():
        logits, _, cache = model.run_with_cache(ids)
        lens = logit_lens(model, cache)
    assert lens.shape[0] == model.config.n_layer + 1
    assert torch.allclose(lens[-1], logits, atol=1e-5)


def test_lens_has_one_row_per_layer_plus_the_embedding(model, ids):
    rows = lens_trajectory(model, _IdentityTokenizer(), ids[0], position=4)
    assert len(rows) == model.config.n_layer + 1
    assert rows[0].label == "embed"
    assert rows[0].layer == -1
    assert rows[-1].label == f"block {model.config.n_layer - 1}"


def test_lens_probabilities_are_valid(model, ids):
    for row in lens_trajectory(model, _IdentityTokenizer(), ids[0], position=4):
        assert 0.0 <= row.target_prob <= 1.0
        assert 1 <= row.target_rank <= VOCAB
        assert sum(p for _, p in row.top) <= 1.0 + 1e-5


def test_lens_rejects_a_position_with_nothing_after_it(model, ids):
    with pytest.raises(ValueError, match="no following token"):
        lens_trajectory(model, _IdentityTokenizer(), ids[0], position=ids.shape[1] - 1)


# --------------------------------------------------------------- attribution


@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
def test_attribution_sums_to_the_logit(norm):
    """The decomposition must reconstruct what it explains, for both norms.

    Both are affine in the residual once the per-position scale is taken from
    the real forward pass, which is what makes this exact rather than a
    linearisation.
    """
    model = build(norm=norm, bias=(norm == "layernorm"))
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (1, 10))
    with torch.no_grad():
        logits, _, cache = model.run_with_cache(ids)
    for position in (0, 4, 9):
        for token in (0, 17, VOCAB - 1):
            contributions, offset = attribute_logit(model, cache, position, token)
            total = sum(c.logit for c in contributions) + offset
            assert total == pytest.approx(float(logits[0, position, token]), abs=2e-3)


def test_attribution_has_one_entry_per_component(model, ids):
    with torch.no_grad():
        _, _, cache = model.run_with_cache(ids)
    contributions, _ = attribute_logit(model, cache, 5, 11)
    assert len(contributions) == 1 + 2 * model.config.n_layer
    assert {c.kind for c in contributions} == {"embedding", "attention", "mlp"}


def test_head_attribution_sums_to_the_layer_attention_contribution(model, ids):
    """Attention output is concat(heads) @ W_O, and that matmul is a sum over
    head-sized slices — so per-head attribution is exactly separable."""
    with torch.no_grad():
        _, _, cache = model.run_with_cache(ids)
    position, token = 6, 23
    per_head = attribute_heads(model, cache, position, token)
    contributions, _ = attribute_logit(model, cache, position, token)
    by_name = {c.name: c.logit for c in contributions}
    for layer in range(model.config.n_layer):
        # Equal up to the attention output bias, which belongs to no head.
        assert float(per_head[layer].sum()) == pytest.approx(by_name[f"L{layer}.attn"], abs=2e-3)


def test_head_attribution_shape(model, ids):
    with torch.no_grad():
        _, _, cache = model.run_with_cache(ids)
    assert attribute_heads(model, cache, 5, 3).shape == (model.config.n_layer, model.config.n_head)


# ------------------------------------------------------------------ patching


def test_patching_the_last_layer_restores_the_clean_logits(model):
    """Patching the final block's output at position p must make the logits at p
    exactly the clean ones — everything downstream of that point is determined."""
    torch.manual_seed(4)
    clean = torch.randint(0, VOCAB, (1, 10))
    corrupt = torch.randint(0, VOCAB, (1, 10))
    with torch.no_grad():
        clean_logits, _ = model(clean)

    position, token = 7, 31
    metric = logit_difference(token, 0, position=position)
    patched = patch_residual(model, clean, corrupt, metric, positions=[position])
    expected = float(clean_logits[0, position, token] - clean_logits[0, position, 0])
    assert float(patched[-1, 0]) == pytest.approx(expected, abs=1e-4)


def test_patching_sweeps_every_layer_and_position(model):
    torch.manual_seed(5)
    clean = torch.randint(0, VOCAB, (1, 8))
    corrupt = torch.randint(0, VOCAB, (1, 8))
    results = patch_residual(model, clean, corrupt, lambda lg: float(lg[0, 4, 12]))
    assert results.shape == (model.config.n_layer, 8)
    assert torch.isfinite(results).all()
    # Patching different sites must not all give the identical answer, or the
    # intervention is not actually being applied.
    assert results.unique().numel() > 1


def test_loss_metric_matches_the_model_loss(model):
    torch.manual_seed(6)
    ids = torch.randint(0, VOCAB, (1, 9))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        logits, loss = model(inputs, targets)
    assert loss_metric(targets)(logits) == pytest.approx(loss.item(), abs=1e-5)


def test_patching_requires_aligned_inputs(model):
    with pytest.raises(ValueError, match="position-for-position"):
        patch_residual(
            model,
            torch.randint(0, VOCAB, (1, 8)),
            torch.randint(0, VOCAB, (1, 6)),
            lambda lg: 0.0,
        )


def test_patching_removes_its_hooks(model):
    """A leaked forward hook silently corrupts every later forward pass."""
    clean = torch.randint(0, VOCAB, (1, 6))
    corrupt = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        before, _ = model(corrupt)
    patch_residual(model, clean, corrupt, lambda lg: float(lg.sum()))
    with torch.no_grad():
        after, _ = model(corrupt)
    assert torch.equal(before, after)
    assert all(not block._forward_hooks for block in model.blocks)


# ------------------------------------------------------------------ ablation


def test_ablation_returns_a_delta_per_head(model, ids):
    deltas = ablate_heads(model, ids[:, :-1], ids[:, 1:])
    assert deltas.shape == (model.config.n_layer, model.config.n_head)
    assert torch.isfinite(deltas).all()
    assert deltas.abs().sum() > 0, "ablating every head in turn changed nothing"


def test_ablation_leaves_the_model_unchanged(model, ids):
    with torch.no_grad():
        before, _ = model(ids)
    ablate_heads(model, ids[:, :-1], ids[:, 1:])
    with torch.no_grad():
        assert torch.equal(model(ids)[0], before)


# ------------------------------------------------------------------- render


def test_render_helpers_produce_output(model, ids):
    predictions = predict_tokens(model, _IdentityTokenizer(), ids[0])
    assert "surprisal:" in colour_by_surprisal(predictions, colour=False)

    with torch.no_grad():
        _, _, cache = model.run_with_cache(ids)
    tokens = [f"<{int(t)}>" for t in ids[0]]
    assert "│" in attention_heatmap(cache.attention(0)[0, 0], tokens, colour=False)

    rows = lens_trajectory(model, _IdentityTokenizer(), ids[0], position=4)
    assert "layer" in lens_table(rows, "<x>", colour=False)

    contributions, offset = attribute_logit(model, cache, 5, 9)
    assert "sum" in attribution_table(contributions, offset)
    assert "Δ loss" in ablation_table(ablate_heads(model, ids[:, :-1], ids[:, 1:]))


def test_colour_output_contains_escapes(model, ids):
    predictions = predict_tokens(model, _IdentityTokenizer(), ids[0])
    assert "\033[" in colour_by_surprisal(predictions, colour=True)
    assert "\033[" not in colour_by_surprisal(predictions, colour=False)


def test_visible_makes_whitespace_readable():
    assert visible("a b\nc\td") == "a·b⏎c→d"


def test_heatmap_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="query, key"):
        attention_heatmap(torch.zeros(2, 3, 4), ["a", "b"])


def test_render_works_with_a_real_tokenizer():
    """Nothing may assume the toy tokenizer's decode behaviour.

    Uses CharTokenizer rather than BPE: byte-level BPE has a 256-token floor, so
    it cannot produce a vocabulary small enough for a test-sized model.
    """
    corpus = ["the quick brown fox jumps over the lazy dog. " * 40]
    tok = CharTokenizer.train(corpus)
    model = build(vocab_size=tok.vocab_size)
    ids = torch.tensor(tok.encode("the quick brown fox")[:10])
    predictions = predict_tokens(model, tok, ids)
    rendered = colour_by_surprisal(predictions, colour=False)
    assert rendered and "·" in rendered  # spaces rendered visibly
