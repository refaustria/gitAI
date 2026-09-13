"""Transformer correctness.

Two tests here carry most of the weight. ``test_causality`` catches an
attention-mask off-by-one, which is a bug that still trains to a plausible loss
curve because the model simply cheats — it can survive for weeks otherwise.
``test_overfits_a_single_batch`` catches nearly everything else in under a
second.
"""

from __future__ import annotations

import math

import pytest
import torch

from gitai.model import (
    RUNGS,
    ActivationCache,
    BigramModel,
    ModelConfig,
    RotaryEmbedding,
    Transformer,
    apply_rope,
)

VOCAB = 128


def tiny(**overrides) -> Transformer:
    config = ModelConfig(vocab_size=VOCAB, seq_len=32, d_model=64, n_layer=2, n_head=4, **overrides)
    model = Transformer(config)
    model.eval()
    return model


def rung_model(rung: str, **base) -> Transformer:
    """Build a rung, letting the rung's own choices win over the test defaults.

    Matters for v1_single_head, which pins n_head=1: a test default of 4 would
    silently make the single-head rung multi-head and the test a lie.
    """
    defaults = dict(vocab_size=VOCAB, seq_len=32, d_model=64, n_layer=2, n_head=4)
    return Transformer(ModelConfig(**{**defaults, **base, **RUNGS[rung]}))


@pytest.fixture
def model() -> Transformer:
    torch.manual_seed(0)
    return tiny()


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (3, 16))


# ------------------------------------------------------------------- causality


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_causality(rung):
    """Perturbing token *t* must not change any logit at a position before *t*.

    This is the single most valuable correctness test in the file. A mask that is
    off by one leaks the target into the input; the model happily uses it, the
    loss curve looks excellent, and every result is worthless.
    """
    torch.manual_seed(0)
    model = rung_model(rung)
    model.eval()
    x = torch.randint(0, VOCAB, (1, 16))

    with torch.no_grad():
        before, _ = model(x)
        perturbed = x.clone()
        position = 8
        perturbed[0, position] = (perturbed[0, position] + 1) % VOCAB
        after, _ = model(perturbed)

    assert torch.equal(before[0, :position], after[0, :position]), (
        "logits before the perturbed position changed: the model can see the future"
    )
    assert not torch.allclose(before[0, position:], after[0, position:]), (
        "logits from the perturbed position onward did not change: input is being ignored"
    )


def test_causality_holds_with_the_instrumented_path(model, batch):
    """The manual attention path has its own mask; it must be causal too."""
    with torch.no_grad():
        before, _, _ = model.run_with_cache(batch[:1, :16])
        perturbed = batch[:1, :16].clone()
        perturbed[0, 8] = (perturbed[0, 8] + 1) % VOCAB
        after, _, _ = model.run_with_cache(perturbed)
    assert torch.equal(before[0, :8], after[0, :8])


def test_attention_pattern_is_lower_triangular(model, batch):
    _, _, cache = model.run_with_cache(batch)
    pattern = cache.attention(0)
    seq = pattern.shape[-1]
    upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    assert pattern[..., upper].abs().max() == 0.0


def test_attention_rows_sum_to_one(model, batch):
    _, _, cache = model.run_with_cache(batch)
    assert torch.allclose(cache.attention(0).sum(-1), torch.ones(1), atol=1e-5)


# --------------------------------------------------- fast vs instrumented path


def test_fused_and_manual_attention_agree(model, batch):
    """The fast path never materialises the attention matrix, so it cannot be
    inspected; the manual path can. Running two implementations is only safe if
    a test proves they agree — same pattern as the dual cross-entropy in the
    autograd engine."""
    with torch.no_grad():
        fused, _ = model(batch)
        manual, _, _ = model.run_with_cache(batch)
    assert torch.allclose(fused, manual, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------ initialisation


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_initial_loss_is_log_vocab_size(rung):
    """Step-0 loss must be ln(vocab_size). If it is not, something is wrong
    before training even begins — and this catches it in a second rather than
    after an overnight run."""
    torch.manual_seed(0)
    model = rung_model(rung)
    model.eval()
    x = torch.randint(0, VOCAB, (8, 16))
    y = torch.randint(0, VOCAB, (8, 16))  # independent of x — see the test below
    with torch.no_grad():
        _, loss = model(x, y)
    assert abs(loss.item() - math.log(VOCAB)) < 0.15


def test_tied_embeddings_start_with_a_copy_prior():
    """A real signature of weight tying, worth pinning down.

    With tied embeddings the output logits are the residual stream dotted with
    the embedding matrix, and at initialisation the residual stream is mostly
    just the input token's own embedding. So the model starts out predicting
    *the current token* — loss on targets==inputs is well below ln(vocab), while
    loss on independent targets sits right at it.

    Not a bug, but if you evaluate on targets==inputs you will measure a
    suspiciously good untrained model and waste a day.
    """
    torch.manual_seed(0)
    tied = tiny(tie_embeddings=True)
    untied = tiny(tie_embeddings=False)
    x = torch.randint(0, VOCAB, (8, 16))
    y = torch.randint(0, VOCAB, (8, 16))

    with torch.no_grad():
        assert tied(x, x)[1].item() < math.log(VOCAB) - 0.5
        assert abs(tied(x, y)[1].item() - math.log(VOCAB)) < 0.2
        assert abs(untied(x, x)[1].item() - math.log(VOCAB)) < 0.2


# ------------------------------------------------------------ parameter counts


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_parameter_count_matches_the_hand_derived_formula(rung):
    """Catches a silent architecture change: if someone adds a bias or widens the
    MLP, the formula and the model disagree and this fails."""
    defaults = dict(vocab_size=512, seq_len=64, d_model=128, n_layer=3, n_head=4)
    config = ModelConfig(**{**defaults, **RUNGS[rung]})
    model = Transformer(config)
    counts = config.parameter_count()
    assert model.num_parameters() == counts["total"]
    assert model.num_parameters(non_embedding=True) == counts["non_embedding"]


def test_tying_saves_exactly_the_head_matrix():
    base = dict(vocab_size=512, seq_len=64, d_model=128, n_layer=2, n_head=4)
    tied = Transformer(ModelConfig(**base, tie_embeddings=True))
    untied = Transformer(ModelConfig(**base, tie_embeddings=False))
    assert untied.num_parameters() - tied.num_parameters() == 512 * 128


def test_tied_head_and_embedding_are_the_same_tensor():
    model = tiny(tie_embeddings=True)
    assert model.head.weight is model.token_embedding.weight


def test_swiglu_matches_gelu_parameter_count_within_rounding():
    """The 8/3 width correction. Without it SwiGLU is 1.5x larger and every
    comparison against a GELU MLP is rigged in its favour."""
    base = dict(vocab_size=512, seq_len=64, d_model=384, n_layer=4, n_head=6, bias=False)
    gelu = ModelConfig(**base, mlp="gelu").parameter_count()["non_embedding"]
    swiglu = ModelConfig(**base, mlp="swiglu").parameter_count()["non_embedding"]
    assert abs(swiglu - gelu) / gelu < 0.01


# -------------------------------------------------------------------- RoPE


def test_rope_scores_depend_only_on_relative_position():
    """The property that justifies RoPE existing. The same pair of vectors placed
    at positions (3, 1) and (10, 8) must produce an identical attention score,
    because both are two apart."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    cos, sin = rope(64)
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q.view(1, 1, 1, 16), cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k.view(1, 1, 1, 16), cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    assert score(3, 1) == pytest.approx(score(10, 8), abs=1e-5)
    assert score(20, 18) == pytest.approx(score(3, 1), abs=1e-5)
    assert score(5, 1) != pytest.approx(score(3, 1), abs=1e-3)  # different distance


def test_rope_preserves_vector_norm():
    """It is a rotation, so lengths must not change."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    cos, sin = rope(32)
    x = torch.randn(2, 4, 32, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_rejects_an_odd_head_dim():
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryEmbedding(head_dim=15, max_seq_len=32)


def test_rope_rejects_a_too_long_sequence():
    with pytest.raises(ValueError, match="exceeds RoPE cache"):
        RotaryEmbedding(head_dim=16, max_seq_len=8)(16)


# ------------------------------------------------------------------ learning


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_overfits_a_single_batch(rung):
    """The highest-value test in machine learning. A correct model must be able
    to memorise 8 short sequences of random tokens. If it cannot, there is a bug,
    and almost every bug shows up here first."""
    torch.manual_seed(0)
    model = rung_model(rung)
    model.train()
    x = torch.randint(0, VOCAB, (8, 16))
    y = torch.randint(0, VOCAB, (8, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"{rung} failed to overfit 8 sequences: loss {loss.item():.4f}"


def test_gradients_reach_every_parameter(model, batch):
    _, loss = model(batch, batch)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters received no gradient: {dead}"


def test_forward_is_deterministic(model, batch):
    with torch.no_grad():
        assert torch.equal(model(batch)[0], model(batch)[0])


def test_same_seed_builds_the_same_model():
    torch.manual_seed(7)
    a = tiny()
    torch.manual_seed(7)
    b = tiny()
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb)


# --------------------------------------------------------------------- cache


def test_cache_captures_every_intermediate(model, batch):
    _, _, cache = model.run_with_cache(batch)
    for i in range(model.config.n_layer):
        for name in (
            "resid_pre",
            "attn_pattern",
            "attn_z",
            "attn_out",
            "resid_mid",
            "mlp_out",
            "resid_post",
        ):
            assert f"blocks.{i}.{name}" in cache, f"missing blocks.{i}.{name}"
    assert "embeddings" in cache and "resid_final" in cache and "logits" in cache


def test_residual_stream_stacks_correctly(model, batch):
    _, _, cache = model.run_with_cache(batch)
    stream = cache.residual_stream()
    assert stream.shape == (model.config.n_layer + 1, *batch.shape, model.config.d_model)
    assert torch.equal(stream[0], cache["embeddings"])
    assert torch.equal(stream[-1], cache.resid_post(model.config.n_layer - 1))


def test_residual_stream_is_the_sum_of_its_contributions(model, batch):
    """The residual stream is additive: every block's output is added, never
    replaced. This is what makes per-component attribution meaningful at all."""
    _, _, cache = model.run_with_cache(batch)
    rebuilt = cache["embeddings"].clone()
    for i in range(model.config.n_layer):
        rebuilt = rebuilt + cache.layer(i, "attn_out") + cache.layer(i, "mlp_out")
    assert torch.allclose(rebuilt, cache.resid_post(model.config.n_layer - 1), atol=1e-5)


def test_cache_detaches_by_default(model, batch):
    _, _, cache = model.run_with_cache(batch)
    assert not cache["logits"].requires_grad
    _, _, with_grad = model.run_with_cache(batch, keep_grad=True)
    assert with_grad["logits"].requires_grad


def test_cache_reports_a_missing_key_helpfully(model, batch):
    _, _, cache = model.run_with_cache(batch)
    with pytest.raises(KeyError, match="was not cached"):
        cache["blocks.0.nonsense"]


def test_empty_cache_summary():
    assert "0 tensors" in ActivationCache().summary()


# ------------------------------------------------------------------ ablation


def test_head_mask_changes_the_output(model, batch):
    mask = torch.ones(model.config.n_layer, model.config.n_head)
    with torch.no_grad():
        baseline, _ = model(batch, head_mask=mask)
        mask[0, 0] = 0.0
        ablated, _ = model(batch, head_mask=mask)
    assert not torch.allclose(baseline, ablated)


def test_all_ones_head_mask_is_a_no_op(model, batch):
    mask = torch.ones(model.config.n_layer, model.config.n_head)
    with torch.no_grad():
        assert torch.allclose(
            model(batch, head_mask=mask)[0], model.run_with_cache(batch)[0], atol=1e-6
        )


# ---------------------------------------------------------------- generation


def test_generate_extends_the_sequence(model):
    out = model.generate(torch.randint(0, VOCAB, (2, 4)), max_new_tokens=6)
    assert out.shape == (2, 10)
    assert out.max() < VOCAB


def test_generate_is_reproducible_with_a_seeded_generator(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    a = model.generate(prompt, 8, generator=torch.Generator().manual_seed(0))
    b = model.generate(prompt, 8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_greedy_generation_is_deterministic(model):
    prompt = torch.randint(0, VOCAB, (1, 4))
    assert torch.equal(
        model.generate(prompt, 6, temperature=0.0), model.generate(prompt, 6, temperature=0.0)
    )


def test_generate_respects_the_context_window(model):
    """A prompt longer than seq_len must be windowed, not crash."""
    long_prompt = torch.randint(0, VOCAB, (1, model.config.seq_len + 5))
    assert model.generate(long_prompt, 3).shape[1] == model.config.seq_len + 8


def test_generate_restores_training_mode():
    m = tiny()
    m.train()
    m.generate(torch.randint(0, VOCAB, (1, 4)), 2)
    assert m.training


# -------------------------------------------------------------------- guards


def test_sequence_longer_than_the_model_is_rejected(model):
    with pytest.raises(ValueError, match="exceeds model maximum"):
        model(torch.randint(0, VOCAB, (1, model.config.seq_len + 1)))


def test_config_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(vocab_size=100, d_model=65, n_head=4)


def test_unknown_rung_is_rejected():
    with pytest.raises(KeyError, match="unknown rung"):
        Transformer.from_rung("v99_imaginary", VOCAB)


# --------------------------------------------------------------------- bigram


def test_bigram_starts_uniform_and_learns():
    """The floor every later model must beat. Its loss turns 'is 2.1 good?' from
    an opinion into a measurement."""
    torch.manual_seed(0)
    model = BigramModel(VOCAB)
    x = torch.randint(0, VOCAB, (4, 16))
    _, loss = model(x, x)
    assert abs(loss.item() - math.log(VOCAB)) < 1e-4  # exactly uniform at init

    opt = torch.optim.AdamW(model.parameters(), lr=0.5)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
    assert loss.item() < math.log(VOCAB) - 1.0


def test_bigram_has_no_non_embedding_parameters():
    assert BigramModel(VOCAB).num_parameters(non_embedding=True) == 0
    assert BigramModel(VOCAB).num_parameters() == VOCAB * VOCAB
