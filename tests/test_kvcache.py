"""KV-cache generation.

The whole test suite here exists to support one claim: caching changes the cost
of generation, not its result. Everything else is bookkeeping.
"""

from __future__ import annotations

import pytest
import torch

from gitai.model import RUNGS, KVCache, ModelConfig, Transformer

VOCAB = 96


def build(rung: str = "v6_modern", **overrides) -> Transformer:
    torch.manual_seed(0)
    defaults = dict(vocab_size=VOCAB, seq_len=32, d_model=64, n_layer=2, n_head=4)
    model = Transformer(ModelConfig(**{**defaults, **overrides, **RUNGS[rung]}))
    model.eval()
    return model


# ---------------------------------------------------------------- equivalence


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_cached_and_uncached_generation_agree(rung):
    """The headline guarantee, on every rung — including the two that use learned
    positional embeddings rather than RoPE, since both need a position offset."""
    model = build(rung)
    prompt = torch.randint(0, VOCAB, (4, 5))
    cached = model.generate(prompt, 20, temperature=0.0, use_cache=True)
    uncached = model.generate(prompt, 20, temperature=0.0, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cached_and_uncached_logits_agree():
    """Not just the argmax — the distributions themselves."""
    model = build()
    ids = torch.randint(0, VOCAB, (3, 12))
    with torch.no_grad():
        plain, _ = model(ids)
        cache = KVCache(2, 3, 4, 32, 16)
        incremental, _ = model(ids, kv_cache=cache)
    assert torch.allclose(plain, incremental, atol=1e-5)


def test_incremental_matches_full_forward_token_by_token():
    """Feed one token at a time through the cache; the final logits must match a
    single full forward over the whole sequence."""
    model = build()
    ids = torch.randint(0, VOCAB, (2, 10))
    with torch.no_grad():
        expected, _ = model(ids)
        cache = KVCache(2, 2, 4, 32, 16)
        for position in range(ids.shape[1]):
            logits, _ = model(ids[:, position : position + 1], kv_cache=cache)
    assert torch.allclose(logits[:, -1], expected[:, -1], atol=1e-5)


def test_agreement_holds_past_the_context_window():
    """Generating beyond seq_len forces the re-prefill path. Both routes must
    still agree, because both condition on the last seq_len tokens."""
    model = build(seq_len=16)
    prompt = torch.randint(0, VOCAB, (2, 4))
    cached = model.generate(prompt, 30, temperature=0.0, use_cache=True)
    uncached = model.generate(prompt, 30, temperature=0.0, use_cache=False)
    assert cached.shape[1] == 34
    assert torch.equal(cached, uncached)


def test_sampling_agrees_given_the_same_generator():
    model = build()
    prompt = torch.randint(0, VOCAB, (2, 4))
    a = model.generate(prompt, 12, temperature=0.9, generator=torch.Generator().manual_seed(1))
    b = model.generate(
        prompt, 12, temperature=0.9, generator=torch.Generator().manual_seed(1), use_cache=False
    )
    assert torch.equal(a, b)


# ------------------------------------------------------------------ mechanics


def test_cache_grows_by_the_number_of_tokens_fed():
    model = build()
    cache = KVCache(2, 1, 4, 32, 16)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 5)), kv_cache=cache)
        assert cache.length == 5
        model(torch.randint(0, VOCAB, (1, 1)), kv_cache=cache)
        assert cache.length == 6


def test_reset_returns_the_cache_to_empty():
    cache = KVCache(1, 1, 2, 8, 4)
    cache.append(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    cache.advance(3)
    cache.reset()
    assert cache.length == 0


def test_overflow_is_an_explicit_error():
    """Silently truncating would corrupt generation in a way nothing downstream
    could detect."""
    cache = KVCache(1, 1, 2, 4, 4)
    with pytest.raises(ValueError, match="holds 4 positions"):
        cache.append(0, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))


def test_forward_rejects_positions_beyond_the_model_maximum():
    model = build(seq_len=8)
    cache = KVCache(2, 1, 4, 8, 16)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 8)), kv_cache=cache)
        with pytest.raises(ValueError, match="exceed model maximum"):
            model(torch.randint(0, VOCAB, (1, 1)), kv_cache=cache)


def test_append_returns_the_whole_history_not_just_the_new_part():
    cache = KVCache(1, 1, 2, 8, 4)
    cache.append(0, torch.ones(1, 2, 2, 4), torch.ones(1, 2, 2, 4))
    cache.advance(2)
    k, _ = cache.append(0, torch.full((1, 2, 1, 4), 9.0), torch.zeros(1, 2, 1, 4))
    assert k.shape[2] == 3
    assert k[0, 0, 0, 0] == 1.0 and k[0, 0, 2, 0] == 9.0


# ----------------------------------------------------------------------- RoPE


def test_rope_offset_selects_absolute_positions():
    """Token 500 must be rotated by its absolute position even when it is the
    only token in the forward pass. Without this, cached generation would encode
    every new token as if it were at position 0."""
    from gitai.model import RotaryEmbedding

    rope = RotaryEmbedding(head_dim=8, max_seq_len=64)
    cos_all, sin_all = rope(64)
    cos_at, sin_at = rope(1, offset=17)
    assert torch.equal(cos_at[0], cos_all[17])
    assert torch.equal(sin_at[0], sin_all[17])


def test_rope_offset_beyond_the_cache_is_rejected():
    from gitai.model import RotaryEmbedding

    with pytest.raises(ValueError, match="exceed RoPE cache"):
        RotaryEmbedding(head_dim=8, max_seq_len=16)(4, offset=14)


# ------------------------------------------------------------------- batching


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_generation_is_independent_across_the_batch(batch):
    """Batched generation must give each row the same result it would get alone —
    otherwise corpus generation silently mixes rows together."""
    model = build()
    torch.manual_seed(2)
    prompts = torch.randint(0, VOCAB, (batch, 6))
    together = model.generate(prompts, 10, temperature=0.0)
    for row in range(min(batch, 4)):
        alone = model.generate(prompts[row : row + 1], 10, temperature=0.0)
        assert torch.equal(together[row : row + 1], alone)
