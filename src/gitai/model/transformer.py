"""The decoder-only transformer.

One model, configured by :class:`~gitai.model.config.ModelConfig`, covering
every rung of the ladder from GPT-2 to the modern Llama-style stack. Pre-norm
throughout: normalise *before* the sublayer and add its output to an unbroken
residual stream, so gradients reach layer 0 without passing through a
normalisation on the way. Post-norm needs careful warmup to train at all;
pre-norm mostly just works.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cache import ActivationCache
from .config import RUNGS, ModelConfig
from .kvcache import KVCache
from .layers import RotaryEmbedding, build_mlp, build_norm, scaled_init_

__all__ = ["Block", "Transformer"]


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from .attention import CausalSelfAttention

        self.norm1 = build_norm(config.norm, config.d_model, config.bias)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_norm(config.norm, config.d_model, config.bias)
        self.mlp = build_mlp(
            config.mlp, config.d_model, config.hidden_dim, config.bias, config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        cache: ActivationCache | None = None,
        layer: int | None = None,
        head_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        if cache is not None and layer is not None:
            cache.put_layer(layer, "resid_pre", x)

        attn_out = self.attn(self.norm1(x), cos, sin, cache, layer, head_mask, kv_cache)
        if cache is not None and layer is not None:
            cache.put_layer(layer, "attn_out", attn_out)
        x = x + attn_out

        if cache is not None and layer is not None:
            cache.put_layer(layer, "resid_mid", x)

        mlp_out = self.mlp(self.norm2(x))
        if cache is not None and layer is not None:
            cache.put_layer(layer, "mlp_out", mlp_out)
        x = x + mlp_out

        if cache is not None and layer is not None:
            cache.put_layer(layer, "resid_post", x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.seq_len, config.d_model) if config.position == "learned" else None
        )
        self.rope = (
            RotaryEmbedding(config.head_dim, config.seq_len, config.rope_base)
            if config.position == "rope"
            else None
        )

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = build_norm(config.norm, config.d_model, config.bias)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        scaled_init_(self, config.init_std, config.n_layer)

        if config.tie_embeddings:
            # Input and output embeddings learn the same thing — a map between
            # tokens and directions in the residual stream — so sharing them
            # halves the embedding cost. At this scale that is not a micro-
            # optimisation: see Decision 4 on the embedding parameter budget.
            self.head.weight = self.token_embedding.weight

    # ---------------------------------------------------------------- helpers

    @classmethod
    def from_rung(cls, rung: str, vocab_size: int, **overrides) -> Transformer:
        """Build a named rung of the ladder (see ``config.RUNGS``)."""
        if rung not in RUNGS:
            raise KeyError(f"unknown rung {rung!r}; available: {sorted(RUNGS)}")
        return cls(ModelConfig(vocab_size=vocab_size, **{**RUNGS[rung], **overrides}))

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            if self.position_embedding is not None:
                total -= self.position_embedding.weight.numel()
        return total

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: ActivationCache | None = None,
        head_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns ``(logits, loss)``. ``loss`` is None when no targets are given.

        Passing a ``cache`` records every intermediate; passing ``head_mask``
        (shape ``(n_layer, n_head)``) zeroes individual attention heads, which is
        how ablation experiments run without touching the weights; passing a
        ``kv_cache`` makes this an incremental generation step, where ``idx``
        holds only the *new* tokens and positions continue from the cache.
        """
        _, seq = idx.shape
        offset = kv_cache.length if kv_cache is not None else 0
        if offset + seq > self.config.seq_len:
            raise ValueError(
                f"positions {offset}..{offset + seq} exceed model maximum {self.config.seq_len}"
            )

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(offset, offset + seq, device=idx.device)
            x = x + self.position_embedding(positions)
        x = self.drop(x)

        cos = sin = None
        if self.rope is not None:
            cos, sin = self.rope(seq, offset=offset)
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        if cache is not None:
            cache.put("embeddings", x)

        for i, block in enumerate(self.blocks):
            layer_mask = head_mask[i] if head_mask is not None else None
            x = block(x, cos, sin, cache, i, layer_mask, kv_cache)

        if kv_cache is not None:
            # After every layer has written, never before — the write offset is
            # shared across layers.
            kv_cache.advance(seq)

        x = self.norm_f(x)
        if cache is not None:
            cache.put("resid_final", x)

        logits = self.head(x)
        if cache is not None:
            cache.put("logits", logits)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
            )
        return logits, loss

    def run_with_cache(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, keep_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, ActivationCache]:
        """Forward pass that also hands back every intermediate."""
        cache = ActivationCache(keep_grad=keep_grad)
        logits, loss = self.forward(idx, targets, cache=cache)
        return logits, loss, cache

    # ------------------------------------------------------------- generation

    def _sample_next(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sample continuations.

        With ``use_cache`` (the default) each new token costs one forward pass
        over a single position instead of re-running attention across the whole
        prefix. ``use_cache=False`` keeps the naive path, which exists so a test
        can assert the two agree — see ``test_cached_and_uncached_generation_agree``.

        ``generator`` makes sampling reproducible, which matters more than it
        sounds: comparing two checkpoints on differently-seeded samples tells
        you nothing.
        """
        was_training = self.training
        self.eval()
        try:
            if not use_cache:
                for _ in range(max_new_tokens):
                    logits, _ = self(idx[:, -self.config.seq_len :])
                    next_token = self._sample_next(logits[:, -1, :], temperature, top_k, generator)
                    idx = torch.cat((idx, next_token), dim=1)
                return idx

            config = self.config
            kv_cache = KVCache(
                n_layer=config.n_layer,
                batch=idx.shape[0],
                n_head=config.n_head,
                max_seq=config.seq_len,
                head_dim=config.head_dim,
                dtype=self.token_embedding.weight.dtype,
                device=idx.device,
            )
            logits, _ = self(idx[:, -config.seq_len :], kv_cache=kv_cache)

            for _ in range(max_new_tokens):
                next_token = self._sample_next(logits[:, -1, :], temperature, top_k, generator)
                idx = torch.cat((idx, next_token), dim=1)

                if kv_cache.length >= config.seq_len:
                    # Context window full: re-prefill from the last seq_len
                    # tokens. Costs one full pass per window rather than per
                    # token, so generation stays linear overall.
                    kv_cache.reset()
                    logits, _ = self(idx[:, -config.seq_len :], kv_cache=kv_cache)
                else:
                    logits, _ = self(next_token, kv_cache=kv_cache)
            return idx
        finally:
            self.train(was_training)
