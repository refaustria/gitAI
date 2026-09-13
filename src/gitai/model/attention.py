"""Causal self-attention, with two forward paths that must agree.

The fast path uses PyTorch's fused ``scaled_dot_product_attention``, which never
materialises the attention matrix — good for speed, useless for looking inside.
The instrumented path computes attention by hand so the weights can be captured.

Running two implementations of one operation is a deliberate pattern, used here
for the same reason as the dual cross-entropy in the autograd engine: the fast
one is what you run, the slow one is what proves it right. A test asserts they
agree to floating-point tolerance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cache import ActivationCache
from .config import ModelConfig
from .layers import apply_rope

__all__ = ["CausalSelfAttention"]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One matrix for q, k and v: a single larger matmul beats three smaller
        # ones on every backend.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.ones(config.seq_len, config.seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cache: ActivationCache | None = None,
        layer: int | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        # (batch, seq, d_model) -> (batch, head, seq, head_dim)
        q = q.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)

        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if cache is None and head_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            # The mask is applied to a (query, key) matrix and keeps key <= query.
            # An off-by-one here still trains to a plausible loss curve, because
            # the model simply cheats; test_causality is what catches it.
            scores = scores.masked_fill(~self.causal_mask[:seq, :seq], float("-inf"))
            pattern = scores.softmax(dim=-1)

            if cache is not None and layer is not None:
                cache.put_layer(layer, "attn_pattern", pattern)

            if head_mask is not None:
                # Ablation: zero a head's contribution while leaving the rest of
                # the forward pass untouched.
                pattern = pattern * head_mask.view(1, -1, 1, 1)

            attended = F.dropout(pattern, self.dropout, self.training)
            out = attended @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        if cache is not None and layer is not None:
            cache.put_layer(layer, "attn_z", out)
        return self.resid_dropout(self.proj(out))
