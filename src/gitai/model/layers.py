"""Normalisation, positional encoding, and feed-forward blocks.

Each component here is paired with the thing it replaces, because the ladder
(config.RUNGS) swaps them one at a time and the comparison is the point.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MLP", "RMSNorm", "RotaryEmbedding", "SwiGLU", "apply_rope", "build_mlp", "build_norm"]


class RMSNorm(nn.Module):
    """Root-mean-square normalisation.

    LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm
    skips the mean subtraction entirely and divides by the RMS. It turns out the
    re-centring was not doing useful work, so this is strictly cheaper — fewer
    reductions, one parameter vector instead of two — at no measured cost.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduction in float32 even when the rest runs in bf16: the sum of
        # squares is exactly where low precision bites.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_norm(kind: str, dim: int, bias: bool) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim, bias=bias)
    raise ValueError(f"unknown norm {kind!r}")


# --------------------------------------------------------------------- RoPE


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    Instead of *adding* a position vector to the token embedding, RoPE *rotates*
    the query and key vectors by an angle proportional to position. The
    consequence is the whole point: the dot product between a query at position
    *m* and a key at position *n* depends only on ``m - n``. Position enters the
    model as relative distance, for free, with zero parameters — where learned
    embeddings cost ``seq_len x d_model`` and cannot extrapolate one token past
    the length they were trained on.

    There is a test asserting exactly that relative-position property, because
    it is the reason this component is here.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inverse_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.arange(max_seq_len).float()
        angles = torch.outer(position, inverse_freq)  # (T, head_dim/2)
        emb = torch.cat((angles, angles), dim=-1)  # (T, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Angles for positions ``[offset, offset + seq_len)``.

        The offset is what lets incremental generation work: token 500 must be
        rotated by its *absolute* position even when it is the only token in the
        forward pass.
        """
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"positions {offset}..{offset + seq_len} exceed RoPE cache {self.max_seq_len}"
            )
        return self.cos[offset : offset + seq_len], self.sin[offset : offset + seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(batch, head, seq, head_dim)`` by position."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------- MLPs


class MLP(nn.Module):
    """The classic two-matrix feed-forward block with a GELU in the middle."""

    def __init__(self, d_model: int, hidden: int, bias: bool, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class SwiGLU(nn.Module):
    """Gated feed-forward: ``down(silu(gate(x)) * up(x))``.

    The extra matrix buys a multiplicative interaction — one projection decides
    *how much* of the other to let through — which reliably beats a plain GELU
    MLP at equal parameter count. Equal parameter count is doing real work in
    that sentence: three matrices at the usual 4x width would be 1.5x larger, so
    the width is set to 8/3 instead (see ``ModelConfig.hidden_dim``). Without
    that correction the comparison is rigged.
    """

    def __init__(self, d_model: int, hidden: int, bias: bool, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=bias)
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def build_mlp(kind: str, d_model: int, hidden: int, bias: bool, dropout: float) -> nn.Module:
    if kind == "swiglu":
        return SwiGLU(d_model, hidden, bias, dropout)
    if kind == "gelu":
        return MLP(d_model, hidden, bias, dropout)
    raise ValueError(f"unknown mlp {kind!r}")


def scaled_init_(module: nn.Module, std: float, n_layer: int) -> None:
    """GPT-2 style initialisation.

    Residual output projections are scaled by ``1/sqrt(2 * n_layer)``. Each layer
    adds its output into the residual stream, so without this the stream's
    variance grows with depth and a deep model either diverges or trains
    hopelessly slowly — a failure that looks exactly like a broken optimiser and
    sends people hunting in the wrong file for days.
    """
    for name, param in module.named_parameters():
        if name.endswith("down.weight") or name.endswith("proj.weight"):
            nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * n_layer))
        elif name.endswith("weight") and param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=std)
        elif name.endswith("bias"):
            nn.init.zeros_(param)
