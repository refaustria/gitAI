"""Key/value cache for generation.

Without a cache, generating token *t* re-runs attention over the entire prefix,
so producing *T* tokens costs O(T^2) forward passes of growing width. The keys
and values for earlier positions do not change, though — only the query moves.
Caching them makes each new token O(1) work against a growing cache, which is
the difference between generating a corpus in minutes and in hours.

This is what makes the Phase 5 collapse experiment affordable: it needs hundreds
of thousands of model-generated tokens, and at uncached speed that is most of a
day per arm.
"""

from __future__ import annotations

import torch

__all__ = ["KVCache"]


class KVCache:
    """Per-layer key/value storage for one generation session.

    Allocated once up front rather than grown by concatenation: repeatedly
    concatenating reallocates the whole tensor on every token, which quietly
    costs more than the attention it was meant to save.

    Every layer must append before :meth:`advance` is called, since the write
    offset is shared across layers.
    """

    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_head: int,
        max_seq: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (batch, n_head, max_seq, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]
        self.max_seq = max_seq
        self.batch = batch
        self.length = 0

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this step's keys/values and return the full history so far."""
        b, _, t, _ = k.shape
        end = self.length + t
        if end > self.max_seq:
            raise ValueError(
                f"KV cache holds {self.max_seq} positions; this step would need {end}. "
                "Generation must window the context to the model's seq_len."
            )
        self.keys[layer][:b, :, self.length : end] = k
        self.values[layer][:b, :, self.length : end] = v
        return self.keys[layer][:b, :, :end], self.values[layer][:b, :, :end]

    def advance(self, steps: int) -> None:
        """Move the write offset. Call once per forward, after every layer."""
        self.length += steps

    def reset(self) -> None:
        self.length = 0
