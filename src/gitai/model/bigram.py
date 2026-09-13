"""Rung v0: the bigram model.

A lookup table. Token *t* predicts token *t+1* with no context whatsoever, no
attention, and no hidden layer — ``vocab x vocab`` parameters and nothing else.

Its value is entirely as a floor. Its loss is the best any context-free model
can do on the corpus, so it converts "is 2.1 a good loss?" from a matter of
opinion into a measurement. A transformer that fails to beat it decisively is
broken, and knowing that on day one is worth the twenty lines.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["BigramModel"]


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.table.weight)  # start at a uniform distribution

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        return 0 if non_embedding else self.table.weight.numel()

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])
            probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
