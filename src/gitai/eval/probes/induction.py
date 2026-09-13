"""Induction: does the model use what it has already seen?

Show a model a random sequence twice: ``[R, R]``. A model with no in-context
ability predicts the second copy exactly as badly as the first. A model with
**induction heads** — a head that attends from the current token back to
whatever followed its previous occurrence — predicts the second copy far better.

The gap between the two, in bits, is the induction score. It is the cleanest
known measurement of in-context learning, and it works on any trained model
including one that only ever saw natural text. Decision 12's question D is
built on this.

The tokens are random, so nothing about the *content* can be memorised. Only the
mechanism helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gitai.interpret.surprisal import surprisal_bits

from .base import Probe

__all__ = ["InductionProbe"]


@dataclass
class InductionProbe(Probe):
    block_len: int = 32
    trials: int = 32
    seed: int = 0
    token_pool: np.ndarray | None = None
    name: str = "induction"

    @torch.no_grad()
    def run(self, model, **kwargs) -> dict[str, float]:
        vocab = model.config.vocab_size if hasattr(model, "config") else model.vocab_size
        max_len = getattr(getattr(model, "config", None), "seq_len", 2 * self.block_len)
        block = min(self.block_len, max_len // 2)

        rng = np.random.default_rng(self.seed)
        # Sampling from a pool of *frequent* tokens matters: uniform sampling
        # over the whole vocabulary draws mostly rare tokens whose embeddings
        # barely moved during training, which measures embedding coverage rather
        # than induction.
        pool = self.token_pool if self.token_pool is not None else np.arange(vocab)

        sequences = np.empty((self.trials, 2 * block), dtype=np.int64)
        for i in range(self.trials):
            repeated = rng.choice(pool, size=block, replace=True)
            sequences[i] = np.concatenate([repeated, repeated])

        ids = torch.from_numpy(sequences)
        was_training = model.training
        model.eval()
        try:
            logits, _ = model(ids[:, :-1])
        finally:
            model.train(was_training)

        bits = surprisal_bits(logits, ids[:, 1:])  # (trials, 2*block - 1)

        # Position i of `bits` predicts token i+1. Skip the first token of each
        # copy: predicting R[0] again at the seam needs no induction, just the
        # observation that a repeat has started.
        first_copy = bits[:, : block - 1]
        second_copy = bits[:, block:]

        per_position = second_copy.mean(0)
        return {
            "induction_first_copy_bits": float(first_copy.mean()),
            "induction_second_copy_bits": float(second_copy.mean()),
            "induction_score_bits": float(first_copy.mean() - second_copy.mean()),
            "induction_late_bits": float(per_position[len(per_position) // 2 :].mean()),
            "induction_uniform_baseline_bits": float(np.log2(len(pool))),
        }

    @torch.no_grad()
    def per_position(self, model) -> np.ndarray:
        """Surprisal at each position of the second copy.

        Induction should switch on almost immediately and then stay flat. A curve
        that decays slowly instead suggests the model is using general recency
        statistics rather than a copying circuit.
        """
        vocab = model.config.vocab_size if hasattr(model, "config") else model.vocab_size
        block = min(self.block_len, getattr(getattr(model, "config", None), "seq_len", 64) // 2)
        rng = np.random.default_rng(self.seed)
        pool = self.token_pool if self.token_pool is not None else np.arange(vocab)

        sequences = np.stack(
            [
                np.concatenate([r, r])
                for r in (rng.choice(pool, size=block, replace=True) for _ in range(self.trials))
            ]
        )
        ids = torch.from_numpy(sequences.astype(np.int64))
        model.eval()
        logits, _ = model(ids[:, :-1])
        bits = surprisal_bits(logits, ids[:, 1:])
        return bits[:, block - 1 :].mean(0).numpy()
