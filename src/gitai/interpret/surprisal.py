"""Per-token prediction: what did the model expect, and how surprised was it?

The most immediately legible view of a language model. Surprisal is measured in
bits — ``-log2 p(actual token)`` — so a value of 1 means the model gave the
right answer a 50% chance, 0 means certainty, and 10 means it was blindsided.

Mean surprisal is exactly the cross-entropy loss in a different unit, and
dividing by bytes-per-token gives bits-per-byte, the project's primary metric
([evaluation.md](../../../docs/evaluation.md)). These are three views of one
number, which is why surprisal is the right thing to look at first: it is the
loss, decomposed to where it was actually incurred.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["TokenPrediction", "bits_per_byte", "predict_tokens", "surprisal_bits"]

LOG2 = 0.6931471805599453  # ln(2)


@dataclass
class TokenPrediction:
    """What the model predicted at one position, and what actually came next."""

    position: int
    context_token: str
    actual_id: int
    actual_token: str
    surprisal: float  # bits; -log2 p(actual)
    entropy: float  # bits; how spread out the whole distribution was
    rank: int  # where the actual token sat in the ranking (1 = top)
    top: list[tuple[str, float]]

    @property
    def confident_and_wrong(self) -> bool:
        """Low entropy but high surprisal — the model was sure, and sure wrong.

        The most interesting failure mode to read: it means the model has learned
        a rule that does not hold here, rather than simply not knowing.
        """
        return self.entropy < 2.0 and self.surprisal > 4.0


def surprisal_bits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``-log2 p(target)`` for each position. Shape ``(batch, seq)``."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) / LOG2


def bits_per_byte(total_surprisal_bits: float, total_utf8_bytes: int) -> float:
    """The primary metric. Tokenizer-independent, unlike perplexity.

    Perplexity is per *token*, so it cannot be compared across vocabularies —
    and vocabulary size is an explicit variable in this project, which would make
    perplexity incomparable between your own experiments. See Decision 10.
    """
    if total_utf8_bytes <= 0:
        raise ValueError("byte count must be positive")
    return total_surprisal_bits / total_utf8_bytes


@torch.no_grad()
def predict_tokens(model, tokenizer, ids: torch.Tensor, top_k: int = 5) -> list[TokenPrediction]:
    """Walk a single sequence and report the prediction made at every position.

    ``ids`` is a 1-D tensor of token ids. Position *i* reports the model's
    prediction for ``ids[i + 1]`` given ``ids[:i + 1]``.
    """
    if ids.dim() != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {tuple(ids.shape)}")
    if ids.numel() < 2:
        raise ValueError("need at least two tokens to have something to predict")

    was_training = model.training
    model.eval()
    try:
        logits, _ = model(ids.unsqueeze(0))
    finally:
        model.train(was_training)

    logits = logits[0].float()
    probs = logits.softmax(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(-1) / LOG2

    out: list[TokenPrediction] = []
    for i in range(ids.numel() - 1):
        actual = int(ids[i + 1])
        distribution = probs[i]
        top_probs, top_ids = distribution.topk(min(top_k, distribution.numel()))
        # Rank of the true token: how many tokens the model rated above it.
        rank = int((distribution > distribution[actual]).sum()) + 1

        out.append(
            TokenPrediction(
                position=i,
                context_token=tokenizer.decode([int(ids[i])]),
                actual_id=actual,
                actual_token=tokenizer.decode([actual]),
                surprisal=float(-log_probs[i, actual] / LOG2),
                entropy=float(entropies[i]),
                rank=rank,
                top=[
                    (tokenizer.decode([int(t)]), float(p))
                    for p, t in zip(top_probs, top_ids, strict=True)
                ],
            )
        )
    return out
