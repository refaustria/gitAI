"""Evaluation metrics, and the result record they produce."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

__all__ = ["EVAL_SUITE_VERSION", "EvalResult", "accuracy_at_k", "perplexity"]

# Bump on ANY change to what the suite measures or how. Results carrying
# different versions are not comparable, and stamping it is what makes that
# detectable six months later instead of invisible.
EVAL_SUITE_VERSION = "1.0.0"


def perplexity(loss_nats: float) -> float:
    """``exp(loss)``. Tokenizer-DEPENDENT — never compare across vocabularies.

    Included because it is conventional and readers expect it, not because it is
    the metric to trust. See Decision 10: bits-per-byte is the headline number
    precisely because vocabulary size is a variable in this project.
    """
    return math.exp(min(loss_nats, 700.0))  # guard against overflow on an untrained model


def accuracy_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Fraction of positions where the target is in the top-k predictions."""
    top = logits.topk(min(k, logits.size(-1)), dim=-1).indices
    return float((top == targets.unsqueeze(-1)).any(-1).float().mean())


@dataclass
class EvalResult:
    """One evaluation of one checkpoint. Written to disk and into the runs index.

    ``suite_version`` travels with every result. Changing the eval invalidates
    comparison with earlier runs, and without the stamp that invalidation is
    silent.
    """

    suite_version: str = EVAL_SUITE_VERSION
    split: str = "val"

    # Primary. Tokenizer-independent, comparable across every configuration.
    bits_per_byte: float = 0.0

    # Secondary. Tokenizer-dependent; comparable only within one tokenizer.
    loss: float = 0.0
    perplexity: float = 0.0
    bits_per_token: float = 0.0
    accuracy_top1: float = 0.0
    accuracy_top5: float = 0.0

    tokens: int = 0
    utf8_bytes: int = 0
    seconds: float = 0.0

    probes: dict[str, float] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"eval suite {self.suite_version} on {self.split!r} "
            f"({self.tokens:,} tokens, {self.seconds:.1f}s)",
            f"  bits/byte      {self.bits_per_byte:.4f}   <- primary, tokenizer-independent",
            f"  loss           {self.loss:.4f}",
            f"  perplexity     {self.perplexity:.2f}   (tokenizer-dependent)",
            f"  bits/token     {self.bits_per_token:.4f}   (tokenizer-dependent)",
            f"  top-1 accuracy {self.accuracy_top1:.4f}",
            f"  top-5 accuracy {self.accuracy_top5:.4f}",
        ]
        if self.probes:
            lines.append("  probes:")
            lines += [f"    {name:<26} {value:.4f}" for name, value in sorted(self.probes.items())]
        return "\n".join(lines)
