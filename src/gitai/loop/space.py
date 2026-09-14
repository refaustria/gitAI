"""The search space: what the loop is permitted to propose.

Everything the loop may vary is declared here, and it may propose nothing else.
That is the concrete form of "the loop is a bounded search procedure" from
docs/self-improvement.md — the bound is a data structure, not a good intention.

**R8 is encoded as a constraint, not a preference.** The space refuses to admit
a synthetic fraction that would leave less real data than ``min_real_fraction``.
The `replace` regime that collapsed to 4.53 BPB — worse than a bigram lookup
table — is therefore *unreachable*, rather than reachable-but-hopefully-caught
by the promotion gate. A failure mode you can prove is out of reach beats one
you have to detect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["OutOfSpace", "Proposal", "SearchSpace"]


class OutOfSpace(Exception):
    """A proposal fell outside the declared search space."""


@dataclass(frozen=True)
class Proposal:
    """One candidate the loop will try. Immutable, and recorded verbatim."""

    iteration: int
    synthetic_fraction: float
    temperature: float
    top_k: int | None
    lr: float
    steps: int
    seed: int

    @property
    def real_fraction(self) -> float:
        return 1.0 - self.synthetic_fraction

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "real_fraction": self.real_fraction}

    def __str__(self) -> str:
        top_k = f", top-k {self.top_k}" if self.top_k else ""
        return (
            f"synthetic {self.synthetic_fraction:.0%}, T{self.temperature:g}{top_k}, "
            f"lr {self.lr:.1e}, {self.steps} steps"
        )


@dataclass(frozen=True)
class SearchSpace:
    """The declared, bounded set of things the loop may try."""

    synthetic_fraction: tuple[float, ...] = (0.0, 0.25, 0.5)
    temperature: tuple[float, ...] = (0.9, 1.0, 1.1)
    top_k: tuple[int | None, ...] = (None,)
    lr: tuple[float, ...] = (1e-3, 3e-3)
    steps: tuple[int, ...] = (300,)

    # R8: a lineage keeping at least this much real data was robust across every
    # sampling temperature tested; one keeping none collapsed catastrophically.
    min_real_fraction: float = 0.25

    _rejected: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_real_fraction <= 1.0:
            raise ValueError(f"min_real_fraction must be in [0, 1], got {self.min_real_fraction}")

        unreachable = [f for f in self.synthetic_fraction if f > 1.0 - self.min_real_fraction]
        if unreachable:
            raise OutOfSpace(
                f"synthetic fractions {unreachable} would leave less than "
                f"{self.min_real_fraction:.0%} real data. R8 measured a lineage with no real "
                f"data collapsing to 4.53 BPB — worse than a bigram baseline. The space must "
                f"not contain that regime."
            )
        for name, values in (
            ("synthetic_fraction", self.synthetic_fraction),
            ("temperature", self.temperature),
            ("lr", self.lr),
            ("steps", self.steps),
        ):
            if not values:
                raise ValueError(f"{name} must offer at least one option")
        if any(t <= 0 for t in self.temperature):
            raise ValueError("temperature must be positive")

    @property
    def size(self) -> int:
        """How many distinct candidates exist. A loop with a budget larger than
        this will necessarily repeat itself, which is worth knowing up front."""
        return (
            len(self.synthetic_fraction)
            * len(self.temperature)
            * len(self.top_k)
            * len(self.lr)
            * len(self.steps)
        )

    def sample(self, iteration: int, rng: np.random.Generator) -> Proposal:
        """Draw one candidate. Random search, deliberately.

        Random search over a small declared grid is hard to get subtly wrong and
        needs no tuning of its own. A smarter proposer (bandit, Bayesian) is a
        reasonable later change — but a buggy optimiser over the search space
        would be indistinguishable from a genuine research finding, which is a
        bad failure mode to invite this early.
        """
        return Proposal(
            iteration=iteration,
            synthetic_fraction=float(rng.choice(self.synthetic_fraction)),
            temperature=float(rng.choice(self.temperature)),
            top_k=self.top_k[int(rng.integers(len(self.top_k)))],
            lr=float(rng.choice(self.lr)),
            steps=int(rng.choice(self.steps)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    def validate(self, proposal: Proposal) -> None:
        """Raise unless every field came from this space.

        Checked on the way *in* to an iteration, not only on the way out of
        ``sample``. A proposal reaching the loop from anywhere else — a resumed
        run, a hand-written config, a future proposer — is held to the same
        bound.
        """
        problems = []
        if proposal.synthetic_fraction not in self.synthetic_fraction:
            problems.append(f"synthetic_fraction {proposal.synthetic_fraction}")
        if proposal.temperature not in self.temperature:
            problems.append(f"temperature {proposal.temperature}")
        if proposal.top_k not in self.top_k:
            problems.append(f"top_k {proposal.top_k}")
        if proposal.lr not in self.lr:
            problems.append(f"lr {proposal.lr}")
        if proposal.steps not in self.steps:
            problems.append(f"steps {proposal.steps}")
        if proposal.real_fraction < self.min_real_fraction:
            problems.append(
                f"real_fraction {proposal.real_fraction:.2f} below the floor "
                f"{self.min_real_fraction:.2f}"
            )
        if problems:
            raise OutOfSpace(f"proposal is outside the declared space: {', '.join(problems)}")

    def describe(self) -> str:
        lines = [f"search space: {self.size} distinct candidates"]
        for name in ("synthetic_fraction", "temperature", "top_k", "lr", "steps"):
            lines.append(f"  {name:<20} {getattr(self, name)}")
        lines.append(f"  {'min_real_fraction':<20} {self.min_real_fraction} (R8; hard floor)")
        return "\n".join(lines)
