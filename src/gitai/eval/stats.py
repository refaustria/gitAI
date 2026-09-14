"""Seed aggregation and significance.

The most important module in the evaluation package, and the one most often
skipped. Small models are noisy: train the same configuration twice with
different seeds and the final loss differs. That spread is the **noise floor**,
and any effect smaller than it does not exist no matter how clean the plot
looks.

Everything here is dependency-free on purpose. The significance test is an exact
permutation test rather than a t-test: with 5 seeds per arm there are only 252
ways to split 10 numbers into two groups, so the exact null distribution can be
enumerated. No normality assumption, no approximation, and it is honest about
how little power 5 seeds actually buys.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

__all__ = ["Comparison", "SeedGroup", "compare_groups", "permutation_test"]


@dataclass(frozen=True)
class SeedGroup:
    """Summary of one configuration measured across several seeds."""

    label: str
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values) / self.n

    @property
    def std(self) -> float:
        """Sample standard deviation (n-1). Zero for a single seed."""
        if self.n < 2:
            return 0.0
        mean = self.mean
        return math.sqrt(sum((v - mean) ** 2 for v in self.values) / (self.n - 1))

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n) if self.n else 0.0

    @property
    def spread(self) -> float:
        """max - min. The blunt, honest version of the noise floor."""
        return max(self.values) - min(self.values) if self.values else 0.0

    def __str__(self) -> str:
        if self.n == 1:
            return f"{self.mean:.4f} (1 seed — no spread, treat as unmeasured)"
        return f"{self.mean:.4f} ± {self.std:.4f} (n={self.n}, spread {self.spread:.4f})"


def permutation_test(a: Sequence[float], b: Sequence[float], max_exact: int = 20000) -> float:
    """Two-sided p-value for a difference in means, by exact permutation.

    Pools the observations, enumerates every way of splitting them back into two
    groups of the original sizes, and counts how often the difference is at
    least as extreme as the observed one. If that is more splits than
    ``max_exact``, falls back to sampling.

    Returns 1.0 for degenerate input (either group empty) rather than raising:
    a comparison with nothing to compare is maximally unsurprising.
    """
    if not a or not b:
        return 1.0

    pooled = list(a) + list(b)
    n_a = len(a)
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    total_sum = sum(pooled)

    def difference(indices: tuple[int, ...]) -> float:
        left = sum(pooled[i] for i in indices)
        return abs(left / n_a - (total_sum - left) / (len(pooled) - n_a))

    splits = math.comb(len(pooled), n_a)
    if splits <= max_exact:
        extreme = sum(
            1
            for combo in combinations(range(len(pooled)), n_a)
            if difference(combo) >= observed - 1e-12
        )
        return extreme / splits

    import random

    rng = random.Random(0)
    indices = list(range(len(pooled)))
    extreme = 0
    for _ in range(max_exact):
        rng.shuffle(indices)
        if difference(tuple(indices[:n_a])) >= observed - 1e-12:
            extreme += 1
    return extreme / max_exact


@dataclass(frozen=True)
class Comparison:
    a: SeedGroup
    b: SeedGroup
    difference: float
    p_value: float
    noise_floor: float
    lower_is_better: bool = True

    @property
    def effect_in_noise_units(self) -> float:
        """The difference measured in noise floors. Below 1.0, forget it."""
        return abs(self.difference) / self.noise_floor if self.noise_floor else float("inf")

    @property
    def min_achievable_p(self) -> float:
        """The smallest p-value this sample size can produce.

        A permutation test enumerates every way of splitting the pooled
        observations, so with n_a and n_b seeds there are C(n_a+n_b, n_a) splits
        and the two most extreme ones give p = 2/C. With three seeds per arm that
        floor is 2/20 = 0.10 — **the test cannot reach p < 0.05 at all**.

        Without this, a 75x-noise effect gets reported as "not significant",
        which is not a null result but an arithmetic property of the sample size.
        """
        if self.a.n < 1 or self.b.n < 1:
            return 1.0
        return min(1.0, 2 / math.comb(self.a.n + self.b.n, self.a.n))

    @property
    def underpowered(self) -> bool:
        """True when no possible outcome could reach p < 0.05."""
        return self.min_achievable_p > 0.05

    @property
    def winner(self) -> str:
        # difference = b.mean - a.mean, so difference > 0 means b scored HIGHER.
        # When lower is better, a higher score means a wins.
        b_scored_higher = self.difference > 0
        return self.a.label if b_scored_higher == self.lower_is_better else self.b.label

    @property
    def verdict(self) -> str:
        if self.effect_in_noise_units < 1.0:
            return "below the noise floor — no effect"
        if self.underpowered:
            return (
                f"{self.winner} better by {self.effect_in_noise_units:.1f}x noise, but "
                f"UNDERPOWERED: n={self.a.n}v{self.b.n} cannot reach p<0.05 "
                f"(floor p={self.min_achievable_p:.2f})"
            )
        if self.p_value > 0.05:
            return f"not significant (p={self.p_value:.3f})"
        return (
            f"{self.winner} better (p={self.p_value:.3f}, {self.effect_in_noise_units:.1f}x noise)"
        )

    def __str__(self) -> str:
        return (
            f"{self.a.label}: {self.a}\n"
            f"{self.b.label}: {self.b}\n"
            f"  difference {self.difference:+.4f}   {self.verdict}"
        )


def compare_groups(
    a: SeedGroup, b: SeedGroup, noise_floor: float | None = None, lower_is_better: bool = True
) -> Comparison:
    """Compare two configurations honestly.

    ``noise_floor`` should come from a measured same-config multi-seed run, not
    from these two groups — using the groups' own spread to judge themselves is
    circular when each has only a handful of seeds.
    """
    floor = noise_floor if noise_floor is not None else max(a.std, b.std)
    return Comparison(
        a=a,
        b=b,
        difference=b.mean - a.mean,
        p_value=permutation_test(a.values, b.values),
        noise_floor=floor,
        lower_is_better=lower_is_better,
    )
