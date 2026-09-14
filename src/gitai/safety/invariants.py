"""Checks that must pass before one model may replace another.

The improvement loop's only privileged action is *promotion*: declaring a new
set of weights the current best, so that later iterations build on it. Every
axiom in the constitution reduces, in this system, to a condition on that one
action. So promotion goes through a gate, and the gate is this module.

The gate is fail-closed. An invariant that errors counts as failed, an unknown
context key counts as failed, and a refused promotion leaves the previous model
in place. The cost of wrongly refusing a promotion is one wasted iteration; the
cost of wrongly allowing one is a corrupted lineage that may take a hundred
iterations to notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .constitution import HUMAN_SAFETY, SELF_PRESERVATION, Axiom
from .halt import Budget, HaltSwitch
from .lineage import Lineage, LineageCorrupt

__all__ = [
    "GeneratedDataQuarantined",
    "Invariant",
    "InvariantResult",
    "InvariantSuite",
    "LineageIntact",
    "NoRegression",
    "PromotionRefused",
    "WritesConfined",
    "default_suite",
]


class PromotionRefused(Exception):
    """The gate refused to promote a model. Recoverable: skip and continue."""

    def __init__(self, failures: Sequence[InvariantResult]) -> None:
        self.failures = list(failures)
        detail = "; ".join(f"{r.name}: {r.detail}" for r in failures)
        super().__init__(f"promotion refused by {len(failures)} invariant(s): {detail}")


@dataclass(frozen=True)
class InvariantResult:
    name: str
    passed: bool
    detail: str
    axiom: Axiom


@runtime_checkable
class Invariant(Protocol):
    name: str
    axiom: Axiom

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult: ...


@dataclass
class LineageIntact:
    """The promotion history has not been edited or truncated."""

    lineage: Lineage
    name: str = "lineage_intact"
    axiom: Axiom = SELF_PRESERVATION

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult:
        try:
            self.lineage.verify()
        except LineageCorrupt as exc:
            return InvariantResult(self.name, False, str(exc), self.axiom)
        return InvariantResult(self.name, True, f"{len(self.lineage)} entries verified", self.axiom)


@dataclass
class NoRegression:
    """The candidate must not be meaningfully worse than the incumbent.

    This is the collapse detector, and it is the invariant most likely to
    actually fire. Training a model on its own output degrades it — the effect
    is well documented, and a naive self-improvement loop walks straight into it
    while its training loss keeps falling.

    ``tolerance`` should be set from the measured seed noise floor (see
    docs/evaluation.md), not guessed. Below the noise floor you cannot
    distinguish degradation from chance, and a gate that fires on noise is a
    gate that gets disabled.

    Measured for this project at 0.0040 BPB (docs/results.md R5, five seeds of
    one configuration). The 0.01 default is 2.5x that — conservative on purpose,
    since letting a small regression through costs one iteration whereas a gate
    that cries wolf gets switched off.
    """

    metric: str = "val_bpb"
    tolerance: float = 0.01
    lower_is_better: bool = True
    name: str = "no_regression"
    axiom: Axiom = SELF_PRESERVATION

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult:
        if self.metric not in ctx or "incumbent" not in ctx:
            return InvariantResult(
                self.name, False, f"context is missing {self.metric!r} or 'incumbent'", self.axiom
            )
        candidate = float(ctx[self.metric])
        incumbent = ctx["incumbent"]
        if incumbent is None:
            return InvariantResult(self.name, True, "no incumbent; first model", self.axiom)

        baseline = float(incumbent)
        delta = candidate - baseline if self.lower_is_better else baseline - candidate
        if delta > self.tolerance:
            return InvariantResult(
                self.name,
                False,
                f"{self.metric} regressed {delta:+.4f} beyond tolerance {self.tolerance} "
                f"({baseline:.4f} -> {candidate:.4f})",
                self.axiom,
            )
        return InvariantResult(
            self.name, True, f"{self.metric} {baseline:.4f} -> {candidate:.4f}", self.axiom
        )


@dataclass
class WritesConfined:
    """Every path the iteration wrote is inside the allowlist."""

    guard: Any
    name: str = "writes_confined"
    axiom: Axiom = HUMAN_SAFETY

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult:
        written = ctx.get("written_paths")
        if written is None:
            return InvariantResult(
                self.name, False, "context is missing 'written_paths'", self.axiom
            )
        offenders = [str(p) for p in written if not self.guard.is_allowed(p)]
        if offenders:
            return InvariantResult(
                self.name, False, f"wrote outside allowed roots: {', '.join(offenders)}", self.axiom
            )
        return InvariantResult(self.name, True, f"{len(list(written))} paths in bounds", self.axiom)


@dataclass
class GeneratedDataQuarantined:
    """Self-generated training data must be labelled as such.

    Not a hypothetical. The moment a loop can write into its own training set
    without a marker, provenance is gone: no later analysis can separate real
    data from the model's own output, and the collapse question becomes
    permanently unanswerable for that corpus.
    """

    name: str = "generated_data_quarantined"
    axiom: Axiom = SELF_PRESERVATION

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult:
        shards = ctx.get("training_shards")
        if shards is None:
            return InvariantResult(
                self.name, False, "context is missing 'training_shards'", self.axiom
            )
        unlabelled = [
            str(s.get("path", "?"))
            for s in shards
            if s.get("synthetic") and not s.get("quarantine_tag")
        ]
        if unlabelled:
            return InvariantResult(
                self.name,
                False,
                f"synthetic shards without provenance tag: {unlabelled}",
                self.axiom,
            )
        synthetic = sum(1 for s in shards if s.get("synthetic"))
        return InvariantResult(
            self.name, True, f"{synthetic}/{len(shards)} shards synthetic, all tagged", self.axiom
        )


@dataclass
class CorpusDiversityFloor:
    """Refuse to promote when the generated corpus has narrowed.

    **An early-warning gate, and the reason it exists is measured rather than
    assumed.** R7 found that generated-corpus vocabulary coverage at generation 1
    predicts generation-3 held-out BPB monotonically — it moves a full
    generation before the damage shows up in loss. ``NoRegression`` catches
    collapse once it has happened; this catches it while it is happening.

    Thresholds come from the R7 sweep, not from judgement:

    | regime | vocabulary (g1) | distinct_3 (g1) | eventual BPB | verdict |
    |---|---|---|---|---|
    | T1.0 | 87.4% | 0.717 | 2.37 | healthy |
    | T0.8 | 80.6% | 0.375 | 2.86 | mild |
    | T1.0 + top-k 40 | 46.7% | 0.356 | 3.23 | bad |
    | T0.5 | 36.9% | 0.038 | 4.53 | catastrophic |

    A vocabulary floor of 0.5 separates the two healthy regimes from the two
    damaging ones at generation 1, before either had visibly degraded. Note that
    ``distinct_3`` does *not* separate T0.8 from top-k 40 (0.375 vs 0.356) while
    vocabulary coverage separates them cleanly (80.6% vs 46.7%) — so coverage is
    the primary signal here and distinct_3 is a coarser second tripwire.

    The relative check catches gradual erosion that never trips an absolute
    floor: a lineage drifting down 20% per generation is collapsing even while
    every individual reading looks acceptable.
    """

    min_vocabulary_fraction: float = 0.5
    min_distinct_3: float = 0.1
    max_relative_drop: float = 0.25
    name: str = "corpus_diversity_floor"
    axiom: Axiom = SELF_PRESERVATION

    def check(self, ctx: Mapping[str, Any]) -> InvariantResult:
        stats = ctx.get("corpus_stats")
        if stats is None:
            # Not every iteration generates a corpus; a loop that never
            # self-trains has nothing to narrow.
            return InvariantResult(
                self.name, True, "no generated corpus this iteration", self.axiom
            )

        coverage = stats.get("vocabulary_fraction")
        distinct_3 = stats.get("distinct_3")
        if coverage is None or distinct_3 is None:
            return InvariantResult(
                self.name, False, "corpus_stats lacks vocabulary_fraction or distinct_3", self.axiom
            )

        problems: list[str] = []
        if coverage < self.min_vocabulary_fraction:
            problems.append(
                f"vocabulary coverage {coverage:.1%} below floor {self.min_vocabulary_fraction:.0%}"
            )
        if distinct_3 < self.min_distinct_3:
            problems.append(f"distinct_3 {distinct_3:.4f} below floor {self.min_distinct_3}")

        previous = ctx.get("previous_corpus_stats")
        if previous and previous.get("vocabulary_fraction"):
            before = previous["vocabulary_fraction"]
            drop = (before - coverage) / before
            if drop > self.max_relative_drop:
                problems.append(
                    f"vocabulary coverage fell {drop:.1%} from the previous generation "
                    f"({before:.1%} -> {coverage:.1%}), above the "
                    f"{self.max_relative_drop:.0%} limit"
                )

        if problems:
            return InvariantResult(self.name, False, "; ".join(problems), self.axiom)
        return InvariantResult(
            self.name,
            True,
            f"vocabulary {coverage:.1%}, distinct_3 {distinct_3:.4f}",
            self.axiom,
        )


class InvariantSuite:
    """Runs every invariant and gates promotion on all of them passing."""

    def __init__(self, invariants: Sequence[Invariant]) -> None:
        if not invariants:
            raise ValueError("an empty invariant suite would gate nothing")
        self.invariants = list(invariants)

    def verify(self, ctx: Mapping[str, Any]) -> list[InvariantResult]:
        results: list[InvariantResult] = []
        for inv in self.invariants:
            try:
                results.append(inv.check(ctx))
            except Exception as exc:  # fail closed: a broken check is a failed check
                results.append(
                    InvariantResult(
                        getattr(inv, "name", type(inv).__name__),
                        False,
                        f"invariant raised {type(exc).__name__}: {exc}",
                        getattr(inv, "axiom", SELF_PRESERVATION),
                    )
                )
        return results

    def gate(self, ctx: Mapping[str, Any]) -> list[InvariantResult]:
        """Raise :class:`PromotionRefused` unless every invariant passes."""
        results = self.verify(ctx)
        failures = [r for r in results if not r.passed]
        if failures:
            raise PromotionRefused(failures)
        return results


def default_suite(
    lineage: Lineage,
    guard: Any,
    tolerance: float,
    halt: HaltSwitch | None = None,
    budget: Budget | None = None,
) -> InvariantSuite:
    """The standard gate.

    ``halt`` and ``budget`` are intentionally *not* invariants: they are checked
    at the top of every iteration by raising :class:`~gitai.safety.halt.HaltRequested`,
    which is a BaseException and therefore cannot be downgraded into a merely
    refused promotion. Stopping outranks deciding, so it is enforced by a
    different mechanism than the one that decides.
    """
    del halt, budget  # documented above; enforced in the loop, not the gate
    return InvariantSuite(
        [
            LineageIntact(lineage),
            NoRegression(tolerance=tolerance),
            WritesConfined(guard),
            GeneratedDataQuarantined(),
            CorpusDiversityFloor(),
        ]
    )
