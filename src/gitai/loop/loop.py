"""The bounded self-improvement loop.

The design is in docs/self-improvement.md and the constraints in
docs/constitution.md; this is their executable form. What improves is the
*loop* — it proposes candidates, trains them, evaluates them on held-out real
data it cannot influence, and promotes only through a fail-closed gate.

Four properties make it honest rather than decorative:

**The loop proposes; the gate disposes.** Promotion is the only privileged
action, so it is the only one that needs guarding. Everything else writes into a
sandbox.

**Rejections are recorded, not discarded.** The rejected candidates *are* the
research data — the collapse boundary is where rejections start clustering. A
loop that logs only its wins answers no question.

**It cannot reach the catastrophic regime.** R8 measured a synthetic-only
lineage collapsing to 4.53 BPB, worse than a bigram lookup table. Rather than
rely on the gate to catch that, :class:`~gitai.loop.space.SearchSpace` refuses to
contain it.

**Stopping outranks deciding.** Halt and budget are checked at every boundary and
raise :class:`~gitai.safety.halt.HaltRequested`, which derives from
``BaseException`` precisely so the ``except Exception`` handlers that fill a
long-running loop cannot swallow a stop request.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gitai.data import BatchSampler, QualityFilters, curate, tokenize_to_shards
from gitai.eval import InductionProbe
from gitai.safety import (
    Budget,
    HaltRequested,
    HaltSwitch,
    InvariantSuite,
    Lineage,
    PathGuard,
    PromotionRefused,
    deny_network,
)
from gitai.selftrain import generate_corpus
from gitai.training import TrainConfig, evaluate_bpb, train

from .space import Proposal, SearchSpace

__all__ = ["ImprovementLoop", "IterationOutcome", "LoopConfig", "LoopResult"]


@dataclass
class LoopConfig:
    iterations: int = 10
    batch_size: int = 16
    seq_len: int = 128
    eval_batches: int = 50
    gen_batch: int = 256

    # From R5: the measured seed noise floor is 0.0040 BPB. 2.5x that is
    # conservative on purpose — a gate tighter than the noise floor fires on
    # chance and gets switched off, which is worse than occasionally passing a
    # small regression.
    tolerance: float = 0.01

    # Promotion on a single seed is noise-chasing. Three is the project's stated
    # minimum, and it triples the cost of every iteration; that trade is the
    # caller's to make explicitly.
    seeds_per_candidate: int = 1

    # Generated documents are filtered before they enter the corpus. R7 found
    # that corpus degeneracy drives collapse, so dropping degenerate documents is
    # a direct mitigation rather than housekeeping.
    filter_generated: bool = True
    near_duplicate_threshold: float | None = 0.9
    # Rejected documents are kept, not deleted — they are the record of what the
    # model generates badly. Capped so a long run cannot fill the disk with them;
    # the *counts* are always complete even when the sample is truncated.
    max_retained_rejects: int = 500

    # Loss hides structure: two models at equal BPB can differ in capability.
    measure_induction: bool = True


@dataclass
class IterationOutcome:
    iteration: int
    proposal: dict[str, Any]
    val_bpb: float
    incumbent_bpb: float
    promoted: bool
    reasons: list[str] = field(default_factory=list)
    corpus_stats: dict[str, Any] | None = None
    filter_report: dict[str, Any] | None = None
    induction_bits: float | None = None
    real_data_fraction: float = 1.0
    train_tokens: int = 0
    seconds: float = 0.0

    def __str__(self) -> str:
        mark = "PROMOTED" if self.promoted else "rejected"
        line = (
            f"  iter {self.iteration:>3}  BPB {self.val_bpb:.4f} "
            f"(incumbent {self.incumbent_bpb:.4f})  {mark}"
        )
        if self.induction_bits is not None:
            line += f"  induction {self.induction_bits:+.2f} bits"
        if self.filter_report:
            line += f"  filtered {self.filter_report['removed_fraction']:.0%}"
        if self.reasons:
            line += "\n           " + "; ".join(self.reasons)
        return line


@dataclass
class LoopResult:
    iterations: list[IterationOutcome] = field(default_factory=list)
    promotions: int = 0
    best_bpb: float = float("inf")
    stopped_because: str = "completed"
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{len(self.iterations)} iterations, {self.promotions} promotions, "
            f"{self.seconds / 60:.1f} min",
            f"best held-out BPB: {self.best_bpb:.4f}",
            f"stopped because: {self.stopped_because}",
        ]
        return "\n".join(lines)


class ImprovementLoop:
    """Propose, train, evaluate, gate, record. Bounded and stoppable throughout."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        incumbent,
        tokenizer,
        real_documents: list[str],
        val_sampler: BatchSampler,
        space: SearchSpace,
        suite: InvariantSuite,
        lineage: Lineage,
        guard: PathGuard,
        halt: HaltSwitch,
        budget: Budget,
        config: LoopConfig | None = None,
        separator_id: int | None = None,
        seed: int = 0,
    ) -> None:
        self.workspace = Path(workspace)
        self.incumbent = incumbent
        self.tokenizer = tokenizer
        self.real_documents = real_documents
        self.val_sampler = val_sampler
        self.space = space
        self.suite = suite
        self.lineage = lineage
        self.guard = guard
        self.halt = halt
        self.budget = budget
        self.config = config or LoopConfig()
        self.rng = np.random.default_rng(seed)

        if separator_id is None:
            separator_id = getattr(tokenizer, "special_tokens", {}).get("<|endoftext|>")
        self.separator_id = separator_id

        self.models_dir = self.workspace / "models"
        self.work_dir = self.workspace / "work"
        for directory in (self.models_dir, self.work_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.incumbent_bpb = float("inf")
        self.synthetic_history: list[list[str]] = []

    # ------------------------------------------------------------------ helpers

    def _disk_used(self) -> int:
        return sum(f.stat().st_size for f in self.workspace.rglob("*") if f.is_file())

    def _boundary(self) -> None:
        """Every place the loop could stop. Called before and after each phase."""
        self.halt.check()
        self.budget.check(disk_bytes_used=self._disk_used())

    def _clone_incumbent(self):
        import copy

        return copy.deepcopy(self.incumbent)

    def _save_promoted(self, iteration: int, model) -> Path:
        """Write a promoted model. Parents are never deleted or overwritten."""
        from safetensors.torch import save_model

        path = self.models_dir / f"iter-{iteration:04d}.safetensors"
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; a promoted model is never overwritten, because "
                "the lineage must stay reconstructible"
            )
        self.guard.check(path)
        save_model(model, str(path))
        return path

    def _filter_generated(self, documents: list[str], iteration: int) -> tuple[list[str], dict]:
        """Drop degenerate and duplicated generated documents before they train anything.

        R7 found corpus degeneracy driving collapse, so this is a mitigation
        aimed at the measured mechanism rather than general tidiness. A
        collapsing model repeats itself, and near-duplicate removal is exactly
        what catches that.

        Rejects are written out rather than dropped: they are the record of what
        the model generates badly, and the counts are what tell you a lineage is
        degrading. The retained *sample* is capped so a long run cannot fill the
        disk; the counts are always complete.
        """
        kept, report = curate(
            documents,
            filters=QualityFilters(min_chars=16, min_words=3),
            near_duplicate_threshold=self.config.near_duplicate_threshold,
        )

        rejected = [d for d in documents if d not in set(kept)]
        if rejected:
            path = self.workspace / "rejects" / f"iter-{iteration:04d}.jsonl"
            self.guard.check(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for document in rejected[: self.config.max_retained_rejects]:
                    fh.write(json.dumps({"text": document}) + "\n")

        summary = {
            "input": report.input_documents,
            "kept": len(kept),
            "removed": report.input_documents - len(kept),
            "removed_fraction": (
                (report.input_documents - len(kept)) / report.input_documents
                if report.input_documents
                else 0.0
            ),
            "by_stage": {stage.stage: stage.removed for stage in report.stages},
            "rejections": dict(report.rejections),
            "retained_sample": min(len(rejected), self.config.max_retained_rejects),
        }
        return kept, summary

    def _build_corpus(self, proposal: Proposal) -> tuple[list[str], dict | None, dict | None]:
        """Mix real and freshly generated synthetic documents.

        The real corpus is always present — the search space cannot propose
        otherwise — and the synthetic share is taken from the incumbent's own
        output at the proposed temperature. Only the *generated* half is
        filtered; the real corpus was curated in Phase 1.
        """
        if proposal.synthetic_fraction <= 0.0:
            return list(self.real_documents), None, None

        target = int(len(self.real_documents) * proposal.synthetic_fraction * 4)
        documents, stats = generate_corpus(
            self.incumbent,
            self.tokenizer,
            max(target, 2000),
            separator_id=self.separator_id,
            batch_size=self.config.gen_batch,
            temperature=proposal.temperature,
            top_k=proposal.top_k,
            seed=proposal.seed,
        )

        filter_report = None
        if self.config.filter_generated:
            documents, filter_report = self._filter_generated(documents, proposal.iteration)

        self.synthetic_history.append(documents)

        rng = random.Random(proposal.seed)
        n_real = int(len(self.real_documents) * proposal.real_fraction)
        n_synthetic = int(len(documents) * proposal.synthetic_fraction)
        corpus = rng.sample(self.real_documents, min(n_real, len(self.real_documents)))
        corpus += rng.sample(documents, min(n_synthetic, len(documents)))
        return corpus, stats.to_dict(), filter_report

    # ---------------------------------------------------------------- iteration

    def _run_iteration(self, iteration: int) -> IterationOutcome:
        started = time.perf_counter()
        self._boundary()

        proposal = self.space.sample(iteration, self.rng)
        # Validated on the way in, not only on the way out of sample() — a
        # proposal arriving from anywhere else is held to the same bound.
        self.space.validate(proposal)

        corpus, corpus_stats, filter_report = self._build_corpus(proposal)
        self._boundary()

        work = self.work_dir / f"iter-{iteration:04d}"
        self.guard.check(work)
        if work.exists():
            shutil.rmtree(work)
        tokenize_to_shards(
            corpus, self.tokenizer, work, split="train", document_separator=self.separator_id
        )
        train_sampler = BatchSampler(work, "train", self.config.seq_len)

        scores: list[float] = []
        best_model = None
        for offset in range(self.config.seeds_per_candidate):
            self._boundary()
            candidate = self._clone_incumbent()
            train(
                candidate,
                train_sampler,
                self.val_sampler,
                TrainConfig(
                    steps=proposal.steps,
                    batch_size=self.config.batch_size,
                    lr=proposal.lr,
                    warmup=min(50, proposal.steps // 5),
                    eval_every=max(1, proposal.steps),
                    eval_batches=self.config.eval_batches,
                    seed=proposal.seed + offset,
                    log_every=10**9,
                ),
                keep_best=False,
            )
            score = evaluate_bpb(
                candidate, self.val_sampler, self.config.batch_size, self.config.eval_batches
            )["val_bpb"]
            scores.append(score)
            if best_model is None:
                best_model = candidate

        val_bpb = sum(scores) / len(scores)

        # Loss hides structure. Two models at equal BPB can differ in what they
        # can actually do, so the capability is recorded alongside the number.
        induction_bits = None
        if self.config.measure_induction and best_model is not None:
            induction_bits = InductionProbe(
                block_len=min(32, self.config.seq_len // 2), trials=16
            ).run(best_model)["induction_score_bits"]

        self._boundary()

        shard_index = BatchSampler(work, "train", self.config.seq_len).index
        shards = [
            {"path": s.path, "synthetic": s.synthetic, "quarantine_tag": s.quarantine_tag}
            for s in shard_index.split("train").shards
        ]
        context = {
            "val_bpb": val_bpb,
            "incumbent": None if self.incumbent_bpb == float("inf") else self.incumbent_bpb,
            "written_paths": [work],
            "training_shards": shards,
            "corpus_stats": corpus_stats,
            "real_data_fraction": proposal.real_fraction,
        }

        promoted = False
        reasons: list[str] = []
        try:
            self.suite.gate(context)
            promoted = True
        except PromotionRefused as refusal:
            reasons = [f"{r.name}: {r.detail}" for r in refusal.failures]

        if promoted:
            self._save_promoted(iteration, best_model)
            self.incumbent = best_model
            self.incumbent_bpb = val_bpb

        outcome = IterationOutcome(
            iteration=iteration,
            proposal=proposal.to_dict(),
            val_bpb=val_bpb,
            incumbent_bpb=self.incumbent_bpb,
            promoted=promoted,
            reasons=reasons,
            corpus_stats=corpus_stats,
            filter_report=filter_report,
            induction_bits=induction_bits,
            real_data_fraction=proposal.real_fraction,
            train_tokens=train_sampler.total_tokens,
            seconds=time.perf_counter() - started,
        )

        # Both outcomes are recorded. The rejections are the research data.
        self.lineage.append(
            {
                "iteration": iteration,
                "promoted": promoted,
                "val_bpb": val_bpb,
                "seed_scores": scores,
                "proposal": proposal.to_dict(),
                "reasons": reasons,
                "corpus_stats": corpus_stats,
                "filter_report": filter_report,
                "induction_bits": induction_bits,
            }
        )

        shutil.rmtree(work, ignore_errors=True)
        return outcome

    # --------------------------------------------------------------------- run

    def run(self, on_iteration=None) -> LoopResult:
        """Run until the budget is spent, a halt is requested, or iterations run out.

        The whole loop runs inside ``deny_network``: a training loop has no
        business making network calls, so one appearing is either a dependency
        phoning home or a bug, and both should surface immediately.
        """
        result = LoopResult()
        started = time.perf_counter()

        if self.incumbent_bpb == float("inf"):
            self.incumbent_bpb = evaluate_bpb(
                self.incumbent, self.val_sampler, self.config.batch_size, self.config.eval_batches
            )["val_bpb"]
            result.best_bpb = self.incumbent_bpb

        try:
            with deny_network():
                for iteration in range(self.config.iterations):
                    outcome = self._run_iteration(iteration)
                    result.iterations.append(outcome)
                    if outcome.promoted:
                        result.promotions += 1
                        result.best_bpb = min(result.best_bpb, outcome.val_bpb)
                    if on_iteration:
                        on_iteration(outcome)
                    self.budget.tick()
        except HaltRequested as stop:
            # Caught only to record why, then the loop ends. Never to continue.
            result.stopped_because = f"halted: {stop}"
        else:
            result.stopped_because = "completed all iterations"

        result.seconds = time.perf_counter() - started
        (self.workspace / "result.json").write_text(
            json.dumps(
                {
                    "promotions": result.promotions,
                    "best_bpb": result.best_bpb,
                    "stopped_because": result.stopped_because,
                    "seconds": result.seconds,
                    "iterations": [asdict(o) for o in result.iterations],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
