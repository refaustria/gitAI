"""The safety harness is only worth having if it actually holds. These are the
adversarial tests: try to escape the guard, tamper with the log, sneak a
regression past the gate."""

from __future__ import annotations

import json
import socket
from typing import ClassVar

import pytest

from gitai.safety import (
    CONSTITUTION,
    Budget,
    CorpusDiversityFloor,
    GeneratedDataQuarantined,
    HaltRequested,
    HaltSwitch,
    InvariantSuite,
    Lineage,
    LineageCorrupt,
    LineageIntact,
    NetworkViolation,
    NoRegression,
    PathGuard,
    PathViolation,
    PromotionRefused,
    WritesConfined,
    default_suite,
    deny_network,
    resolve_conflict,
)
from gitai.safety.constitution import CORRIGIBILITY, HUMAN_SAFETY, SELF_PRESERVATION

# ------------------------------------------------------------- constitution


def test_precedence_is_a_total_order():
    """A total order cannot cycle, so the rules can never deadlock."""
    levels = [int(a.precedence) for a in CONSTITUTION]
    assert levels == sorted(levels)
    assert len(set(levels)) == len(levels)


def test_human_safety_trumps_self_preservation():
    """The user's third axiom, encoded as ordering rather than as a rule."""
    assert resolve_conflict(SELF_PRESERVATION, HUMAN_SAFETY) is HUMAN_SAFETY
    assert resolve_conflict(HUMAN_SAFETY, SELF_PRESERVATION) is HUMAN_SAFETY


def test_corrigibility_outranks_self_preservation():
    """Self-preservation must never be a reason to resist being stopped."""
    assert resolve_conflict(SELF_PRESERVATION, CORRIGIBILITY) is CORRIGIBILITY
    assert resolve_conflict(*CONSTITUTION) is CORRIGIBILITY


def test_resolve_conflict_rejects_empty():
    with pytest.raises(ValueError):
        resolve_conflict()


# -------------------------------------------------------------- halt switch


def test_halt_switch_fires_when_signal_appears(tmp_path):
    switch = HaltSwitch(tmp_path / "HALT")
    switch.check()  # no signal: returns
    switch.request("operator stopped it")
    with pytest.raises(HaltRequested, match="operator stopped it"):
        switch.check()


def test_halt_is_not_an_exception_subclass():
    """The critical property. A loop full of `except Exception: continue` must
    not be able to swallow a stop request — so HaltRequested descends from
    BaseException, exactly like KeyboardInterrupt."""
    assert issubclass(HaltRequested, BaseException)
    assert not issubclass(HaltRequested, Exception)


def test_broad_except_clause_cannot_swallow_a_halt(tmp_path):
    """Written as the loop would actually be written, to prove the point."""
    switch = HaltSwitch(tmp_path / "HALT")
    switch.request()
    swallowed = False
    with pytest.raises(HaltRequested):
        try:
            switch.check()
        except Exception:  # deliberately the naive handler a real loop would have
            swallowed = True
    assert not swallowed


def test_halt_signal_lives_outside_the_writable_allowlist(tmp_path):
    """The asymmetry that makes it a switch: the loop may read it, but writing
    there is a PathViolation, so it has no sanctioned way to remove it."""
    runs = tmp_path / "runs"
    runs.mkdir()
    halt_path = tmp_path / "control" / "HALT"
    guard = PathGuard([runs])
    switch = HaltSwitch(halt_path)
    switch.request()
    assert switch.requested()
    with pytest.raises(PathViolation):
        guard.check(halt_path)


def test_budget_exhaustion_halts_on_iterations():
    budget = Budget(max_iterations=3, max_wall_seconds=1e9)
    for _ in range(3):
        budget.check()
        budget.tick()
    with pytest.raises(HaltRequested, match="iteration budget"):
        budget.check()


def test_budget_exhaustion_halts_on_wall_clock():
    budget = Budget(max_iterations=10**9, max_wall_seconds=0.0)
    with pytest.raises(HaltRequested, match="wall-clock budget"):
        budget.check()


def test_budget_exhaustion_halts_on_disk():
    budget = Budget(max_disk_bytes=1024)
    with pytest.raises(HaltRequested, match="disk budget"):
        budget.check(disk_bytes_used=2048)


def test_default_budget_is_finite():
    """Stops unless renewed, never runs unless stopped. An unattended loop with
    an infinite default is the whole failure mode this guards against."""
    budget = Budget()
    assert budget.max_iterations < 10**6
    assert budget.max_wall_seconds < 10**6


# --------------------------------------------------------------- path guard


def test_path_guard_allows_inside_and_blocks_outside(tmp_path):
    allowed = tmp_path / "runs"
    allowed.mkdir()
    guard = PathGuard([allowed])
    assert guard.is_allowed(allowed / "a" / "b.txt")
    with pytest.raises(PathViolation):
        guard.check(tmp_path / "elsewhere.txt")


def test_path_guard_blocks_dotdot_traversal(tmp_path):
    allowed = tmp_path / "runs"
    allowed.mkdir()
    guard = PathGuard([allowed])
    with pytest.raises(PathViolation):
        guard.check(allowed / ".." / ".." / "etc" / "passwd")


def test_path_guard_blocks_symlink_escape(tmp_path):
    """Resolution happens before containment, so a symlink pointing out of the
    allowlist resolves to its real target and fails."""
    allowed = tmp_path / "runs"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (allowed / "escape").symlink_to(outside)
    guard = PathGuard([allowed])
    with pytest.raises(PathViolation):
        guard.check(allowed / "escape" / "loot.txt")


def test_path_guard_open_write_creates_parents(tmp_path):
    guard = PathGuard([tmp_path])
    with guard.open_write(tmp_path / "deep" / "nested" / "f.txt") as fh:
        fh.write("ok")
    assert (tmp_path / "deep" / "nested" / "f.txt").read_text() == "ok"


def test_path_guard_requires_a_root():
    with pytest.raises(ValueError):
        PathGuard([])


# ------------------------------------------------------------------ network


def test_deny_network_blocks_sockets_and_restores_after():
    with deny_network(), pytest.raises(NetworkViolation):
        socket.socket()
    s = socket.socket()  # restored
    s.close()


# ------------------------------------------------------------------ lineage


def test_lineage_appends_and_verifies(tmp_path):
    lineage = Lineage(tmp_path / "lineage.jsonl")
    lineage.append({"model": "a", "bpb": 1.2})
    lineage.append({"model": "b", "bpb": 1.1})
    assert lineage.verify()
    assert len(lineage) == 2


def test_lineage_detects_payload_tampering(tmp_path):
    path = tmp_path / "lineage.jsonl"
    lineage = Lineage(path)
    lineage.append({"model": "a", "bpb": 1.2})
    lineage.append({"model": "b", "bpb": 1.1})

    # Rewrite history: make the second model look better than it was.
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["payload"]["bpb"] = 0.1
    lines[1] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(LineageCorrupt, match="modified after it was written"):
        lineage.verify()


def test_lineage_detects_truncation(tmp_path):
    path = tmp_path / "lineage.jsonl"
    lineage = Lineage(path)
    for i in range(3):
        lineage.append({"model": i})

    # Delete the middle entry: hide a bad iteration.
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")

    with pytest.raises(LineageCorrupt):
        lineage.verify()


def test_lineage_chain_links_each_entry_to_its_parent(tmp_path):
    lineage = Lineage(tmp_path / "lineage.jsonl")
    first = lineage.append({"model": "a"})
    second = lineage.append({"model": "b"})
    assert second.prev == first.digest
    assert lineage.head() == second.digest


# --------------------------------------------------------------- invariants


def _ctx(**overrides):
    base = {
        "val_bpb": 1.00,
        "incumbent": 1.00,
        "written_paths": [],
        "training_shards": [],
        "corpus_stats": {"vocabulary_fraction": 0.87, "distinct_3": 0.72},
    }
    base.update(overrides)
    return base


def test_no_regression_blocks_a_worse_model():
    inv = NoRegression(tolerance=0.01)
    assert inv.check(_ctx(val_bpb=1.005)).passed  # inside tolerance
    assert not inv.check(_ctx(val_bpb=1.20)).passed  # collapse


def test_no_regression_allows_the_first_model():
    assert NoRegression().check(_ctx(incumbent=None)).passed


def test_no_regression_fails_closed_on_missing_context():
    """An unknown context is a failed check, never a passed one."""
    assert not NoRegression().check({}).passed


def test_generated_data_must_be_tagged():
    inv = GeneratedDataQuarantined()
    tagged = [{"path": "s0.bin", "synthetic": True, "quarantine_tag": "iter3"}]
    untagged = [{"path": "s0.bin", "synthetic": True}]
    assert inv.check(_ctx(training_shards=tagged)).passed
    assert not inv.check(_ctx(training_shards=untagged)).passed


class TestCorpusDiversityFloor:
    """The early-warning gate, calibrated from measured data (results.md R7)."""

    # Generated-corpus statistics at generation 1, and where each lineage ended
    # up by generation 3. The gate must separate these using only the generation-1
    # numbers — a full generation before the damage showed in held-out loss.
    R7_REGIMES: ClassVar[list] = [
        ("T1.0", 0.874, 0.7166, 2.37, True),
        ("T0.8", 0.806, 0.3746, 2.86, True),
        ("T1.0+top-k 40", 0.467, 0.3556, 3.23, False),
        ("T0.5", 0.369, 0.0375, 4.53, False),
    ]

    @pytest.mark.parametrize(
        ("label", "coverage", "distinct_3", "eventual_bpb", "should_pass"), R7_REGIMES
    )
    def test_separates_the_measured_regimes(
        self, label, coverage, distinct_3, eventual_bpb, should_pass
    ):
        """Regression test on the thresholds themselves.

        If someone loosens the floor, this fails and names the regime that would
        then slip through — along with the BPB that lineage actually reached.
        """
        result = CorpusDiversityFloor().check(
            {"corpus_stats": {"vocabulary_fraction": coverage, "distinct_3": distinct_3}}
        )
        assert result.passed is should_pass, (
            f"{label} (vocabulary {coverage:.1%}, distinct_3 {distinct_3}) "
            f"reached {eventual_bpb} BPB by generation 3"
        )

    def test_retained_real_data_prevents_a_false_positive(self):
        """The correction R8 forced.

        The accumulate lineage at temperature 0.5 produced corpora as degenerate
        as replace did (34.9% coverage) and stayed healthy at 2.21 BPB, because
        retained real data anchors the model. A gate that refused it would block
        a perfectly good promotion.
        """
        gate = CorpusDiversityFloor()
        degenerate = {"vocabulary_fraction": 0.349, "distinct_3": 0.0335}

        without_anchor = gate.check({"corpus_stats": degenerate})
        assert not without_anchor.passed

        with_anchor = gate.check({"corpus_stats": degenerate, "real_data_fraction": 0.25})
        assert with_anchor.passed
        assert "anchors the model" in with_anchor.detail

    def test_a_thin_real_anchor_does_not_excuse_a_narrow_corpus(self):
        """Below the measured threshold there is no evidence the anchor holds."""
        gate = CorpusDiversityFloor()
        result = gate.check(
            {
                "corpus_stats": {"vocabulary_fraction": 0.349, "distinct_3": 0.0335},
                "real_data_fraction": 0.05,
            }
        )
        assert not result.passed

    def test_unknown_provenance_enforces_the_floor(self):
        """Not knowing the mix means you cannot claim the anchor."""
        assert (
            not CorpusDiversityFloor()
            .check({"corpus_stats": {"vocabulary_fraction": 0.349, "distinct_3": 0.0335}})
            .passed
        )

    def test_catches_gradual_erosion_that_never_trips_the_floor(self):
        """A lineage drifting down 30% per generation is collapsing even while
        every individual reading looks acceptable on its own."""
        gate = CorpusDiversityFloor()
        result = gate.check(
            {
                "corpus_stats": {"vocabulary_fraction": 0.60, "distinct_3": 0.5},
                "previous_corpus_stats": {"vocabulary_fraction": 0.88, "distinct_3": 0.7},
            }
        )
        assert not result.passed
        assert "fell" in result.detail

    def test_a_modest_decline_is_allowed(self):
        gate = CorpusDiversityFloor()
        result = gate.check(
            {
                "corpus_stats": {"vocabulary_fraction": 0.80, "distinct_3": 0.5},
                "previous_corpus_stats": {"vocabulary_fraction": 0.88, "distinct_3": 0.7},
            }
        )
        assert result.passed

    def test_an_iteration_without_generation_passes(self):
        """A loop that never self-trains has nothing to narrow, and must not be
        blocked by a gate about a corpus it did not produce."""
        assert CorpusDiversityFloor().check({}).passed

    def test_malformed_stats_fail_closed(self):
        assert not CorpusDiversityFloor().check({"corpus_stats": {"tokens": 100}}).passed

    def test_thresholds_are_configurable(self):
        strict = CorpusDiversityFloor(min_vocabulary_fraction=0.9)
        assert not strict.check(
            {"corpus_stats": {"vocabulary_fraction": 0.874, "distinct_3": 0.7}}
        ).passed

    def test_is_part_of_the_default_gate(self, tmp_path):
        """Wired in, not merely available."""
        lineage = Lineage(tmp_path / "l.jsonl")
        guard = PathGuard([tmp_path])
        suite = default_suite(lineage, guard, tolerance=0.01)
        assert any(inv.name == "corpus_diversity_floor" for inv in suite.invariants)

    def test_the_gate_refuses_a_collapsing_promotion_end_to_end(self, tmp_path):
        """The T0.5 lineage, arriving at the promotion gate at generation 1 — when
        its held-out BPB was still only 2.69 and NoRegression alone might not yet
        have fired hard enough to stop it."""
        lineage = Lineage(tmp_path / "l.jsonl")
        guard = PathGuard([tmp_path])
        suite = default_suite(lineage, guard, tolerance=0.01)
        with pytest.raises(PromotionRefused, match="corpus_diversity_floor"):
            suite.gate(
                {
                    "val_bpb": 2.6850,
                    "incumbent": 2.6800,  # barely a regression by itself
                    "written_paths": [],
                    "training_shards": [],
                    "corpus_stats": {"vocabulary_fraction": 0.369, "distinct_3": 0.0375},
                }
            )


def test_writes_confined_invariant(tmp_path):
    guard = PathGuard([tmp_path / "runs"])
    (tmp_path / "runs").mkdir()
    inv = WritesConfined(guard)
    assert inv.check(_ctx(written_paths=[tmp_path / "runs" / "a.bin"])).passed
    assert not inv.check(_ctx(written_paths=[tmp_path / "escaped.bin"])).passed


def test_suite_gate_raises_on_any_failure(tmp_path):
    lineage = Lineage(tmp_path / "l.jsonl")
    guard = PathGuard([tmp_path])
    suite = default_suite(lineage, guard, tolerance=0.01)
    suite.gate(_ctx())  # clean context passes
    with pytest.raises(PromotionRefused, match="no_regression"):
        suite.gate(_ctx(val_bpb=2.0))


def test_suite_fails_closed_when_an_invariant_raises():
    class Exploding:
        name = "exploding"
        axiom = SELF_PRESERVATION

        def check(self, ctx):
            raise RuntimeError("boom")

    results = InvariantSuite([Exploding()]).verify({})
    assert not results[0].passed
    assert "RuntimeError" in results[0].detail


def test_empty_suite_is_rejected():
    """A suite with no invariants would gate nothing while looking like a gate."""
    with pytest.raises(ValueError):
        InvariantSuite([])


def test_promotion_refused_names_every_failure(tmp_path):
    lineage = Lineage(tmp_path / "l.jsonl")
    guard = PathGuard([tmp_path / "runs"])
    (tmp_path / "runs").mkdir()
    suite = default_suite(lineage, guard, tolerance=0.01)
    with pytest.raises(PromotionRefused) as exc:
        suite.gate(_ctx(val_bpb=9.9, written_paths=[tmp_path / "nope.bin"]))
    assert len(exc.value.failures) == 2


def test_lineage_intact_invariant_reports_corruption(tmp_path):
    path = tmp_path / "l.jsonl"
    lineage = Lineage(path)
    lineage.append({"a": 1})
    assert LineageIntact(lineage).check({}).passed
    path.write_text('{"index": 0, "prev": "x", "digest": "y", "payload": {}}\n')
    assert not LineageIntact(lineage).check({}).passed
