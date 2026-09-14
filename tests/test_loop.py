"""The bounded self-improvement loop.

Most of these are safety tests. The loop's only privileged action is promotion,
so that is what needs guarding — but the properties that make it *stoppable* and
*auditable* matter just as much, because they are what let it run unattended at
all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gitai.data import BatchSampler, tokenize_to_shards
from gitai.loop import ImprovementLoop, LoopConfig, OutOfSpace, Proposal, SearchSpace
from gitai.model import RUNGS, ModelConfig, Transformer
from gitai.safety import (
    Budget,
    HaltRequested,
    HaltSwitch,
    Lineage,
    PathGuard,
    default_suite,
)
from gitai.tokenizer import ByteBPETokenizer

DOCS = [
    f"scene {i}: the fox ran across the field and into the trees at dusk. " * 3 for i in range(80)
]


# ============================================================== search space


class TestSearchSpace:
    def test_refuses_a_space_containing_the_collapse_regime(self):
        """R8 encoded as a constraint rather than a preference: the synthetic-only
        lineage that reached 4.53 BPB — worse than a bigram — must be unreachable,
        not merely caught by the gate afterwards."""
        with pytest.raises(OutOfSpace, match=r"4\.53 BPB"):
            SearchSpace(synthetic_fraction=(0.0, 1.0))

    def test_allows_fractions_that_keep_enough_real_data(self):
        space = SearchSpace(synthetic_fraction=(0.0, 0.75), min_real_fraction=0.25)
        assert space.size > 0

    def test_every_sample_is_inside_the_space(self):
        space = SearchSpace()
        rng = np.random.default_rng(0)
        for i in range(50):
            space.validate(space.sample(i, rng))  # raises if not

    def test_every_sample_keeps_the_real_data_floor(self):
        space = SearchSpace()
        rng = np.random.default_rng(1)
        for i in range(50):
            assert space.sample(i, rng).real_fraction >= space.min_real_fraction

    def test_validate_rejects_an_outside_proposal(self):
        """Checked on the way in, so a proposal from a resumed run or a
        hand-written config is held to the same bound as a sampled one."""
        space = SearchSpace()
        smuggled = Proposal(
            iteration=0,
            synthetic_fraction=0.99,
            temperature=1.0,
            top_k=None,
            lr=1e-3,
            steps=300,
            seed=0,
        )
        with pytest.raises(OutOfSpace, match="real_fraction"):
            space.validate(smuggled)

    def test_validate_rejects_an_undeclared_hyperparameter(self):
        space = SearchSpace(lr=(1e-3,))
        with pytest.raises(OutOfSpace, match="lr"):
            space.validate(Proposal(0, 0.0, 1.0, None, lr=9.9, steps=300, seed=0))

    def test_sampling_is_reproducible(self):
        space = SearchSpace()
        a = space.sample(0, np.random.default_rng(3))
        b = space.sample(0, np.random.default_rng(3))
        assert a == b

    def test_size_counts_the_grid(self):
        space = SearchSpace(
            synthetic_fraction=(0.0, 0.25),
            temperature=(1.0,),
            top_k=(None,),
            lr=(1e-3, 3e-3),
            steps=(100,),
        )
        assert space.size == 4

    def test_rejects_an_empty_dimension(self):
        with pytest.raises(ValueError, match="at least one option"):
            SearchSpace(temperature=())

    def test_rejects_a_nonpositive_temperature(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            SearchSpace(temperature=(0.0,))

    def test_describe_names_the_r8_floor(self):
        assert "R8" in SearchSpace().describe()


# ================================================================= the loop


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("loop-corpus")
    tokenizer = ByteBPETokenizer.train(DOCS, vocab_size=320, special_tokens=["<|endoftext|>"])
    eot = tokenizer.special_tokens["<|endoftext|>"]
    tokenize_to_shards(DOCS[:64], tokenizer, root, split="train", document_separator=eot)
    tokenize_to_shards(DOCS[64:], tokenizer, root, split="val", document_separator=eot)
    return root, tokenizer


def build_loop(tmp_path, setup, **overrides):
    root, tokenizer = setup
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            seq_len=32,
            d_model=32,
            n_layer=2,
            n_head=4,
            **RUNGS["v6_modern"],
        )
    )
    workspace = tmp_path / "loop"
    workspace.mkdir(parents=True, exist_ok=True)
    guard = PathGuard([workspace])
    lineage = Lineage(tmp_path / "lineage.jsonl")

    defaults = dict(
        workspace=workspace,
        incumbent=model,
        tokenizer=tokenizer,
        real_documents=DOCS[:64],
        val_sampler=BatchSampler(root, "val", 32),
        space=SearchSpace(
            synthetic_fraction=(0.0, 0.25), temperature=(1.0,), lr=(3e-3,), steps=(5,)
        ),
        suite=default_suite(lineage, guard, tolerance=0.05),
        lineage=lineage,
        guard=guard,
        halt=HaltSwitch(tmp_path / "control" / "HALT"),
        budget=Budget(max_iterations=3, max_wall_seconds=600),
        config=LoopConfig(iterations=2, batch_size=4, seq_len=32, eval_batches=2, gen_batch=8),
    )
    defaults.update(overrides)
    return ImprovementLoop(**defaults)


class TestLoopSafety:
    def test_a_pending_halt_stops_before_any_work(self, tmp_path, setup):
        """The stop button must win at the very first boundary."""
        loop = build_loop(tmp_path, setup)
        loop.halt.request("operator said stop")
        result = loop.run()
        assert result.iterations == []
        assert "operator said stop" in result.stopped_because

    def test_an_exhausted_budget_stops_the_loop(self, tmp_path, setup):
        loop = build_loop(
            tmp_path,
            setup,
            budget=Budget(max_iterations=1, max_wall_seconds=600),
            config=LoopConfig(iterations=10, batch_size=4, seq_len=32, eval_batches=2, gen_batch=8),
        )
        result = loop.run()
        assert len(result.iterations) == 1
        assert "budget" in result.stopped_because

    def test_halt_is_not_swallowed_by_the_loops_own_handler(self, tmp_path, setup):
        """HaltRequested derives from BaseException so `except Exception` cannot
        catch it. The loop catches it deliberately, to record why — and then ends,
        never continuing."""
        assert issubclass(HaltRequested, BaseException)
        assert not issubclass(HaltRequested, Exception)

    def test_a_zero_wall_clock_budget_stops_immediately(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup, budget=Budget(max_iterations=99, max_wall_seconds=0.0))
        result = loop.run()
        assert result.iterations == []
        assert "wall-clock" in result.stopped_because

    def test_promoted_models_are_never_overwritten(self, tmp_path, setup):
        """A promoted model is part of the lineage; overwriting one would make the
        history unreconstructible."""
        loop = build_loop(tmp_path, setup)
        loop.incumbent_bpb = 99.0  # make anything an improvement
        loop._save_promoted(0, loop.incumbent)
        with pytest.raises(FileExistsError, match="never overwritten"):
            loop._save_promoted(0, loop.incumbent)

    def test_a_promoted_model_can_actually_be_loaded_back(self, tmp_path, setup):
        """The sibling test above refuses to overwrite a promoted model so the
        lineage stays reconstructible. That guarantee was hollow: safetensors
        stores tensors, not the class that produced them, and the loop never
        recorded the architecture. Reconstructibility means a checkpoint plus
        what is on disk is enough to get the model back, so assert the
        round-trip rather than the file's continued existence.
        """
        from safetensors.torch import load_model

        loop = build_loop(tmp_path, setup)
        path = loop._save_promoted(0, loop.incumbent)

        spec = json.loads((loop.models_dir / "model.json").read_text())
        rebuilt = Transformer(ModelConfig(**spec["config"]))
        load_model(rebuilt, str(path))  # strict: raises on any shape mismatch

        x = torch.randint(0, spec["config"]["vocab_size"], (2, 16))
        loop.incumbent.eval()
        rebuilt.eval()
        with torch.no_grad():
            assert torch.equal(rebuilt(x)[0], loop.incumbent(x)[0])


class TestLoopBehaviour:
    def test_it_records_both_promotions_and_rejections(self, tmp_path, setup):
        """The rejections are the research data — a loop that logs only its wins
        answers no question."""
        loop = build_loop(tmp_path, setup)
        result = loop.run()
        entries = loop.lineage.entries()
        assert len(entries) == len(result.iterations)
        assert all("promoted" in e.payload for e in entries)
        assert loop.lineage.verify()

    def test_two_loops_with_the_same_seed_produce_the_same_lineage(self, tmp_path, setup):
        """The loop's audit story rests on this and nothing was checking it.

        `scripts/run_loop.py` built its incumbent before seeding anything --
        `train()` seeds inside itself, long after the weights were drawn. Same
        seed, identical proposals, different measured BPB, and at iteration 12
        of a real run the difference was enough to flip a promotion into a
        rejection. A lineage you cannot regenerate is not an audit trail.

        Asserting the proposals match is not enough: those were identical
        through the entire bug, because the search space has its own generator.
        The measurements are what diverged, so the measurements are what this
        asserts.
        """
        first = build_loop(tmp_path / "a", setup).run()
        second = build_loop(tmp_path / "b", setup).run()

        assert [o.proposal for o in first.iterations] == [o.proposal for o in second.iterations]
        assert [o.val_bpb for o in first.iterations] == [o.val_bpb for o in second.iterations]
        assert [o.promoted for o in first.iterations] == [o.promoted for o in second.iterations]

    def test_the_lineage_records_which_incumbent_it_started_from(self, tmp_path, setup):
        """The loop cannot see how its incumbent was seeded -- it is handed a
        built model. Recording the starting weights' fingerprint is what lets a
        reader tell two runs apart that were supposed to be identical, instead
        of discovering it from drifting numbers three experiments later.
        """
        _, tokenizer = setup

        def incumbent(seed: int):
            torch.manual_seed(seed)
            return Transformer(
                ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    seq_len=32,
                    d_model=32,
                    n_layer=2,
                    n_head=4,
                    **RUNGS["v6_modern"],
                )
            )

        same = build_loop(tmp_path / "a", setup, incumbent=incumbent(0))
        also_same = build_loop(tmp_path / "b", setup, incumbent=incumbent(0))
        different = build_loop(tmp_path / "c", setup, incumbent=incumbent(999))

        def fingerprint(loop):
            return json.loads((loop.models_dir / "model.json").read_text())["incumbent_fingerprint"]

        assert fingerprint(same) == fingerprint(also_same)
        assert fingerprint(same) != fingerprint(different)

    def test_the_lineage_stays_tamper_evident(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup)
        loop.run()
        assert loop.lineage.verify()

    def test_every_iteration_is_recorded_with_its_proposal(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup)
        loop.run()
        for entry in loop.lineage.entries():
            assert "proposal" in entry.payload
            assert "real_fraction" in entry.payload["proposal"]

    def test_a_rejection_names_the_invariant_that_refused(self, tmp_path, setup):
        """Refuse everything, and check the reason survives into the record."""
        loop = build_loop(tmp_path, setup)
        loop.incumbent_bpb = 0.0001  # nothing can beat this
        result = loop.run()
        rejected = [o for o in result.iterations if not o.promoted]
        assert rejected
        assert any("no_regression" in r for o in rejected for r in o.reasons)

    def test_it_writes_a_result_file(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup)
        loop.run()
        payload = json.loads((loop.workspace / "result.json").read_text())
        assert "iterations" in payload and "stopped_because" in payload

    def test_the_incumbent_only_changes_on_promotion(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup)
        loop.incumbent_bpb = 0.0001  # force rejection of everything
        before = {k: v.clone() for k, v in loop.incumbent.state_dict().items()}
        loop.run()
        after = loop.incumbent.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_generated_documents_are_filtered_before_training(self, tmp_path, setup):
        """R7 found corpus degeneracy driving collapse, so filtering the generated
        half is a mitigation aimed at the measured mechanism."""
        loop = build_loop(
            tmp_path,
            setup,
            space=SearchSpace(
                synthetic_fraction=(0.5,), temperature=(1.0,), lr=(3e-3,), steps=(5,)
            ),
        )
        result = loop.run()
        reports = [o.filter_report for o in result.iterations if o.filter_report]
        assert reports, "synthetic iterations produced no filter report"
        assert all("removed" in r and "by_stage" in r for r in reports)

    def test_rejected_documents_are_retained_not_deleted(self, tmp_path, setup):
        """The rejects are the record of what the model generates badly."""
        loop = build_loop(
            tmp_path,
            setup,
            space=SearchSpace(
                synthetic_fraction=(0.5,), temperature=(1.0,), lr=(3e-3,), steps=(5,)
            ),
        )
        result = loop.run()
        removed = sum(o.filter_report["removed"] for o in result.iterations if o.filter_report)
        if removed:
            assert list((loop.workspace / "rejects").glob("iter-*.jsonl"))

    def test_reject_retention_is_capped(self, tmp_path, setup):
        """A long run must not fill the disk with rejects; the counts stay
        complete even when the retained sample is truncated."""
        loop = build_loop(
            tmp_path,
            setup,
            space=SearchSpace(
                synthetic_fraction=(0.5,), temperature=(1.0,), lr=(3e-3,), steps=(5,)
            ),
            config=LoopConfig(
                iterations=1,
                batch_size=4,
                seq_len=32,
                eval_batches=2,
                gen_batch=8,
                max_retained_rejects=2,
            ),
        )
        loop.run()
        for path in (loop.workspace / "rejects").glob("iter-*.jsonl"):
            assert len(path.read_text().splitlines()) <= 2

    def test_induction_is_measured_per_iteration(self, tmp_path, setup):
        """Loss hides structure: two models at equal BPB can differ in capability."""
        loop = build_loop(tmp_path, setup)
        result = loop.run()
        assert all(o.induction_bits is not None for o in result.iterations)
        assert all("induction_bits" in e.payload for e in loop.lineage.entries())

    def test_probes_can_be_switched_off(self, tmp_path, setup):
        loop = build_loop(
            tmp_path,
            setup,
            config=LoopConfig(
                iterations=1,
                batch_size=4,
                seq_len=32,
                eval_batches=2,
                gen_batch=8,
                measure_induction=False,
            ),
        )
        result = loop.run()
        assert all(o.induction_bits is None for o in result.iterations)

    def test_real_only_iterations_skip_filtering(self, tmp_path, setup):
        """There is nothing to filter when no synthetic data was generated, and
        the real corpus was already curated in Phase 1."""
        loop = build_loop(
            tmp_path,
            setup,
            space=SearchSpace(
                synthetic_fraction=(0.0,), temperature=(1.0,), lr=(3e-3,), steps=(5,)
            ),
        )
        result = loop.run()
        assert all(o.filter_report is None for o in result.iterations)

    def test_work_directories_are_cleaned_up(self, tmp_path, setup):
        loop = build_loop(tmp_path, setup)
        loop.run()
        assert not list(loop.work_dir.glob("iter-*"))
