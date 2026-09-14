"""Evaluation harness, probes, statistics and reporting.

The statistics tests carry the most weight here. Everything downstream —
every comparison, every claim that one configuration beats another — rests on
the noise floor and the significance test being right.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from gitai.data import BatchSampler, tokenize_to_shards
from gitai.eval import (
    EVAL_SUITE_VERSION,
    CopyTask,
    DyckTask,
    EvalResult,
    InductionProbe,
    ModularArithmeticTask,
    SeedGroup,
    SortTask,
    accuracy_at_k,
    build_index,
    compare_groups,
    comparison_table,
    connect,
    evaluate,
    evaluate_task,
    frequent_tokens,
    leaderboard,
    load_groups,
    noise_floor,
    permutation_test,
    perplexity,
)
from gitai.model import ModelConfig, Transformer
from gitai.tokenizer import CharTokenizer

DOCS = [f"document {i}: the fox ran across the field and into the trees. " * 4 for i in range(40)]


# ============================================================ statistics


class TestSeedGroup:
    def test_mean_and_std(self):
        group = SeedGroup("x", (1.0, 2.0, 3.0))
        assert group.mean == 2.0
        assert group.std == pytest.approx(1.0)
        assert group.spread == 2.0
        assert group.n == 3

    def test_single_seed_has_no_spread(self):
        """And must be rendered as a warning, not as a measurement."""
        group = SeedGroup("x", (1.5,))
        assert group.std == 0.0
        assert group.sem == 0.0
        assert "no spread" in str(group)

    def test_empty_group_is_harmless(self):
        assert SeedGroup("x", ()).spread == 0.0


class TestPermutationTest:
    def test_identical_groups_are_maximally_unsurprising(self):
        assert permutation_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_cleanly_separated_groups_are_significant(self):
        """With 5 v 5 there are exactly C(10,5)=252 splits, and only the observed
        one and its mirror are this extreme — so p = 2/252."""
        p = permutation_test([1.0, 2.0, 3.0, 4.0, 5.0], [11.0, 12.0, 13.0, 14.0, 15.0])
        assert p == pytest.approx(2 / 252, abs=1e-6)

    def test_overlapping_groups_are_not_significant(self):
        assert permutation_test([1.0, 3.0, 5.0, 7.0, 9.0], [2.0, 4.0, 6.0, 8.0, 10.0]) > 0.5

    def test_is_symmetric(self):
        a, b = [1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 7.0]
        assert permutation_test(a, b) == permutation_test(b, a)

    def test_empty_input_returns_one(self):
        assert permutation_test([], [1.0]) == 1.0

    def test_falls_back_to_sampling_when_exhaustive_is_too_large(self):
        a = list(range(15))
        b = list(range(15, 30))
        assert 0.0 <= permutation_test(a, b, max_exact=500) <= 1.0


class TestComparison:
    def test_effect_below_the_noise_floor_is_reported_as_no_effect(self):
        """The single most important guard in the project. A difference smaller
        than seed variance does not exist, however clean the plot looks."""
        a = SeedGroup("a", (1.000, 1.010, 1.020))
        b = SeedGroup("b", (1.005, 1.015, 1.025))
        comparison = compare_groups(a, b, noise_floor=0.05)
        assert comparison.effect_in_noise_units < 1.0
        assert "below the noise floor" in comparison.verdict

    def test_large_clean_effect_is_reported_as_significant(self):
        """Labels are deliberately multi-character and distinctive: single-letter
        labels make substring assertions meaningless, because "b" matches inside
        the word "better" and the test passes whichever way the logic goes."""
        a = SeedGroup("baseline", (1.00, 1.01, 1.02, 1.03, 1.04))
        b = SeedGroup("candidate", (0.50, 0.51, 0.52, 0.53, 0.54))
        comparison = compare_groups(a, b, noise_floor=0.01)
        assert "candidate better" in comparison.verdict
        assert "baseline better" not in comparison.verdict

    def test_lower_is_better_flag_picks_the_right_winner(self):
        """The direction of the comparison. Getting this backwards would report
        the loser as the winner in every table in the project."""
        low = SeedGroup("scored_low", (1.0, 1.0, 1.0, 1.0, 1.0))
        high = SeedGroup("scored_high", (2.0, 2.0, 2.0, 2.0, 2.0))

        minimising = compare_groups(low, high, 0.01, lower_is_better=True)
        assert "scored_low better" in minimising.verdict

        maximising = compare_groups(low, high, 0.01, lower_is_better=False)
        assert "scored_high better" in maximising.verdict

    def test_three_seeds_per_arm_cannot_reach_significance(self):
        """A permutation test over 3v3 enumerates C(6,3)=20 splits, so the
        smallest possible p-value is 2/20 = 0.10. Reporting such a comparison as
        "not significant" would be an arithmetic property of the sample size
        masquerading as a null result — so it is reported as underpowered."""
        a = SeedGroup("control", (2.03, 2.04, 2.05))
        b = SeedGroup("replace", (2.33, 2.35, 2.37))
        comparison = compare_groups(a, b, noise_floor=0.004)
        assert comparison.underpowered
        assert comparison.min_achievable_p == pytest.approx(0.1)
        assert "UNDERPOWERED" in comparison.verdict
        assert "control better" in comparison.verdict
        assert comparison.effect_in_noise_units > 50

    def test_five_seeds_per_arm_can_reach_significance(self):
        """2/C(10,5) = 0.0079, so five seeds per arm clears the 0.05 bar."""
        a = SeedGroup("control", (2.03, 2.04, 2.05, 2.06, 2.07))
        b = SeedGroup("replace", (2.33, 2.35, 2.37, 2.39, 2.41))
        comparison = compare_groups(a, b, noise_floor=0.004)
        assert not comparison.underpowered
        assert "UNDERPOWERED" not in comparison.verdict
        assert "control better" in comparison.verdict

    def test_noise_floor_veto_outranks_the_underpowered_warning(self):
        """A tiny effect is no effect regardless of how many seeds there are."""
        a = SeedGroup("a_group", (1.0000, 1.0001, 1.0002))
        b = SeedGroup("b_group", (1.0003, 1.0004, 1.0005))
        assert "below the noise floor" in compare_groups(a, b, noise_floor=0.05).verdict

    def test_separated_but_tiny_effect_is_still_rejected(self):
        """Statistically significant yet below the noise floor: the noise-floor
        check must veto, or you publish an artefact of a too-small sample."""
        a = SeedGroup("a", (1.0000, 1.0001, 1.0002, 1.0003, 1.0004))
        b = SeedGroup("b", (1.0010, 1.0011, 1.0012, 1.0013, 1.0014))
        comparison = compare_groups(a, b, noise_floor=0.05)
        assert comparison.p_value < 0.01
        assert "below the noise floor" in comparison.verdict


# ============================================================== metrics


def test_perplexity_of_uniform_predictions_is_vocab_size():
    assert perplexity(math.log(50)) == pytest.approx(50.0)


def test_perplexity_guards_against_overflow():
    assert math.isfinite(perplexity(10_000.0))


def test_accuracy_at_k():
    logits = torch.tensor([[[0.0, 5.0, 1.0, 2.0]]])
    assert accuracy_at_k(logits, torch.tensor([[1]]), k=1) == 1.0
    assert accuracy_at_k(logits, torch.tensor([[3]]), k=1) == 0.0
    assert accuracy_at_k(logits, torch.tensor([[3]]), k=2) == 1.0


def test_eval_result_carries_the_suite_version():
    """Changing the eval invalidates comparison with earlier runs. Without the
    stamp that invalidation is silent."""
    result = EvalResult()
    assert result.suite_version == EVAL_SUITE_VERSION
    assert "eval suite" in str(result)
    assert result.to_dict()["suite_version"] == EVAL_SUITE_VERSION


# ============================================================== harness


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    tokenizer = CharTokenizer.train(DOCS)
    tokenize_to_shards(DOCS[:32], tokenizer, root, split="train")
    tokenize_to_shards(DOCS[32:], tokenizer, root, split="val")
    return root, tokenizer


@pytest.fixture
def model(corpus) -> Transformer:
    _, tokenizer = corpus
    torch.manual_seed(0)
    return Transformer(
        ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4)
    )


def test_evaluate_produces_a_complete_result(corpus, model):
    root, tokenizer = corpus
    sampler = BatchSampler(root, "val", seq_len=32)
    result = evaluate(model, sampler, batch_size=4, tokenizer=tokenizer, prompts=("the",))

    assert result.suite_version == EVAL_SUITE_VERSION
    assert result.split == "val"
    assert result.tokens > 0
    assert result.bits_per_byte > 0
    assert result.loss > 0
    assert 0.0 <= result.accuracy_top1 <= result.accuracy_top5 <= 1.0
    assert len(result.samples) == 1


def test_untrained_model_scores_near_uniform(corpus, model):
    """An untrained model must be surprised by exactly log2(vocab) bits per
    token. A different number means the eval is wrong, not the model."""
    root, tokenizer = corpus
    sampler = BatchSampler(root, "val", seq_len=32)
    result = evaluate(model, sampler, batch_size=4, prompts=())
    assert result.bits_per_token == pytest.approx(math.log2(tokenizer.vocab_size), abs=0.4)


def test_evaluation_is_deterministic(corpus, model):
    root, _ = corpus
    sampler = BatchSampler(root, "val", seq_len=32)
    a = evaluate(model, sampler, batch_size=4, prompts=())
    b = evaluate(model, sampler, batch_size=4, prompts=())
    assert a.bits_per_byte == b.bits_per_byte
    assert a.accuracy_top1 == b.accuracy_top1


def test_samples_are_reproducible_with_a_fixed_seed(corpus, model):
    root, tokenizer = corpus
    sampler = BatchSampler(root, "val", seq_len=32)
    a = evaluate(model, sampler, 4, tokenizer=tokenizer, prompts=("the fox",), seed=7)
    b = evaluate(model, sampler, 4, tokenizer=tokenizer, prompts=("the fox",), seed=7)
    assert a.samples == b.samples


def test_truncated_evaluation_still_reports_correct_bits_per_byte(corpus, model):
    """The byte denominator must scale with how much was actually evaluated."""
    root, _ = corpus
    sampler = BatchSampler(root, "val", seq_len=32)
    full = evaluate(model, sampler, batch_size=4, prompts=())
    partial = evaluate(model, sampler, batch_size=4, max_batches=1, prompts=())
    assert partial.tokens < full.tokens
    assert partial.bits_per_byte == pytest.approx(full.bits_per_byte, rel=0.25)


def test_evaluate_restores_training_mode(corpus, model):
    root, _ = corpus
    model.train()
    evaluate(model, BatchSampler(root, "val", 32), batch_size=4, prompts=())
    assert model.training


def test_frequent_tokens_returns_real_ids(corpus, model):
    root, tokenizer = corpus
    pool = frequent_tokens(BatchSampler(root, "train", 32), top=10)
    assert len(pool) == 10
    assert pool.max() < tokenizer.vocab_size


# =============================================================== probes


def test_induction_probe_on_an_untrained_model_shows_no_induction(model):
    """An untrained model predicts the second copy exactly as badly as the
    first, so the score sits at zero. That is the control."""
    scores = InductionProbe(block_len=12, trials=8).run(model)
    assert abs(scores["induction_score_bits"]) < 1.0
    assert scores["induction_first_copy_bits"] > 0


def test_induction_probe_reports_every_metric(model):
    scores = InductionProbe(block_len=12, trials=8).run(model)
    assert set(scores) == {
        "induction_first_copy_bits",
        "induction_second_copy_bits",
        "induction_score_bits",
        "induction_late_bits",
        "induction_uniform_baseline_bits",
    }


def test_induction_probe_respects_a_token_pool(model):
    pool = np.array([1, 2, 3, 4, 5])
    scores = InductionProbe(block_len=8, trials=4, token_pool=pool).run(model)
    assert scores["induction_uniform_baseline_bits"] == pytest.approx(math.log2(5))


def test_induction_per_position_has_one_entry_per_position(model):
    curve = InductionProbe(block_len=12, trials=4).per_position(model)
    assert len(curve) == 12
    assert np.isfinite(curve).all()


@pytest.mark.slow
def test_a_model_trained_to_copy_develops_induction():
    """The positive control, and the measurement Decision 12 question D rests on.

    Train on repeated random sequences and the induction score must rise well
    above zero — the model has learned to look back at what followed the previous
    occurrence of the current token.
    """
    vocab, block = 32, 16
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(vocab_size=vocab, seq_len=2 * block, d_model=64, n_layer=2, n_head=4)
    )
    probe = InductionProbe(block_len=block, trials=16, seed=1)
    before = probe.run(model)["induction_score_bits"]

    rng = np.random.default_rng(0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(400):
        blocks = rng.integers(0, vocab, size=(32, block))
        batch = torch.from_numpy(np.concatenate([blocks, blocks], axis=1).astype(np.int64))
        _, loss = model(batch[:, :-1], batch[:, 1:])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    after = probe.run(model)["induction_score_bits"]
    assert after > 1.0, f"no induction learned: {after:.3f} bits"
    assert after > before + 1.0


class TestSyntheticTasks:
    @pytest.mark.parametrize("task", [CopyTask(), SortTask(), ModularArithmeticTask(), DyckTask()])
    def test_sample_returns_prompt_and_answer(self, task):
        prompt, answer = task.sample(np.random.default_rng(0))
        assert prompt and answer
        assert answer.endswith("\n")

    def test_sort_answer_is_actually_sorted(self):
        prompt, answer = SortTask().sample(np.random.default_rng(0))
        numbers = [int(x) for x in answer.split()]
        assert numbers == sorted(numbers)
        assert sorted(int(x) for x in prompt.replace("->", "").split()) == numbers

    def test_copy_answer_equals_the_prompt_body(self):
        prompt, answer = CopyTask().sample(np.random.default_rng(0))
        assert prompt.replace("|", "").split() == answer.split()

    def test_modular_arithmetic_is_correct(self):
        task = ModularArithmeticTask(modulus=7)
        for _ in range(20):
            prompt, answer = task.sample(np.random.default_rng(0))
            a, _, b, _, m, _ = prompt.split()
            assert int(answer) == (int(a) + int(b)) % int(m)

    def test_dyck_answer_closes_the_prefix(self):
        task = DyckTask(max_depth=4, length=10)
        closing = {")": "(", "]": "[", "}": "{"}
        for seed in range(10):
            prefix, answer = task.sample(np.random.default_rng(seed))
            stack = []
            for char in prefix.strip() + answer.strip():
                if char in "([{":
                    stack.append(char)
                else:
                    assert stack and stack.pop() == closing[char]
            assert not stack, "sequence is not balanced"

    def test_corpus_is_deterministic(self):
        assert SortTask().corpus(20, seed=3) == SortTask().corpus(20, seed=3)
        assert SortTask().corpus(20, seed=3) != SortTask().corpus(20, seed=4)

    def test_evaluate_task_scores_an_untrained_model_near_zero(self):
        task = SortTask(length=3, maximum=9)
        tokenizer = CharTokenizer.train(task.corpus(200, seed=0))
        torch.manual_seed(0)
        model = Transformer(
            ModelConfig(
                vocab_size=tokenizer.vocab_size, seq_len=64, d_model=32, n_layer=2, n_head=4
            )
        )
        scores = evaluate_task(task, model, tokenizer, n=10)
        assert scores["sort_exact_match"] == 0.0


# =========================================================== index + report


@pytest.fixture
def runs_dir(tmp_path):
    """Three fake runs of one config plus one of another."""
    for i, (name, seed, bpb) in enumerate(
        [("nf-s0", 0, 1.90), ("nf-s1", 1, 1.88), ("nf-s2", 2, 1.92), ("other", 0, 1.50)]
    ):
        run = tmp_path / f"2026010{i}-000000-{name}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "config.yaml").write_text(
            json.dumps(
                {
                    "model": {
                        "d_model": 128,
                        "n_layer": 4,
                        "n_head": 4,
                        "vocab_size": 2048,
                        "seq_len": 128,
                    },
                    "training": {
                        "name": name,
                        "seed": seed,
                        "steps": 1000,
                        "rung": "v6_modern",
                        "lr": 3e-3,
                        "batch_size": 16,
                    },
                }
            )
        )
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "git": {"sha": "abc123", "dirty": False},
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "tokenizer_fingerprint": "deadbeef",
                    "parameters": {"total": 1_049_216, "non_embedding": 787_072},
                }
            )
        )
        (run / "checkpoints" / "best.json").write_text(
            json.dumps({"step": 999, "val_bpb": bpb, "val_loss": bpb * 1.96})
        )
        (run / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "loss": 7.6, "tokens_per_sec": 19000, "elapsed": 1.0})
            + "\n"
            + json.dumps({"step": 999, "loss": 3.5, "tokens_per_sec": 19200, "elapsed": 100.0})
            + "\n"
        )
    return tmp_path


def test_index_builds_and_is_queryable(runs_dir):
    db = build_index(runs_dir)
    connection = connect(db)
    rows = connection.execute("SELECT * FROM runs ORDER BY name").fetchall()
    assert len(rows) == 4
    assert rows[0]["params_non_embedding"] == 787_072
    assert rows[0]["git_sha"] == "abc123"
    # 4 runs x 2 lines x 3 numeric keys (loss, tokens_per_sec, elapsed); `step`
    # is the row key, not a metric.
    assert connection.execute("SELECT count(*) FROM metrics").fetchone()[0] == 24


def test_index_is_rebuildable_from_scratch(runs_dir):
    """A derived index, never the source of truth — delete it and rebuild."""
    db = build_index(runs_dir)
    first = connect(db).execute("SELECT count(*) FROM runs").fetchone()[0]
    db.unlink()
    rebuilt = build_index(runs_dir)
    assert connect(rebuilt).execute("SELECT count(*) FROM runs").fetchone()[0] == first


def test_index_skips_directories_that_are_not_runs(runs_dir):
    (runs_dir / "not-a-run").mkdir()
    db = build_index(runs_dir)
    assert connect(db).execute("SELECT count(*) FROM runs").fetchone()[0] == 4


def test_groups_fold_seed_suffixes_together(runs_dir):
    groups = load_groups(build_index(runs_dir))
    assert "nf" in groups
    assert groups["nf"].n == 3
    assert groups["other"].n == 1


def test_noise_floor_comes_from_the_largest_group(runs_dir):
    floor = noise_floor(build_index(runs_dir), name_like="nf%")
    assert floor == pytest.approx(SeedGroup("x", (1.88, 1.90, 1.92)).std)


def test_noise_floor_is_none_without_repeats(tmp_path):
    """Better to say 'unmeasured' than to invent a threshold."""
    assert noise_floor(build_index(tmp_path)) is None


def test_leaderboard_and_comparison_table_render(runs_dir):
    db = build_index(runs_dir)
    assert "best_bpb" in leaderboard(db)
    table = comparison_table(load_groups(db), floor=0.02)
    assert "noise floor" in table
    assert "single seed" in table  # the n=1 group must be flagged


def test_leaderboard_with_no_runs(tmp_path):
    assert "no runs indexed" in leaderboard(build_index(tmp_path))
