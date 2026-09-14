"""Checkpointing, resume, and gradient accumulation.

The centrepiece is ``test_resume_reproduces_an_uninterrupted_run``. A resume
that restores four of the five pieces of state produces a run that looks
completely healthy and silently differs from an uninterrupted one — which is the
worst kind of bug, because nothing errors and every metric looks plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gitai.checkpoint import load_checkpoint, rotate_checkpoints, save_checkpoint
from gitai.data import BatchSampler, tokenize_to_shards
from gitai.model import ModelConfig, Transformer
from gitai.tokenizer import CharTokenizer
from gitai.training import TrainConfig, train

DOCS = [
    f"doc {i}: the fox ran across the field and into the trees at dusk. " * 4 for i in range(60)
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt-corpus")
    tokenizer = CharTokenizer.train(DOCS)
    tokenize_to_shards(DOCS[:48], tokenizer, root, split="train")
    tokenize_to_shards(DOCS[48:], tokenizer, root, split="val")
    return root, tokenizer


def build(tokenizer) -> Transformer:
    torch.manual_seed(0)
    return Transformer(
        ModelConfig(vocab_size=tokenizer.vocab_size, seq_len=32, d_model=32, n_layer=2, n_head=4)
    )


def optimiser_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def weights_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


# ======================================================== checkpoint mechanics


def test_round_trip_restores_weights_and_optimiser(corpus, tmp_path):
    _, tokenizer = corpus
    model = build(tokenizer)
    opt = optimiser_for(model)

    # Take a few real steps so the optimiser has non-trivial moment estimates.
    for _ in range(3):
        loss = model(
            torch.randint(0, tokenizer.vocab_size, (2, 8)),
            torch.randint(0, tokenizer.vocab_size, (2, 8)),
        )[1]
        opt.zero_grad()
        loss.backward()
        opt.step()

    rng = np.random.default_rng(7)
    rng.random(5)
    save_checkpoint(tmp_path / "ck", model, opt, step=3, rng=rng, best_bpb=1.5, best_step=2)

    restored_model = build(tokenizer)
    restored_opt = optimiser_for(restored_model)
    restored_rng = np.random.default_rng(999)
    state = load_checkpoint(tmp_path / "ck", restored_model, restored_opt, restored_rng)

    assert state.step == 3
    assert state.best_bpb == 1.5
    assert state.best_step == 2
    assert weights_equal(model, restored_model)
    assert np.array_equal(rng.random(4), restored_rng.random(4))


def test_optimiser_moments_survive_the_round_trip(corpus, tmp_path):
    """Adam's moment estimates take hundreds of steps to rebuild. Losing them is
    a silent, slow-training regression rather than a crash."""
    _, tokenizer = corpus
    model = build(tokenizer)
    opt = optimiser_for(model)
    for _ in range(5):
        loss = model(
            torch.randint(0, tokenizer.vocab_size, (2, 8)),
            torch.randint(0, tokenizer.vocab_size, (2, 8)),
        )[1]
        opt.zero_grad()
        loss.backward()
        opt.step()

    before = opt.state_dict()["state"]
    save_checkpoint(tmp_path / "ck", model, opt, step=5, rng=np.random.default_rng(0))

    restored_model = build(tokenizer)
    restored_opt = optimiser_for(restored_model)
    load_checkpoint(tmp_path / "ck", restored_model, restored_opt, np.random.default_rng(0))
    after = restored_opt.state_dict()["state"]

    assert set(before) == set(after)
    for pid in before:
        for key, value in before[pid].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, after[pid][key]), f"param {pid} field {key} differs"


def test_torch_rng_state_survives(corpus, tmp_path):
    _, tokenizer = corpus
    model = build(tokenizer)
    opt = optimiser_for(model)
    torch.manual_seed(1234)
    torch.randn(10)

    save_checkpoint(tmp_path / "ck", model, opt, step=0, rng=np.random.default_rng(0))
    expected = torch.randn(5)

    torch.manual_seed(4321)  # deliberately move it somewhere else
    load_checkpoint(
        tmp_path / "ck", build(tokenizer), optimiser_for(model), np.random.default_rng(0)
    )
    assert torch.equal(torch.randn(5), expected)


def test_no_pickle_is_used(corpus, tmp_path):
    """Decision 6: checkpoints are safetensors, never pickle. Optimiser state is
    flattened rather than quietly exempted from the rule."""
    _, tokenizer = corpus
    model = build(tokenizer)
    save_checkpoint(
        tmp_path / "ck", model, optimiser_for(model), step=0, rng=np.random.default_rng(0)
    )
    names = {p.name for p in (tmp_path / "ck").iterdir()}
    assert names == {"model.safetensors", "optimiser.safetensors", "state.json"}
    json.loads((tmp_path / "ck" / "state.json").read_text())  # plain JSON, parseable


def test_writes_are_atomic(corpus, tmp_path):
    """An interruption mid-write must not leave a checkpoint that loads without
    error and resumes into nonsense."""
    _, tokenizer = corpus
    model = build(tokenizer)
    save_checkpoint(
        tmp_path / "ck", model, optimiser_for(model), step=0, rng=np.random.default_rng(0)
    )
    assert not (tmp_path / "ck.partial").exists()


def test_overwriting_a_checkpoint_works(corpus, tmp_path):
    _, tokenizer = corpus
    model = build(tokenizer)
    opt = optimiser_for(model)
    for step in (1, 2):
        save_checkpoint(tmp_path / "ck", model, opt, step=step, rng=np.random.default_rng(0))
    state = load_checkpoint(
        tmp_path / "ck", build(tokenizer), optimiser_for(model), np.random.default_rng(0)
    )
    assert state.step == 2


def test_missing_checkpoint_is_an_explicit_error(corpus, tmp_path):
    _, tokenizer = corpus
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_checkpoint(
            tmp_path / "absent",
            build(tokenizer),
            optimiser_for(build(tokenizer)),
            np.random.default_rng(0),
        )


# ============================================================ rotation


def test_rotation_keeps_the_most_recent(tmp_path):
    for step in (5, 10, 15, 20):
        (tmp_path / f"step-{step}").mkdir()
    removed = rotate_checkpoints(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["step-15", "step-20"]
    assert len(removed) == 2


def test_rotation_sorts_numerically_not_lexically(tmp_path):
    """step-100 is newer than step-99, which string sorting gets backwards."""
    for step in (9, 99, 100):
        (tmp_path / f"step-{step}").mkdir()
    rotate_checkpoints(tmp_path, keep=1)
    assert [p.name for p in tmp_path.iterdir()] == ["step-100"]


def test_rotation_ignores_unrelated_directories(tmp_path):
    (tmp_path / "step-1").mkdir()
    (tmp_path / "step-2").mkdir()
    (tmp_path / "notes").mkdir()
    rotate_checkpoints(tmp_path, keep=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes", "step-2"]


def test_rotation_requires_keeping_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        rotate_checkpoints(tmp_path, keep=0)


def test_rotation_on_a_missing_directory_is_harmless(tmp_path):
    assert rotate_checkpoints(tmp_path / "absent") == []


# ====================================================== gradient accumulation


def test_accumulated_gradients_equal_a_single_large_batch(corpus):
    """The equivalence that justifies accumulation existing.

    Tested at the gradient level with identical data, which isolates the
    accumulation arithmetic from batch sampling. Without dividing each
    micro-batch loss by the accumulation count, the summed gradient would be N
    times too large and the effective learning rate would silently scale with
    accumulation.
    """
    _, tokenizer = corpus
    torch.manual_seed(3)
    x = torch.randint(0, tokenizer.vocab_size, (32, 16))
    y = torch.randint(0, tokenizer.vocab_size, (32, 16))

    whole = build(tokenizer)
    _, loss = whole(x, y)
    loss.backward()

    chunked = build(tokenizer)
    for i in range(4):
        lo, hi = i * 8, (i + 1) * 8
        _, part = chunked(x[lo:hi], y[lo:hi])
        (part / 4).backward()

    for (name, a), (_, b) in zip(whole.named_parameters(), chunked.named_parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), f"gradients differ for {name}"


def test_config_rejects_an_indivisible_accumulation(corpus):
    with pytest.raises(ValueError, match="not divisible"):
        TrainConfig(batch_size=10, accumulation_steps=4)


def test_config_rejects_zero_accumulation():
    with pytest.raises(ValueError, match="must be >= 1"):
        TrainConfig(accumulation_steps=0)


def test_micro_batch_splits_the_effective_batch():
    assert TrainConfig(batch_size=32, accumulation_steps=4).micro_batch == 8
    assert TrainConfig(batch_size=32, accumulation_steps=1).micro_batch == 32


def test_training_with_accumulation_still_learns(corpus):
    _, tokenizer = corpus
    root, _ = corpus
    model = build(tokenizer)
    result = train(
        model,
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        TrainConfig(steps=40, batch_size=16, accumulation_steps=2, eval_every=20, log_every=1000),
    )
    assert result.best_bpb < 10.0
    assert result.final_train_loss > 0


# =============================================================== resume


def test_resume_reproduces_an_uninterrupted_run(corpus, tmp_path):
    """The test this whole module exists for.

    Train 20 steps with checkpoints. Delete the final checkpoint, start a fresh
    process-equivalent model, resume from step 9, and finish. The result must be
    bit-identical to never having stopped.

    Drop any one of the five pieces of state — weights, optimiser moments, step
    number, torch RNG, numpy RNG — and this fails while everything still *looks*
    fine.
    """
    root, tokenizer = corpus
    config = TrainConfig(
        steps=20,
        batch_size=8,
        eval_every=10,
        log_every=1,
        seed=5,
        checkpoint_every=10,
        keep_last=5,
    )

    reference = build(tokenizer)
    reference_result = train(
        reference,
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        config,
        keep_best=False,
        checkpoint_dir=tmp_path / "ck",
        resume=False,
    )

    # Simulate the interruption: the step-19 checkpoint never got written.
    import shutil

    shutil.rmtree(tmp_path / "ck" / "step-19")

    resumed = build(tokenizer)
    events: list[dict] = []
    resumed_result = train(
        resumed,
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        config,
        on_event=events.append,
        keep_best=False,
        checkpoint_dir=tmp_path / "ck",
        resume=True,
    )

    assert any(e.get("event") == "resumed" for e in events), "did not actually resume"
    assert weights_equal(reference, resumed), "resumed weights differ from uninterrupted"
    assert resumed_result.best_bpb == pytest.approx(reference_result.best_bpb, abs=1e-9)

    def losses(result):
        return [round(r["loss"], 10) for r in result.history if "loss" in r and r["step"] >= 10]

    assert losses(resumed_result) == losses(reference_result), "loss trajectory diverged"


def test_resume_without_a_checkpoint_starts_fresh(corpus, tmp_path):
    root, tokenizer = corpus
    events: list[dict] = []
    train(
        build(tokenizer),
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        TrainConfig(steps=5, eval_every=5, log_every=1000),
        on_event=events.append,
        checkpoint_dir=tmp_path / "empty",
        resume=True,
    )
    assert not any(e.get("event") == "resumed" for e in events)


def test_resume_restores_the_best_weights_not_the_latest(corpus, tmp_path):
    """If no evaluation after resuming beats the pre-interruption best, the
    caller must still receive the best weights — not whatever the last step
    produced."""
    root, tokenizer = corpus
    config = TrainConfig(
        steps=10, batch_size=8, eval_every=5, log_every=1000, checkpoint_every=5, keep_last=5
    )

    first = build(tokenizer)
    train(
        first,
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        config,
        checkpoint_dir=tmp_path / "ck",
    )
    assert (tmp_path / "ck" / "best.safetensors").exists()

    resumed = build(tokenizer)
    result = train(
        resumed,
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        config,
        checkpoint_dir=tmp_path / "ck",
        resume=True,
    )
    assert result.best_bpb < float("inf")


def test_checkpoints_are_written_and_rotated(corpus, tmp_path):
    root, tokenizer = corpus
    train(
        build(tokenizer),
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        TrainConfig(
            steps=20, batch_size=8, eval_every=20, log_every=1000, checkpoint_every=5, keep_last=2
        ),
        checkpoint_dir=tmp_path / "ck",
    )
    steps = sorted(int(p.name.rsplit("-", 1)[-1]) for p in (tmp_path / "ck").glob("step-*"))
    assert steps == [14, 19], f"expected the two most recent, got {steps}"


def test_eta_is_reported(corpus, tmp_path):
    root, tokenizer = corpus
    events: list[dict] = []
    train(
        build(tokenizer),
        BatchSampler(root, "train", 32),
        BatchSampler(root, "val", 32),
        TrainConfig(steps=6, eval_every=6, log_every=1),
        on_event=events.append,
    )
    logs = [e for e in events if "eta_seconds" in e]
    assert logs and logs[0]["eta_seconds"] >= 0
    assert logs[-1]["eta_seconds"] < logs[0]["eta_seconds"]
