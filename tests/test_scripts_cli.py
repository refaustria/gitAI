"""The command-line entry points, exercised as a user runs them.

Every script in `scripts/` was untested until this file existed, and that gap
is where two real bugs lived: `run_loop.py` built its incumbent before seeding
anything, and `train.py` parsed `--checkpoint-every`, `--keep-last` and
`--resume` and then ignored all three. Both were invisible to the library tests
because the library was correct in each case -- only the callers were not.

These run the scripts in a subprocess, because that is the thing that broke.
Importing main() and calling it would not have caught either bug.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gitai.data import tokenize_to_shards
from gitai.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent

DOCS = [
    f"story {i}: the quick brown fox jumps over the lazy dog near the river bank. " * 4
    for i in range(48)
]

TRAIN_ARGS = [
    "--steps",
    "20",
    "--batch-size",
    "4",
    "--seq-len",
    "32",
    "--d-model",
    "32",
    "--n-layer",
    "2",
    "--n-head",
    "4",
    "--lr",
    "3e-3",
    "--warmup",
    "5",
    "--eval-every",
    "10",
    "--seed",
    "0",
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cli-corpus")
    tokenizer = CharTokenizer.train(DOCS)
    tokenize_to_shards(DOCS[:40], tokenizer, root, split="train")
    tokenize_to_shards(DOCS[40:], tokenizer, root, split="val")
    tokenizer.save(root / "tokenizer.json")
    return root


def run_train(corpus: Path, *extra: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--data",
            str(corpus),
            *TRAIN_ARGS,
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def best(run_dir: Path) -> dict:
    return json.loads((run_dir / "checkpoints" / "best.json").read_text())


@pytest.mark.slow
def test_checkpoint_every_actually_writes_checkpoints(corpus, tmp_path):
    """The flag was parsed and dropped. Assert the artifact, not the exit code."""
    out = run_train(corpus, "--checkpoint-every", "5", "--keep-last", "2", "--name", "cli-ckpt")
    run_dir = Path(out.stdout.strip().rsplit("run: ", 1)[-1].strip())
    steps = sorted(p.name for p in (run_dir / "checkpoints").glob("step-*"))
    assert steps, "no step checkpoints written despite --checkpoint-every"
    assert len(steps) == 2, f"--keep-last 2 should retain exactly 2, found {steps}"


@pytest.mark.slow
def test_resume_from_the_cli_reproduces_an_uninterrupted_run(corpus, tmp_path):
    """The library's resume was tested and exact; the CLI could not reach it.

    Asserted bit-for-bit on purpose: a resume that restores four of five pieces
    of state looks completely healthy and diverges silently, so 'close enough'
    is exactly the assertion that would pass while broken.
    """
    whole = run_train(corpus, "--checkpoint-every", "5", "--keep-last", "9", "--name", "cli-whole")
    whole_dir = Path(whole.stdout.strip().rsplit("run: ", 1)[-1].strip())

    # Simulate an interruption: keep only an early checkpoint, drop the rest.
    partial = tmp_path / "interrupted"
    (partial / "checkpoints").mkdir(parents=True)
    early = whole_dir / "checkpoints" / "step-9"
    assert early.is_dir(), sorted(p.name for p in (whole_dir / "checkpoints").iterdir())
    subprocess.run(["cp", "-r", str(early), str(partial / "checkpoints")], check=True)

    resumed = run_train(corpus, "--checkpoint-every", "5", "--resume", str(partial))
    assert "resuming from" in resumed.stdout

    assert best(partial)["val_bpb"] == best(whole_dir)["val_bpb"]
    assert best(partial)["step"] == best(whole_dir)["step"]


@pytest.mark.slow
def test_resume_without_a_checkpoint_fails_loudly(corpus, tmp_path):
    """It used to train from scratch and print a normal-looking result -- the
    most expensive possible failure on a multi-hour run."""
    empty = tmp_path / "no-checkpoints"
    (empty / "checkpoints").mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--data",
            str(corpus),
            *TRAIN_ARGS,
            "--resume",
            str(empty),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "no step-* checkpoint" in (result.stdout + result.stderr)
