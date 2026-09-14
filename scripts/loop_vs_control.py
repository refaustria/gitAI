#!/usr/bin/env python3
"""Compare the improvement loop against two compute-matched controls, on a split
neither of them selected on.

    uv run python scripts/loop_vs_control.py

The loop gates promotion on ``val_bpb`` and its summary then reports the best
``val_bpb`` it saw, labelled "held-out". That label is only half true: the split
is held out from *training* but not from *selection*, and a minimum taken over
promotions is biased downward by exactly the amount the selection was worth. The
controls have the same problem from the other side -- ``train.py`` keeps the best
checkpoint over ~30 evaluations, so it gets more draws at the minimum than the
loop's ~11 promotions. Comparing those two numbers measures the difference in
selection pressure as much as the difference in method.

So this script ignores both reported numbers and re-evaluates every arm's chosen
checkpoint on ``test``, which nothing in either pipeline has ever looked at. That
is the only number here that estimates generalisation.

Arms, per seed:
  loop      the promoted checkpoint with the lowest val BPB
  acc       control trained for the loop's *accumulated* steps (promoted only)
  tot       control trained for the loop's *total spent* steps (including
            the work thrown away on rejected candidates)

The two controls bracket the question "was the search worth it?". Beating `acc`
means the selection added something beyond the training it kept; beating `tot`
means it added something beyond the compute it burned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from safetensors.torch import load_model

from gitai.data import BatchSampler, ShardIndex
from gitai.eval.stats import SeedGroup, compare_groups
from gitai.model import ModelConfig, Transformer
from gitai.training import evaluate_bpb

ROOT = Path(__file__).resolve().parent.parent

# Measured across same-config seeds in R2; see docs/results.md. Using these
# groups' own spread to judge themselves would be circular at three seeds.
NOISE_FLOOR = 0.0040


def loop_best_checkpoint(workspace: Path) -> tuple[Path, float, str]:
    """The promoted checkpoint the loop would hand you, and why it stopped."""
    result = json.loads((workspace / "result.json").read_text())
    promoted = [o for o in result["iterations"] if o["promoted"]]
    if not promoted:
        raise SystemExit(f"{workspace} promoted nothing; there is no model to evaluate")
    best = min(promoted, key=lambda o: o["val_bpb"])
    path = workspace / "models" / f"iter-{best['iteration']:04d}.safetensors"
    return path, best["val_bpb"], result["stopped_because"]


def load_transformer(checkpoint: Path, spec: dict) -> Transformer:
    model = Transformer(ModelConfig(**spec))
    load_model(model, str(checkpoint))
    model.eval()
    return model


def spent_steps(workspace: Path, seed_steps: int) -> tuple[int, int]:
    """What the loop actually cost, read from its own record.

    Both numbers must come from ``result.json`` rather than from
    ``iterations * steps``. The original run assumed the latter and matched a
    truncated seed's control to compute the loop never spent.
    """
    result = json.loads((workspace / "result.json").read_text())
    accumulated = sum(o["proposal"]["steps"] for o in result["iterations"] if o["promoted"])
    total = sum(o["proposal"]["steps"] for o in result["iterations"])
    return accumulated + seed_steps, total + seed_steps


def drive(seed: int, args) -> None:
    """Run the loop, then both controls matched to what the loop actually spent."""
    root = ROOT
    workspace = root / "runs" / f"lvc-loop-s{seed}"
    if workspace.exists():
        raise SystemExit(f"{workspace} exists; move it aside rather than mixing runs")

    print(f"\n=== seed {seed}: loop ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_loop.py"),
            "--workspace",
            str(workspace),
            "--iterations",
            str(args.iterations),
            "--steps",
            str(args.steps),
            "--seed-steps",
            str(args.seed_steps),
            "--max-wall-seconds",
            str(args.max_wall_seconds),
            "--seed",
            str(seed),
        ],
        check=True,
    )

    accumulated, total = spent_steps(workspace, args.seed_steps)
    print(f"\nseed {seed}: accumulated={accumulated} total_spent={total}", flush=True)

    for arm, steps in (("acc", accumulated), ("tot", total)):
        print(f"\n=== seed {seed}: control {arm} at {steps} steps ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "train.py"),
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--name",
                f"lvc-{arm}-s{seed}",
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the loop and both controls before reporting, instead of "
        "reading runs already on disk",
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--steps", type=int, default=200, help="steps per candidate")
    parser.add_argument("--seed-steps", type=int, default=150)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=36000.0,
        help="deliberately far above what an iteration budget needs. A wall "
        "budget that binds turns machine load into an experimental variable; "
        "it is here as a backstop, not as the terminator.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data)
    index = ShardIndex.load(data_dir)

    if args.run:
        for seed in args.seeds:
            drive(seed, args)

    arms: dict[str, list[float]] = {"loop": [], "acc": [], "tot": []}
    truncated: list[int] = []
    rows: list[tuple] = []

    for seed in args.seeds:
        workspace = ROOT / "runs" / f"lvc-loop-s{seed}"
        loop_ckpt, loop_val, stopped = loop_best_checkpoint(workspace)
        if "budget exhausted" in stopped:
            truncated.append(seed)

        controls = {}
        for arm in ("acc", "tot"):
            matches = sorted((ROOT / "runs").glob(f"*-lvc-{arm}-s{seed}"))
            if not matches:
                raise SystemExit(f"no control run found for arm {arm!r}, seed {seed}")
            controls[arm] = matches[-1]

        spec = json.loads((controls["tot"] / "checkpoints" / "best.json").read_text())["model"]
        sampler = BatchSampler(data_dir, args.split, spec["seq_len"], index=index)

        def test_bpb(model, sampler=sampler) -> float:
            return evaluate_bpb(model, sampler, args.batch_size, args.eval_batches)["val_bpb"]

        with torch.no_grad():
            loop_test = test_bpb(load_transformer(loop_ckpt, spec))
            arms["loop"].append(loop_test)
            control_tests = {}
            for arm, run_dir in controls.items():
                ckpt = run_dir / "checkpoints" / "best.safetensors"
                value = test_bpb(load_transformer(ckpt, spec))
                arms[arm].append(value)
                control_tests[arm] = value

        rows.append(
            (seed, stopped, loop_val, loop_test, control_tests["acc"], control_tests["tot"])
        )

    print(f"\nEvaluated on the {args.split!r} split, which neither arm selected on.\n")
    header = (
        f"{'seed':>4}  {'loop val':>9}  {'loop test':>9}  {'acc test':>9}  {'tot test':>9}  stopped"
    )
    print(header)
    print("-" * len(header))
    for seed, stopped, loop_val, loop_test, acc, tot in rows:
        flag = "  <-- TRUNCATED" if "budget exhausted" in stopped else ""
        print(
            f"{seed:>4}  {loop_val:>9.4f}  {loop_test:>9.4f}  "
            f"{acc:>9.4f}  {tot:>9.4f}  {stopped}{flag}"
        )

    if truncated:
        print(
            f"\nSeeds {truncated} stopped on the wall-clock budget rather than on their "
            "iteration count.\nThose arms ran a different procedure -- how busy the machine "
            "was became an\nexperimental variable -- so they are reported but excluded from "
            "the pooled\ncomparison below. See docs/evaluation.md."
        )

    gaps = [row[3] - row[2] for row in rows]
    print(
        f"\nSelection bias: the loop's reported val BPB is optimistic by "
        f"{sum(gaps) / len(gaps):+.4f} on average\n(per seed: "
        + ", ".join(f"{g:+.4f}" for g in gaps)
        + f"), against a noise floor of {NOISE_FLOOR:.4f}. That gap is what\n"
        "selecting the minimum over promotions on the gating split buys you, and it is\n"
        "not generalisation."
    )

    clean = [i for i, s in enumerate(args.seeds) if s not in truncated]
    if len(clean) < 2:
        print("\nToo few untruncated seeds to pool. Re-run the truncated ones first.")
        return

    groups = {k: SeedGroup(k, tuple(v[i] for i in clean)) for k, v in arms.items()}
    print(f"\nPooled over {len(clean)} untruncated seed(s); noise floor {NOISE_FLOOR:.4f} BPB.\n")
    for control in ("acc", "tot"):
        print(compare_groups(groups[control], groups["loop"], noise_floor=NOISE_FLOOR))
        print()


if __name__ == "__main__":
    main()
