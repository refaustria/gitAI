#!/usr/bin/env python3
"""Run the bounded self-improvement loop.

    uv run python scripts/run_loop.py --iterations 10 --workspace runs/loop-01

Operating instructions, including how to stop it, are in docs/runbook.md.
Stop it with:  python scripts/halt.py stop "reason"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import BatchSampler, ShardIndex, read_documents
from gitai.loop import ImprovementLoop, LoopConfig, SearchSpace
from gitai.model import RUNGS, ModelConfig, Transformer
from gitai.safety import Budget, HaltSwitch, Lineage, PathGuard, default_suite
from gitai.tokenizer import ByteBPETokenizer
from gitai.training import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
EOT = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument("--parquet", default=str(ROOT / "data/interim/tinyshakespeare.parquet"))
    parser.add_argument("--workspace", default=str(ROOT / "runs/loop"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--seed-steps",
        type=int,
        default=150,
        help="steps for the starting incumbent; deliberately under-trained "
        "so the loop has headroom to find something",
    )
    parser.add_argument("--steps", type=int, default=200, help="steps per candidate")
    parser.add_argument("--seeds-per-candidate", type=int, default=1)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="NoRegression tolerance; 2.5x the measured noise floor of 0.0040",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--max-disk-gib", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Before anything constructs a module. scripts/train.py seeds here too, but
    # this script did not, and `train()` seeds only once it is already inside
    # itself -- by which point the incumbent's weights have been drawn from an
    # unseeded generator. Every candidate is a clone of that incumbent, so a
    # single missing line made the entire lineage unreproducible: same seed,
    # same proposals, different measurements, and at least one promotion
    # decision flipped by the difference. See docs/evaluation.md.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    index = ShardIndex.load(data_dir)
    tokenizer = ByteBPETokenizer.load(data_dir / "tokenizer.json")
    real_documents = read_documents(args.parquet, split="train")
    val_sampler = BatchSampler(data_dir, "val", 128, index=index)

    print("=" * 72)
    print("seeding the incumbent")
    print("=" * 72)
    model = Transformer(
        ModelConfig(
            vocab_size=index.vocab_size,
            seq_len=128,
            d_model=128,
            n_layer=4,
            n_head=4,
            **RUNGS["v6_modern"],
        )
    )
    train(
        model,
        BatchSampler(data_dir, "train", 128, index=index),
        val_sampler,
        TrainConfig(
            steps=args.seed_steps,
            batch_size=16,
            lr=3e-3,
            eval_every=args.seed_steps,
            log_every=10**9,
            seed=args.seed,
        ),
    )

    space = SearchSpace(
        synthetic_fraction=(0.0, 0.25, 0.5),
        temperature=(0.9, 1.0, 1.1),
        lr=(1e-3, 3e-3),
        steps=(args.steps,),
    )
    guard = PathGuard([workspace])
    lineage = Lineage(workspace / "lineage.jsonl")

    loop = ImprovementLoop(
        workspace=workspace,
        incumbent=model,
        tokenizer=tokenizer,
        real_documents=real_documents,
        val_sampler=val_sampler,
        space=space,
        suite=default_suite(lineage, guard, tolerance=args.tolerance),
        lineage=lineage,
        guard=guard,
        # Outside every root the loop may write to, so it can see the signal and
        # has no sanctioned way to remove it.
        halt=HaltSwitch(ROOT / "control" / "HALT"),
        budget=Budget(
            max_iterations=args.iterations + 1,
            max_wall_seconds=args.max_wall_seconds,
            max_disk_bytes=int(args.max_disk_gib * 1024**3),
        ),
        config=LoopConfig(
            iterations=args.iterations,
            seeds_per_candidate=args.seeds_per_candidate,
            tolerance=args.tolerance,
        ),
        separator_id=tokenizer.special_tokens[EOT],
        seed=args.seed,
    )

    print()
    print(space.describe())
    print(
        f"\nincumbent starts at {loop.incumbent_bpb if loop.incumbent_bpb < 1e9 else 'unmeasured'}"
    )
    print('stop with: python scripts/halt.py stop "reason"\n')
    print("=" * 72)

    result = loop.run(on_iteration=lambda o: print(o))

    print()
    print("=" * 72)
    print(result.summary())
    print(f"workspace: {workspace}")


if __name__ == "__main__":
    main()
