#!/usr/bin/env python3
"""Rebuild the runs index and print comparison tables.

    uv run python scripts/report.py
    uv run python scripts/report.py --compare noisefloor other

Every number here is a query against ``runs/index.db``, which is itself rebuilt
from the run directories on each invocation. Nothing is transcribed by hand —
that is where errors get into results and become impossible to find.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.eval import (
    build_index,
    compare_named,
    comparison_table,
    leaderboard,
    load_groups,
    noise_floor,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--metric", default="best_bpb")
    parser.add_argument("--noise-like", default="noisefloor%")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()

    db = build_index(args.runs)
    print(f"index: {db}\n")

    print("=" * 78)
    print("all runs")
    print("=" * 78)
    print(leaderboard(db, metric=args.metric))

    floor = noise_floor(db, name_like=args.noise_like, metric=args.metric)
    print()
    print("=" * 78)
    print("seed-aggregated")
    print("=" * 78)
    print(comparison_table(load_groups(db, metric=args.metric), floor=floor))

    print()
    print("=" * 78)
    print("noise floor")
    print("=" * 78)
    if floor is None:
        print(
            "NOT MEASURED. Train one configuration with several seeds before\n"
            "comparing anything — until then no difference in this project is\n"
            "distinguishable from chance."
        )
    else:
        print(f"1 std of one configuration across seeds: {floor:.4f} {args.metric}")
        print(f"significance threshold: differences below ~{floor:.4f} do not exist.")

    if args.compare:
        a, b = args.compare
        print()
        print("=" * 78)
        print(f"{a} vs {b}")
        print("=" * 78)
        print(compare_named(db, a, b, metric=args.metric, floor=floor))


if __name__ == "__main__":
    main()
