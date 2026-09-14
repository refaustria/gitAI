#!/usr/bin/env python3
"""Analyse the collapse experiment against the measured noise floor.

    uv run python scripts/analyse_collapse.py

Reads ``runs/collapse/results.jsonl`` and produces the seed-aggregated table.
Every arm is compared against the control at the same generation, and any
difference below the noise floor is reported as no effect regardless of p-value.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.eval import SeedGroup, compare_groups, noise_floor
from gitai.eval.index import build_index

ROOT = Path(__file__).resolve().parent.parent
ARM_ORDER = ["control", "accumulate", "replace"]


def load(path: Path) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    corpora: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "val_bpb" in row:
            grouped[(row["arm"], row["generation"])].append(row["val_bpb"])
        stats = row.get("own_corpus_stats") or row.get("corpus_stats")
        if stats:
            corpora[(row["arm"], row["generation"])].append(stats)
    return grouped, corpora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "runs/collapse/results.jsonl"))
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--floor", type=float, default=None)
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        sys.exit(f"no results at {path} — run scripts/collapse_experiment.py first")

    grouped, corpora = load(path)

    floor = args.floor
    if floor is None:
        floor = noise_floor(build_index(args.runs), name_like="noisefloor%")
    if floor is None:
        sys.exit(
            "no noise floor measured — run `make noise-floor` first; "
            "without it nothing here means anything"
        )
    print(f"noise floor: {floor:.4f} BPB (differences below this are not effects)\n")

    generations = sorted({g for _, g in grouped if g > 0})

    # ---------------------------------------------------------------- table
    header = f"{'gen':>4}" + "".join(f"{arm:>26}" for arm in ARM_ORDER)
    print("=" * len(header))
    print("held-out BPB on REAL validation data   (mean ± std across seeds)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for generation in generations:
        cells = []
        for arm in ARM_ORDER:
            values = grouped.get((arm, generation), [])
            if not values:
                cells.append(f"{'—':>26}")
                continue
            group = SeedGroup(arm, tuple(values))
            cells.append(f"{group.mean:>16.4f} ±{group.std:.4f} ")
        print(f"{generation:>4}" + "".join(cells))

    # ------------------------------------------------- vs control, per generation
    print()
    print("=" * 78)
    print("each arm vs control, at the same generation")
    print("=" * 78)
    for generation in generations:
        control = grouped.get(("control", generation), [])
        if not control:
            continue
        for arm in ARM_ORDER:
            if arm == "control":
                continue
            values = grouped.get((arm, generation), [])
            if not values:
                continue
            comparison = compare_groups(
                SeedGroup("control", tuple(control)),
                SeedGroup(arm, tuple(values)),
                noise_floor=floor,
            )
            print(
                f"  gen {generation}  {arm:<11} {comparison.difference:+.4f} BPB  "
                f"({comparison.effect_in_noise_units:>5.1f}x noise)  {comparison.verdict}"
            )

    # ------------------------------------------------------------- drift
    print()
    print("=" * 78)
    print("drift from generation 1 to the last")
    print("=" * 78)
    last = generations[-1]
    for arm in ARM_ORDER:
        first_vals = grouped.get((arm, generations[0]), [])
        last_vals = grouped.get((arm, last), [])
        if not first_vals or not last_vals:
            continue
        comparison = compare_groups(
            SeedGroup(f"{arm}@g{generations[0]}", tuple(first_vals)),
            SeedGroup(f"{arm}@g{last}", tuple(last_vals)),
            noise_floor=floor,
        )
        print(
            f"  {arm:<11} {comparison.difference:+.4f} BPB over {last - generations[0] + 1} "
            f"generations  ({comparison.effect_in_noise_units:>5.1f}x noise)  {comparison.verdict}"
        )

    # ---------------------------------------------------------- diversity
    if corpora:
        print()
        print("=" * 78)
        print("generated-corpus diversity (does the distribution narrow?)")
        print("=" * 78)
        print(
            f"{'arm':<14}{'gen':>4}{'distinct_1':>12}{'distinct_2':>12}"
            f"{'distinct_3':>12}{'vocab':>9}"
        )
        print("-" * 78)
        for (arm, generation), entries in sorted(corpora.items()):
            means = {
                key: sum(entry[key] for entry in entries) / len(entries)
                for key in ("distinct_1", "distinct_2", "distinct_3", "vocabulary_fraction")
            }
            print(
                f"{arm:<14}{generation:>4}{means['distinct_1']:>12.4f}"
                f"{means['distinct_2']:>12.4f}{means['distinct_3']:>12.4f}"
                f"{means['vocabulary_fraction']:>8.1%}"
            )


if __name__ == "__main__":
    main()
