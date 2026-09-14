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


def regime(row: dict) -> str:
    """Label a row by its sampling regime.

    Rows written before the temperature sweep carry no temperature field; they
    were all produced at 1.0, so that is the default rather than a guess.
    """
    temperature = row.get("temperature", 1.0)
    top_k = row.get("top_k")
    return f"T{temperature:g}" + (f",k{top_k}" if top_k else "")


def load(path: Path):
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    corpora: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    losses: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["arm"], regime(row), row["generation"])
        if "val_bpb" in row:
            grouped[key].append(row["val_bpb"])
        if "train_loss" in row:
            losses[key].append(row["train_loss"])
        stats = row.get("own_corpus_stats") or row.get("corpus_stats")
        if stats:
            corpora[key].append(stats)
    return grouped, corpora, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "runs/collapse/results.jsonl"))
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--floor", type=float, default=None)
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        sys.exit(f"no results at {path} — run scripts/collapse_experiment.py first")

    grouped, corpora, losses = load(path)

    floor = args.floor
    if floor is None:
        floor = noise_floor(build_index(args.runs), name_like="noisefloor%")
    if floor is None:
        sys.exit(
            "no noise floor measured — run `make noise-floor` first; "
            "without it nothing here means anything"
        )
    print(f"noise floor: {floor:.4f} BPB (differences below this are not effects)\n")

    regimes = sorted({r for _, r, _ in grouped})
    base = "T1" if "T1" in regimes else regimes[0]
    generations = sorted({g for _, r, g in grouped if g > 0 and r == base})

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
            values = grouped.get((arm, base, generation), [])
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
        control = grouped.get(("control", base, generation), [])
        if not control:
            continue
        for arm in ARM_ORDER:
            if arm == "control":
                continue
            values = grouped.get((arm, base, generation), [])
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
        first_vals = grouped.get((arm, base, generations[0]), [])
        last_vals = grouped.get((arm, base, last), [])
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
        for (arm, _r, generation), entries in sorted(corpora.items()):
            if _r != base:
                continue
            means = {
                key: sum(entry[key] for entry in entries) / len(entries)
                for key in ("distinct_1", "distinct_2", "distinct_3", "vocabulary_fraction")
            }
            print(
                f"{arm:<14}{generation:>4}{means['distinct_1']:>12.4f}"
                f"{means['distinct_2']:>12.4f}{means['distinct_3']:>12.4f}"
                f"{means['vocabulary_fraction']:>8.1%}"
            )

    # ------------------------------------------------------- temperature sweep
    sweep_regimes = sorted(
        {r for (arm, r, _g) in grouped if arm == "replace"},
        key=lambda r: (float(r.split(",")[0][1:]), r),
    )
    if len(sweep_regimes) > 1:
        print()
        print("=" * 78)
        print("temperature sweep — does the sampling regime select the failure mode?")
        print("=" * 78)

        header = (
            f"{'regime':<10}" + "".join(f"{f'gen {g}':>12}" for g in generations) + f"{'drift':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in sweep_regimes:
            cells, first, last_val = [], None, None
            for generation in generations:
                values = grouped.get(("replace", r, generation), [])
                if not values:
                    cells.append(f"{'—':>12}")
                    continue
                mean = sum(values) / len(values)
                cells.append(f"{mean:>12.4f}")
                first = mean if first is None else first
                last_val = mean
            drift = (
                f"{last_val - first:+.4f}" if first is not None and last_val is not None else "—"
            )
            print(f"{r:<10}" + "".join(cells) + f"{drift:>10}")

        print()
        print(
            f"{'regime':<10}{'gen':>4}{'distinct_2':>12}"
            f"{'distinct_3':>12}{'vocab':>9}{'train loss':>12}"
        )
        print("-" * 78)
        for r in sweep_regimes:
            for generation in sorted({g for (a, rr, g) in corpora if a == "replace" and rr == r}):
                entries = corpora[("replace", r, generation)]
                means = {
                    key: sum(e[key] for e in entries) / len(entries)
                    for key in ("distinct_2", "distinct_3", "vocabulary_fraction")
                }
                loss_vals = losses.get(("replace", r, generation), [])
                loss = sum(loss_vals) / len(loss_vals) if loss_vals else float("nan")
                print(
                    f"{r:<10}{generation:>4}{means['distinct_2']:>12.4f}"
                    f"{means['distinct_3']:>12.4f}{means['vocabulary_fraction']:>8.1%}"
                    f"{loss:>12.4f}"
                )


if __name__ == "__main__":
    main()
