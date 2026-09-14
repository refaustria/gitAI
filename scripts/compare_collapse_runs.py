#!/usr/bin/env python3
"""Compare two collapse-experiment result files cell by cell.

    uv run python scripts/compare_collapse_runs.py OLD.jsonl NEW.jsonl

Written for the re-run that followed the seeding fix, but it applies whenever a
result is regenerated: the question "did the conclusions move, or only the
digits?" should be answered by a diff against the noise floor, not by reading
two tables and forming an impression.

A cell is (arm, temperature, top_k, real_fraction) at the final generation,
aggregated over seeds. Differences are reported in units of the measured noise
floor, because that is the only scale on which "the same" means anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent

# R2's measured same-config seed spread; see docs/results.md.
NOISE_FLOOR = 0.0040


def load(path: Path, generation: int | None) -> dict[tuple, list[float]]:
    cells: dict[tuple, list[float]] = defaultdict(list)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if generation is None:
        generation = max(r["generation"] for r in rows)
    for r in rows:
        if r.get("val_bpb") is None or r["generation"] != generation:
            continue
        if r["arm"] == "generation0":
            continue
        cells[(r["arm"], r.get("temperature"), r.get("top_k"), r.get("real_fraction"))].append(
            r["val_bpb"]
        )
    return cells


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--generation", type=int, default=None, help="default: the last")
    args = parser.parse_args()

    old = load(Path(args.old), args.generation)
    new = load(Path(args.new), args.generation)

    only_old = sorted(set(old) - set(new), key=str)
    only_new = sorted(set(new) - set(old), key=str)
    shared = sorted(set(old) & set(new), key=str)

    header = (
        f"{'arm':12} {'T':>5} {'top_k':>6} {'real':>5} "
        f"{'old':>8} {'new':>8} {'diff':>9} {'noise':>7}"
    )
    print(header)
    print("-" * len(header))
    moved = []
    for cell in shared:
        arm, temp, top_k, real = cell
        o, n = mean(old[cell]), mean(new[cell])
        d = n - o
        units = abs(d) / NOISE_FLOOR
        if units >= 1.0:
            moved.append((cell, d, units))
        print(
            f"{arm:12} {temp!s:>5} {top_k!s:>6} {real!s:>5} "
            f"{o:>8.4f} {n:>8.4f} {d:>+9.4f} {units:>6.1f}x"
        )

    for label, cells in (("only in old", only_old), ("only in new", only_new)):
        if cells:
            print(f"\n{label}: {', '.join(str(c) for c in cells)}")

    print(
        f"\n{len(moved)} of {len(shared)} cells moved by at least one noise floor "
        f"({NOISE_FLOOR:.4f} BPB)."
    )
    if moved:
        worst = max(moved, key=lambda m: m[2])
        print(f"largest move: {worst[0]} {worst[1]:+.4f} ({worst[2]:.1f}x noise)")
    print(
        "\nA cell moving is not by itself a problem -- these are different runs. "
        "What matters\nis whether any *comparison* between cells changes sign or "
        "crosses the noise floor,\nwhich is what the conclusions rest on."
    )


if __name__ == "__main__":
    main()
