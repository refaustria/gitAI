#!/usr/bin/env python3
"""The morning digest for a loop run.

    uv run python scripts/report_loop.py --workspace runs/loop-01

Answers, in the order docs/runbook.md says to read them: did it stop for the
expected reason, does the lineage verify, what is the promotion rate, and where
are the rejections clustering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.safety import Lineage, LineageCorrupt

ROOT = Path(__file__).resolve().parent.parent
NOISE_FLOOR = 0.0040  # results.md R5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(ROOT / "runs/loop"))
    args = parser.parse_args()

    workspace = Path(args.workspace)
    result_path = workspace / "result.json"
    if not result_path.exists():
        sys.exit(f"no result.json in {workspace} — has the loop run?")
    payload = json.loads(result_path.read_text())
    iterations = payload["iterations"]

    print("=" * 72)
    print(f"loop report — {workspace}")
    print("=" * 72)
    print(f"stopped because : {payload['stopped_because']}")
    print(f"wall clock      : {payload['seconds'] / 60:.1f} min")
    print(f"iterations      : {len(iterations)}")
    print(f"promotions      : {payload['promotions']}")
    print(f"best held-out   : {payload['best_bpb']:.4f} BPB")

    # ---- lineage integrity comes before any interpretation of the numbers
    print()
    lineage = Lineage(workspace / "lineage.jsonl")
    try:
        lineage.verify()
        print(f"lineage         : VERIFIED ({len(lineage)} entries)")
    except LineageCorrupt as corrupt:
        print(f"lineage         : ***CORRUPT*** {corrupt}")
        print("\nStop here. Nothing after the break is trustworthy.")
        return

    if not iterations:
        print("\nNo iterations ran.")
        return

    rate = payload["promotions"] / len(iterations)
    print(f"promotion rate  : {rate:.0%}")
    if rate > 0.9:
        print("  ^ suspiciously high — is the gate looser than the noise floor?")
    elif rate == 0:
        print("  ^ zero. May be correct: refusing to promote noise is the gate working.")

    print()
    print("=" * 72)
    print("iterations")
    print("=" * 72)
    print(f"{'iter':>5} {'BPB':>9} {'synth':>7} {'temp':>6} {'lr':>9}  outcome")
    print("-" * 72)
    for row in iterations:
        proposal = row["proposal"]
        mark = "PROMOTED" if row["promoted"] else "rejected"
        print(
            f"{row['iteration']:>5} {row['val_bpb']:>9.4f} "
            f"{proposal['synthetic_fraction']:>7.0%} {proposal['temperature']:>6.2f} "
            f"{proposal['lr']:>9.1e}  {mark}"
        )

    rejections = [r for row in iterations if not row["promoted"] for r in row["reasons"]]
    if rejections:
        print()
        print("=" * 72)
        print("why candidates were rejected — the research data")
        print("=" * 72)
        for reason, count in Counter(r.split(":")[0] for r in rejections).most_common():
            print(f"  {reason:<28} {count:>3}")

        print()
        print("rejections by synthetic fraction:")
        by_fraction: Counter = Counter()
        total: Counter = Counter()
        for row in iterations:
            key = f"{row['proposal']['synthetic_fraction']:.0%}"
            total[key] += 1
            if not row["promoted"]:
                by_fraction[key] += 1
        for key in sorted(total):
            print(f"  {key:>5}: {by_fraction[key]}/{total[key]} rejected")
        print("\n  Clustering at one corner of the space is telling you something")
        print("  about that corner — that is the output, not a nuisance.")

    print()
    print(f"(noise floor {NOISE_FLOOR} BPB — differences below it are not effects)")


if __name__ == "__main__":
    main()
