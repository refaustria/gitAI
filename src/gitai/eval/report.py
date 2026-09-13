"""Comparison tables, generated from the index — never assembled by hand.

Hand-copying numbers out of logs into a table is where transcription errors get
into results, and they are almost impossible to find afterwards. Every table
here is a query.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .index import connect
from .stats import Comparison, SeedGroup, compare_groups

__all__ = ["compare_named", "comparison_table", "leaderboard", "load_groups", "noise_floor"]


def load_groups(
    db: sqlite3.Connection | Path | str,
    metric: str = "best_bpb",
    group_by: str = "name",
    name_like: str | None = None,
    strip_seed_suffix: bool = True,
) -> dict[str, SeedGroup]:
    """Group runs and collect one metric per run.

    ``strip_seed_suffix`` folds ``noisefloor-s0 … noisefloor-s4`` into one group,
    which is what makes a multi-seed sweep aggregate automatically.
    """
    connection = db if isinstance(db, sqlite3.Connection) else connect(db)
    query = f"SELECT {group_by} AS label, {metric} AS value FROM runs WHERE {metric} IS NOT NULL"
    params: tuple = ()
    if name_like:
        query += " AND name LIKE ?"
        params = (name_like,)

    grouped: dict[str, list[float]] = {}
    for row in connection.execute(query, params):
        label = str(row["label"])
        if strip_seed_suffix and "-s" in label and label.rsplit("-s", 1)[-1].isdigit():
            label = label.rsplit("-s", 1)[0]
        grouped.setdefault(label, []).append(float(row["value"]))

    return {label: SeedGroup(label, tuple(sorted(values))) for label, values in grouped.items()}


def noise_floor(
    db: sqlite3.Connection | Path | str, name_like: str = "noisefloor%", metric: str = "best_bpb"
) -> float | None:
    """The measured spread of one configuration across seeds.

    This is the significance threshold for every later comparison. Returns None
    if fewer than two same-config runs exist — in which case no comparison in
    this project means anything yet, and saying so is better than inventing a
    number.
    """
    groups = load_groups(db, metric=metric, name_like=name_like)
    if not groups:
        return None
    largest = max(groups.values(), key=lambda g: g.n)
    return largest.std if largest.n >= 2 else None


def leaderboard(
    db: sqlite3.Connection | Path | str, metric: str = "best_bpb", limit: int = 25
) -> str:
    connection = db if isinstance(db, sqlite3.Connection) else connect(db)
    rows = connection.execute(
        f"SELECT name, rung, seed, params_total, params_non_embedding, steps, {metric} AS value, "
        f"tokens_per_sec FROM runs WHERE {metric} IS NOT NULL ORDER BY value ASC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return "no runs indexed"

    header = (
        f"{'name':<22} {'rung':<14} {'seed':>4} {'params':>10} {'non-emb':>10} "
        f"{'steps':>6} {metric:>10} {'tok/s':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['name'])[:22]:<22} {str(row['rung'] or '-')[:14]:<14} "
            f"{row['seed'] if row['seed'] is not None else '-':>4} "
            f"{row['params_total'] or 0:>10,} {row['params_non_embedding'] or 0:>10,} "
            f"{row['steps'] or 0:>6} {row['value']:>10.4f} "
            f"{row['tokens_per_sec'] or 0:>9,.0f}"
        )
    return "\n".join(lines)


def comparison_table(groups: dict[str, SeedGroup], floor: float | None = None) -> str:
    """Seed-aggregated results, sorted best first."""
    if not groups:
        return "no groups to compare"

    header = f"{'configuration':<26} {'mean':>9} {'std':>9} {'spread':>9} {'n':>3}   values"
    lines = [header, "-" * (len(header) + 20)]
    for group in sorted(groups.values(), key=lambda g: g.mean):
        values = " ".join(f"{v:.4f}" for v in group.values)
        warn = "  ⚠ single seed" if group.n == 1 else ""
        lines.append(
            f"{group.label[:26]:<26} {group.mean:>9.4f} {group.std:>9.4f} "
            f"{group.spread:>9.4f} {group.n:>3}   {values}{warn}"
        )
    if floor is not None:
        lines += [
            "",
            f"noise floor (1 std of one config across seeds): {floor:.4f}",
            f"differences below ~{floor:.4f} are not distinguishable from chance.",
        ]
    return "\n".join(lines)


def compare_named(
    db: sqlite3.Connection | Path | str,
    a: str,
    b: str,
    metric: str = "best_bpb",
    floor: float | None = None,
) -> Comparison:
    """Compare two named configurations, using the measured noise floor."""
    groups = load_groups(db, metric=metric)
    for label in (a, b):
        if label not in groups:
            raise KeyError(f"no runs named {label!r}; have {sorted(groups)}")
    if floor is None:
        floor = noise_floor(db, metric=metric)
    return compare_groups(groups[a], groups[b], noise_floor=floor)
