"""``runs/`` -> SQLite.

A *derived index*, never the source of truth. The run directories hold the real
record; this file can be deleted and rebuilt at any time, and that property is
what keeps the data safe from a bug in here.

Once there are 200 runs, "every run with d_model=384, grouped by depth, mean and
spread across seeds" is a query, and grepping JSON is not the way to answer it.
See Decision 7.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

__all__ = ["SCHEMA", "build_index", "connect"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    name                TEXT,
    path                TEXT,
    started_at          TEXT,
    git_sha             TEXT,
    git_dirty           INTEGER,
    rung                TEXT,
    seed                INTEGER,
    steps               INTEGER,
    batch_size          INTEGER,
    seq_len             INTEGER,
    d_model             INTEGER,
    n_layer             INTEGER,
    n_head             INTEGER,
    vocab_size          INTEGER,
    lr                  REAL,
    params_total        INTEGER,
    params_non_embedding INTEGER,
    tokenizer           TEXT,
    best_step           INTEGER,
    best_bpb            REAL,
    best_loss           REAL,
    final_train_loss    REAL,
    tokens_per_sec      REAL,
    wall_clock          REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT,
    step    INTEGER,
    key     TEXT,
    value   REAL
);
CREATE INDEX IF NOT EXISTS metrics_run_key ON metrics (run_id, key);
CREATE INDEX IF NOT EXISTS runs_name ON runs (name);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_index(runs_dir: Path | str, db_path: Path | str | None = None) -> Path:
    """Rebuild the index from scratch. Safe to run any time; drops and re-reads."""
    runs_dir = Path(runs_dir)
    db_path = Path(db_path) if db_path else runs_dir / "index.db"
    runs_dir.mkdir(parents=True, exist_ok=True)

    connection = connect(db_path)
    with connection:
        connection.executescript("DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS metrics;")
        connection.executescript(SCHEMA)

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config = _read_json(run_dir / "config.yaml")
        manifest = _read_json(run_dir / "manifest.json")
        best = _read_json(run_dir / "checkpoints" / "best.json")
        if not config and not manifest:
            continue  # not a run directory

        training = config.get("training", {})
        model = config.get("model", {})
        parameters = manifest.get("parameters", {})

        metrics: list[tuple] = []
        final_train_loss = None
        tokens_per_sec = None
        wall_clock = None
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                step = record.get("step")
                for key, value in record.items():
                    if key != "step" and isinstance(value, int | float):
                        metrics.append((run_dir.name, step, key, float(value)))
                if "loss" in record:
                    final_train_loss = record["loss"]
                    tokens_per_sec = record.get("tokens_per_sec", tokens_per_sec)
                    wall_clock = record.get("elapsed", wall_clock)

        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (:run_id,:name,:path,:started_at,:git_sha,"
                ":git_dirty,:rung,:seed,:steps,:batch_size,:seq_len,:d_model,:n_layer,:n_head,"
                ":vocab_size,:lr,:params_total,:params_non_embedding,:tokenizer,:best_step,"
                ":best_bpb,:best_loss,:final_train_loss,:tokens_per_sec,:wall_clock)",
                {
                    "run_id": run_dir.name,
                    "name": training.get("name") or run_dir.name.split("-", 2)[-1],
                    "path": str(run_dir),
                    "started_at": manifest.get("started_at"),
                    "git_sha": manifest.get("git", {}).get("sha"),
                    "git_dirty": int(bool(manifest.get("git", {}).get("dirty"))),
                    "rung": training.get("rung") or best.get("rung"),
                    "seed": training.get("seed"),
                    "steps": training.get("steps"),
                    "batch_size": training.get("batch_size"),
                    "seq_len": model.get("seq_len") or training.get("seq_len"),
                    "d_model": model.get("d_model"),
                    "n_layer": model.get("n_layer"),
                    "n_head": model.get("n_head"),
                    "vocab_size": model.get("vocab_size"),
                    "lr": training.get("lr"),
                    "params_total": parameters.get("total"),
                    "params_non_embedding": parameters.get("non_embedding"),
                    "tokenizer": manifest.get("tokenizer_fingerprint"),
                    "best_step": best.get("step"),
                    "best_bpb": best.get("val_bpb"),
                    "best_loss": best.get("val_loss"),
                    "final_train_loss": final_train_loss,
                    "tokens_per_sec": tokens_per_sec,
                    "wall_clock": wall_clock,
                },
            )
            if metrics:
                connection.executemany("INSERT INTO metrics VALUES (?,?,?,?)", metrics)

    connection.close()
    return db_path
