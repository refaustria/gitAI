"""Append-only, hash-chained record of every model promotion.

Each entry commits to the one before it, so the log is tamper-evident: editing
or deleting any record breaks every hash after it, and :meth:`Lineage.verify`
says exactly where.

This is the concrete form of "never harm itself". A self-improving loop that can
rewrite its own history cannot be audited, and a result you cannot audit is not
a result. The chain costs almost nothing and makes the record trustworthy even
against the loop that wrote it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["GENESIS", "Lineage", "LineageCorrupt", "LineageEntry"]

GENESIS = "0" * 64


class LineageCorrupt(Exception):
    """The hash chain does not verify: the log has been edited or truncated."""


@dataclass(frozen=True)
class LineageEntry:
    index: int
    prev: str
    digest: str
    payload: dict[str, Any]


def _digest(index: int, prev: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(f"{index}|{prev}|".encode() + blob).hexdigest()


class Lineage:
    """A JSONL file, opened only in append mode, never rewritten."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ read

    def entries(self) -> list[LineageEntry]:
        if not self.path.exists():
            return []
        out: list[LineageEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            out.append(
                LineageEntry(
                    index=record["index"],
                    prev=record["prev"],
                    digest=record["digest"],
                    payload=record["payload"],
                )
            )
        return out

    def head(self) -> str:
        entries = self.entries()
        return entries[-1].digest if entries else GENESIS

    def __len__(self) -> int:
        return len(self.entries())

    # ----------------------------------------------------------------- write

    def append(self, payload: dict[str, Any]) -> LineageEntry:
        """Commit one promotion. Never overwrites; the file only grows."""
        entries = self.entries()
        index = len(entries)
        prev = entries[-1].digest if entries else GENESIS
        entry = LineageEntry(index, prev, _digest(index, prev, payload), payload)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:  # append mode only
            fh.write(
                json.dumps(
                    {
                        "index": entry.index,
                        "prev": entry.prev,
                        "digest": entry.digest,
                        "payload": entry.payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return entry

    # ---------------------------------------------------------------- verify

    def verify(self) -> bool:
        """Recompute the chain. Raises :class:`LineageCorrupt` at the first break."""
        prev = GENESIS
        for i, entry in enumerate(self.entries()):
            if entry.index != i:
                raise LineageCorrupt(f"entry {i} claims index {entry.index}: a record was removed")
            if entry.prev != prev:
                raise LineageCorrupt(f"entry {i} does not follow entry {i - 1}: chain broken")
            expected = _digest(entry.index, entry.prev, entry.payload)
            if entry.digest != expected:
                raise LineageCorrupt(f"entry {i} payload was modified after it was written")
            prev = entry.digest
        return True
