"""Corpus curation: Parquet on disk, DuckDB for the questions.

This is the stage where a database earns its place — and the reasoning for
choosing this one over Pandas or Postgres is in docs/data-and-storage.md. Before
tokenizing you need to answer analytical questions ("how much of this is
near-duplicate?", "what does the length distribution look like?", "what survives
filter X?") over more data than fits in RAM. That is DuckDB's exact purpose:
SQL, out-of-core, zero server, reading Parquet directly.

Data curation is the highest-leverage work in this project. The TinyStories
result is a *data* result, not an architecture result — at this scale a better
corpus beats a better model essentially every time. Treat this file as more
important than the transformer.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "CurationReport",
    "QualityFilters",
    "StageReport",
    "assign_split",
    "corpus_stats",
    "curate",
    "minhash_signatures",
    "near_duplicate_groups",
    "read_documents",
    "write_parquet",
]

SPLITS = ("train", "val", "test")
_MERSENNE_61 = (1 << 61) - 1


# --------------------------------------------------------------------- splits


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_split(text: str, val_permille: int = 10, test_permille: int = 10) -> str:
    """Assign a document to train/val/test by hashing its *content*.

    Deliberately not random, and deliberately not by line. Hashing the content
    means an identical document always lands in the same split, so duplicates
    cannot straddle the boundary and leak the validation set into training. A
    random split re-run after adding data also reshuffles every existing
    document, which silently invalidates every earlier result.
    """
    bucket = int(document_hash(text)[:8], 16) % 1000
    if bucket < val_permille:
        return "val"
    if bucket < val_permille + test_permille:
        return "test"
    return "train"


# --------------------------------------------------------------------- filters


@dataclass
class QualityFilters:
    """Filters applied in order. Every rejection is counted and reported.

    Defaults are deliberately gentle: aggressive filtering on a small corpus
    removes more signal than noise, and you cannot tell which without measuring.
    Tighten these as an experiment, with the results written down.
    """

    min_chars: int = 16
    max_chars: int | None = None
    min_words: int = 3
    max_line_repeat_ratio: float = 0.5
    max_non_alpha_ratio: float = 0.6
    require_printable: bool = True

    def check(self, text: str) -> str | None:
        """Return the name of the first failing filter, or None if the doc passes."""
        if len(text) < self.min_chars:
            return "too_short"
        if self.max_chars is not None and len(text) > self.max_chars:
            return "too_long"

        words = text.split()
        if len(words) < self.min_words:
            return "too_few_words"

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and len(lines) > 2:
            most_common = max(_counts(lines).values())
            if most_common / len(lines) > self.max_line_repeat_ratio:
                return "repetitive_lines"

        alpha = sum(1 for c in text if c.isalpha() or c.isspace())
        if alpha / len(text) < (1.0 - self.max_non_alpha_ratio):
            return "mostly_non_alphabetic"

        # Cc = control characters. Tab/newline/carriage-return are fine; the rest
        # usually indicate a binary file or a decoding accident.
        if self.require_printable and any(
            unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in text
        ):
            return "control_characters"
        return None


def _counts(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


# --------------------------------------------------------------------- reports


@dataclass
class StageReport:
    stage: str
    removed: int
    remaining: int


@dataclass
class CurationReport:
    input_documents: int = 0
    stages: list[StageReport] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    output_documents: int = 0
    output_bytes: int = 0

    def record(self, stage: str, removed: int, remaining: int) -> None:
        self.stages.append(StageReport(stage, removed, remaining))

    def __str__(self) -> str:
        lines = [f"input: {self.input_documents:,} documents", ""]
        for stage in self.stages:
            pct = 100.0 * stage.removed / self.input_documents if self.input_documents else 0.0
            lines.append(
                f"  {stage.stage:<22} -{stage.removed:>9,} ({pct:5.2f}%)  -> {stage.remaining:>9,}"
            )
        if self.rejections:
            lines += ["", "  rejections by filter:"]
            for name, count in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name:<24} {count:>9,}")
        lines += ["", f"output: {self.output_documents:,} documents, {self.output_bytes:,} bytes"]
        if self.split_counts:
            counts = "  ".join(f"{k}={v:,}" for k, v in sorted(self.split_counts.items()))
            lines.append("  " + counts)
        return "\n".join(lines)


# ---------------------------------------------------------------- parquet i/o


def write_parquet(documents: Iterable[str], path: Path | str, source: str = "unknown") -> int:
    """Write documents to Parquet — columnar, compressed, typed, universally readable.

    Not JSONL (bloated, untyped) and not CSV (no types, quoting hell).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    texts = list(documents)
    table = pa.table(
        {
            "id": pa.array([document_hash(t)[:16] for t in texts], pa.string()),
            "text": pa.array(texts, pa.string()),
            "source": pa.array([source] * len(texts), pa.string()),
            "chars": pa.array([len(t) for t in texts], pa.int64()),
            "bytes": pa.array([len(t.encode("utf-8")) for t in texts], pa.int64()),
            "words": pa.array([len(t.split()) for t in texts], pa.int64()),
            "split": pa.array([assign_split(t) for t in texts], pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd")
    return len(texts)


def read_documents(path: Path | str, split: str | None = None) -> list[str]:
    import duckdb

    query = f"SELECT text FROM read_parquet('{Path(path).as_posix()}')"
    if split:
        query += f" WHERE split = '{split}'"
    query += " ORDER BY id"  # deterministic order, independent of file layout
    with duckdb.connect() as con:
        return [row[0] for row in con.execute(query).fetchall()]


def corpus_stats(path: Path | str) -> dict:
    """The questions worth asking before tokenizing anything."""
    import duckdb

    source = Path(path).as_posix()
    with duckdb.connect() as con:
        row = con.execute(f"""
            SELECT count(*), sum(bytes), sum(words),
                   min(chars), max(chars),
                   avg(chars), median(chars),
                   quantile_cont(chars, 0.95),
                   count(DISTINCT id)
            FROM read_parquet('{source}')
        """).fetchone()
        splits = dict(
            con.execute(
                f"SELECT split, count(*) FROM read_parquet('{source}') GROUP BY split"
            ).fetchall()
        )

    total, byte_total, words, lo, hi, mean, median, p95, distinct = row
    return {
        "documents": total,
        "distinct_documents": distinct,
        "exact_duplicate_documents": total - distinct,
        "bytes": byte_total,
        "words": words,
        "chars_min": lo,
        "chars_max": hi,
        "chars_mean": round(mean, 1) if mean else 0.0,
        "chars_median": median,
        "chars_p95": p95,
        "splits": splits,
    }


# --------------------------------------------------------------------- minhash


def _shingle_hashes(text: str, ngram: int) -> np.ndarray:
    """64-bit hashes of overlapping character n-grams."""
    data = text.encode("utf-8")
    if len(data) <= ngram:
        shingles = [data]
    else:
        shingles = [data[i : i + ngram] for i in range(len(data) - ngram + 1)]
    unique = {hashlib.blake2b(s, digest_size=8).digest() for s in shingles}
    return np.array([int.from_bytes(h, "big") for h in unique], dtype=np.uint64)


def minhash_signatures(
    texts: list[str], num_perm: int = 64, ngram: int = 8, seed: int = 0
) -> np.ndarray:
    """MinHash signatures: an (n, num_perm) matrix approximating Jaccard similarity.

    Two documents' signatures agree in roughly the same fraction of positions as
    their shingle sets overlap, so a 64-number signature stands in for a set of
    thousands. That is what makes near-duplicate detection affordable.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE_61, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE_61, size=num_perm, dtype=np.uint64)

    signatures = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, text in enumerate(texts):
        hashes = _shingle_hashes(text, ngram)
        if hashes.size == 0:
            continue
        # Universal hashing: (a*h + b) mod prime, one permutation per row.
        permuted = (a[:, None] * (hashes[None, :] % _MERSENNE_61) + b[:, None]) % _MERSENNE_61
        signatures[i] = permuted.min(axis=1)
    return signatures


def near_duplicate_groups(
    signatures: np.ndarray, threshold: float = 0.8, bands: int = 16
) -> list[set[int]]:
    """Group near-duplicates using banded LSH, then verify each candidate pair.

    Comparing all pairs is O(n^2) and hopeless past a few thousand documents.
    Banding hashes slices of each signature into buckets; documents that collide
    in any band become candidates, and only those are compared properly. The
    estimated similarity check is what keeps false positives out.
    """
    n, num_perm = signatures.shape
    if n < 2:
        return []
    if num_perm % bands:
        raise ValueError(f"num_perm {num_perm} must divide evenly into {bands} bands")
    rows = num_perm // bands

    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = signatures[:, band * rows : (band + 1) * rows]
        for i in range(n):
            buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) > 1:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))

    # Union-find over verified pairs.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked = False
    for i, j in candidates:
        similarity = float((signatures[i] == signatures[j]).mean())
        if similarity >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
                linked = True

    if not linked:
        return []
    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return [g for g in groups.values() if len(g) > 1]


# -------------------------------------------------------------------- pipeline


def curate(
    documents: Iterable[str],
    filters: QualityFilters | None = None,
    near_duplicate_threshold: float | None = 0.85,
    ngram: int = 8,
    seed: int = 0,
) -> tuple[list[str], CurationReport]:
    """Exact dedup -> quality filters -> near-duplicate removal.

    Order matters: exact dedup first because it is cheap and removes the bulk,
    filters next because they are cheaper than MinHash, near-duplicate removal
    last on the smallest surviving set.

    Every stage's removals are counted. A pipeline that silently drops 80% of a
    corpus is a bug you want to find now, not after a three-day training run.
    """
    filters = filters or QualityFilters()
    report = CurationReport()

    docs = list(documents)
    report.input_documents = len(docs)

    seen: set[str] = set()
    deduped: list[str] = []
    for doc in docs:
        key = document_hash(doc)
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    report.record("exact_duplicates", len(docs) - len(deduped), len(deduped))

    kept: list[str] = []
    for doc in deduped:
        reason = filters.check(doc)
        if reason is None:
            kept.append(doc)
        else:
            report.rejections[reason] = report.rejections.get(reason, 0) + 1
    report.record("quality_filters", len(deduped) - len(kept), len(kept))

    if near_duplicate_threshold is not None and len(kept) > 1:
        signatures = minhash_signatures(kept, ngram=ngram, seed=seed)
        groups = near_duplicate_groups(signatures, threshold=near_duplicate_threshold)
        # Keep the first member of each group; index order is stable.
        drop = {i for group in groups for i in sorted(group)[1:]}
        survivors = [doc for i, doc in enumerate(kept) if i not in drop]
        report.record("near_duplicates", len(drop), len(survivors))
        kept = survivors

    report.output_documents = len(kept)
    report.output_bytes = sum(len(d.encode("utf-8")) for d in kept)
    for doc in kept:
        split = assign_split(doc)
        report.split_counts[split] = report.split_counts.get(split, 0) + 1
    return kept, report


def split_documents(documents: Iterable[str]) -> dict[str, list[str]]:
    """Partition by content hash. Identical documents always co-locate."""
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for doc in documents:
        out[assign_split(doc)].append(doc)
    return out


def check_leakage(splits: dict[str, list[str]]) -> dict[str, int]:
    """Count documents appearing in more than one split. Must be all zeros.

    With content-hash splitting this is structurally impossible, which is
    precisely why it is worth asserting: the test proves the property rather
    than the intention.
    """
    hashes = {name: {document_hash(d) for d in docs} for name, docs in splits.items()}
    overlaps: dict[str, int] = {}
    names = sorted(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}&{b}"] = len(hashes[a] & hashes[b])
    return overlaps


def iter_lines(path: Path | str, encoding: str = "utf-8") -> Iterator[str]:
    with Path(path).open(encoding=encoding) as fh:
        for line in fh:
            yield line.rstrip("\n")
