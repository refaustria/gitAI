"""Tokenized corpora as flat, memory-mappable binary shards.

Why a raw binary array and not a database — the reasoning is in
docs/data-and-storage.md, but the short version is that the training loop needs
millions of random-offset reads per second with zero deserialisation. A flat
``uint16`` array mapped by the OS *is* the optimal structure for that: the page
cache does the work, and there is no query planner, no index, and no decode step
in the way.

``uint16`` because the vocabulary is 4k-8k (Decision 4), so ids fit in 16 bits
with room to spare. That halves the bytes and therefore halves the I/O, which on
a CPU-bound laptop is the difference that matters.

Each shard records whether it is synthetic and carries a provenance tag if so.
That is not bookkeeping for its own sake: it is what
``safety.GeneratedDataQuarantined`` checks before the improvement loop may
promote anything, and once real and generated data are mixed without markers,
the model-collapse question becomes permanently unanswerable for that corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gitai.tokenizer.base import Tokenizer

__all__ = ["ShardIndex", "ShardInfo", "SplitInfo", "choose_dtype", "tokenize_to_shards"]

META_FILENAME = "meta.json"


def choose_dtype(vocab_size: int) -> str:
    """Smallest integer type that can hold every id."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return "uint16"
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return "uint32"
    raise ValueError(f"vocab_size {vocab_size} is implausible")


@dataclass
class ShardInfo:
    path: str
    tokens: int
    documents: int
    utf8_bytes: int
    synthetic: bool = False
    quarantine_tag: str | None = None


@dataclass
class SplitInfo:
    tokens: int = 0
    documents: int = 0
    utf8_bytes: int = 0
    shards: list[ShardInfo] = field(default_factory=list)


class ShardIndex:
    """``meta.json``: the record of what is in a processed corpus directory.

    ``utf8_bytes`` is tracked per split and is not incidental — it is the
    denominator of bits-per-byte, the project's primary metric. Perplexity is
    per-token and therefore incomparable across vocabularies; BPB needs the raw
    byte count of the original text, and the only honest time to record it is
    while tokenizing.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.vocab_size: int = 0
        self.dtype: str = "uint16"
        self.tokenizer_fingerprint: str = ""
        self.splits: dict[str, SplitInfo] = {}

    # ------------------------------------------------------------------- io

    @classmethod
    def load(cls, root: Path | str) -> ShardIndex:
        root = Path(root)
        payload = json.loads((root / META_FILENAME).read_text(encoding="utf-8"))
        index = cls(root)
        index.vocab_size = payload["vocab_size"]
        index.dtype = payload["dtype"]
        index.tokenizer_fingerprint = payload["tokenizer_fingerprint"]
        for name, split in payload["splits"].items():
            index.splits[name] = SplitInfo(
                tokens=split["tokens"],
                documents=split["documents"],
                utf8_bytes=split["utf8_bytes"],
                shards=[ShardInfo(**s) for s in split["shards"]],
            )
        return index

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "dtype": self.dtype,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "splits": {
                name: {
                    "tokens": split.tokens,
                    "documents": split.documents,
                    "utf8_bytes": split.utf8_bytes,
                    "shards": [asdict(s) for s in split.shards],
                }
                for name, split in self.splits.items()
            },
        }
        (self.root / META_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- queries

    @property
    def np_dtype(self):
        return np.dtype(self.dtype)

    def split(self, name: str) -> SplitInfo:
        if name not in self.splits:
            raise KeyError(f"no split {name!r} in {self.root}; have {sorted(self.splits)}")
        return self.splits[name]

    def shard_paths(self, split: str) -> list[Path]:
        return [self.root / s.path for s in self.split(split).shards]

    def describe(self) -> str:
        lines = [
            f"{self.root}",
            f"  vocab {self.vocab_size}   dtype {self.dtype}   "
            f"tokenizer {self.tokenizer_fingerprint}",
        ]
        for name, split in sorted(self.splits.items()):
            ratio = split.utf8_bytes / split.tokens if split.tokens else 0.0
            synthetic = sum(s.tokens for s in split.shards if s.synthetic)
            lines.append(
                f"  {name:<6} {split.tokens:>12,} tokens  {split.documents:>9,} docs  "
                f"{ratio:.3f} bytes/token  {len(split.shards)} shard(s)"
                + (f"  [{synthetic:,} synthetic]" if synthetic else "")
            )
        return "\n".join(lines)


def tokenize_to_shards(
    documents: Iterable[str],
    tokenizer: Tokenizer,
    root: Path | str,
    split: str,
    shard_tokens: int = 50_000_000,
    document_separator: int | None = None,
    synthetic: bool = False,
    quarantine_tag: str | None = None,
) -> ShardIndex:
    """Tokenize ``documents`` into ``root/<split>_NNN.bin`` and update ``meta.json``.

    ``document_separator`` is usually the ``<|endoftext|>`` id. Without one, the
    model learns to continue straight from the end of one document into the
    start of the next, which is a real and commonly-shipped bug: it shows up as
    a model that cannot stop generating.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if synthetic and not quarantine_tag:
        raise ValueError(
            "synthetic shards require a quarantine_tag; untagged generated data "
            "makes the corpus permanently uninterpretable"
        )

    index = ShardIndex.load(root) if (root / META_FILENAME).exists() else ShardIndex(root)
    if index.vocab_size and index.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"{root} was built with vocab_size {index.vocab_size}, "
            f"but this tokenizer has {tokenizer.vocab_size}"
        )
    if index.tokenizer_fingerprint and index.tokenizer_fingerprint != tokenizer.fingerprint():
        raise ValueError(
            f"{root} was built with tokenizer {index.tokenizer_fingerprint}, "
            f"this one is {tokenizer.fingerprint()}. Mixing tokenizers in one "
            "corpus produces silently meaningless data."
        )
    index.vocab_size = tokenizer.vocab_size
    index.dtype = choose_dtype(tokenizer.vocab_size)
    index.tokenizer_fingerprint = tokenizer.fingerprint()

    info = index.splits.setdefault(split, SplitInfo())
    shard_number = len(info.shards)
    dtype = np.dtype(index.dtype)

    buffer: list[int] = []
    shard_docs = 0
    shard_bytes = 0

    def flush() -> None:
        nonlocal buffer, shard_number, shard_docs, shard_bytes
        if not buffer:
            return
        name = f"{split}_{shard_number:03d}.bin"
        np.asarray(buffer, dtype=dtype).tofile(root / name)
        info.shards.append(
            ShardInfo(
                path=name,
                tokens=len(buffer),
                documents=shard_docs,
                utf8_bytes=shard_bytes,
                synthetic=synthetic,
                quarantine_tag=quarantine_tag,
            )
        )
        info.tokens += len(buffer)
        info.documents += shard_docs
        info.utf8_bytes += shard_bytes
        shard_number += 1
        buffer, shard_docs, shard_bytes = [], 0, 0

    for document in documents:
        ids = tokenizer.encode(document)
        if document_separator is not None:
            ids = [*ids, document_separator]
        buffer.extend(ids)
        shard_docs += 1
        shard_bytes += len(document.encode("utf-8"))
        if len(buffer) >= shard_tokens:
            flush()
    flush()

    index.save()
    return index
