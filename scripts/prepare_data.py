#!/usr/bin/env python3
"""Raw corpus -> curated Parquet -> tokenizer -> memory-mapped training shards.

    uv run python scripts/prepare_data.py --corpus tinyshakespeare --vocab-size 1024
    uv run python scripts/prepare_data.py --corpus tinystories --vocab-size 4096
    uv run python scripts/prepare_data.py --corpus tinyshakespeare --sweep

Every stage prints what it removed and why. A pipeline that silently drops most
of a corpus is a bug worth finding now rather than after a three-day run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import (
    BatchSampler,
    QualityFilters,
    check_leakage,
    corpus_stats,
    curate,
    fetch,
    split_documents,
    tokenize_to_shards,
    write_parquet,
)
from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
EOT = "<|endoftext|>"


def split_into_documents(text: str, corpus: str) -> list[str]:
    """Split a raw corpus into documents.

    Corpora do not agree on how documents are separated, and guessing wrong is
    expensive but silent: TinyStories uses an explicit ``<|endoftext|>`` marker,
    and splitting it on blank lines instead yields 2,629 enormous documents
    where there should be 27,631 — each one dozens of unrelated stories glued
    together. Nothing errors; the corpus is simply wrong.

    So the marker is checked for first, and blank-line splitting is the fallback
    for corpora like TinyShakespeare that have no explicit separator.
    """
    chunks = text.split(EOT) if EOT in text else text.split("\n\n")
    return [c.strip() for c in chunks if c.strip()]


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="tinyshakespeare")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--char", action="store_true", help="character tokenizer instead of BPE")
    parser.add_argument("--sweep", action="store_true", help="compare vocab sizes and stop")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--no-near-dedup", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ---------------------------------------------------------------- acquire
    banner(f"1. acquire — {args.corpus}")
    raw_path = fetch(args.corpus, dest_dir=data_dir / "raw")
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    documents = split_into_documents(text, args.corpus)
    if args.max_documents:
        documents = documents[: args.max_documents]
    print(f"{raw_path}: {len(text):,} chars -> {len(documents):,} documents")

    # ----------------------------------------------------------------- curate
    banner("2. curate")
    started = time.perf_counter()
    kept, report = curate(
        documents,
        filters=QualityFilters(min_chars=16, min_words=3),
        near_duplicate_threshold=None if args.no_near_dedup else 0.85,
    )
    print(report)
    print(f"\n({time.perf_counter() - started:.1f}s)")

    # ---------------------------------------------------------------- parquet
    banner("3. parquet + duckdb")
    interim = data_dir / "interim" / f"{args.corpus}.parquet"
    write_parquet(kept, interim, source=args.corpus)
    stats = corpus_stats(interim)
    for key, value in stats.items():
        print(f"  {key:<28} {value}")

    splits = split_documents(kept)
    leakage = check_leakage(splits)
    print(f"\n  leakage between splits: {leakage}")
    assert all(v == 0 for v in leakage.values()), "split leakage detected"

    # -------------------------------------------------------------- tokenizer
    if args.sweep:
        banner("4. vocabulary sweep")
        sample = "\n".join(kept[: min(len(kept), 2000)])
        print(f"{'vocab':>8} {'merges':>8} {'bytes/token':>13} {'tokens':>12} {'train s':>9}")
        print("-" * 56)
        rows = []
        for size in (256, 512, 1024, 2048, 4096, 8192):
            begun = time.perf_counter()
            tok = ByteBPETokenizer.train(splits["train"], vocab_size=size)
            elapsed = time.perf_counter() - begun
            ratio = tok.compression_ratio(sample)
            count = len(tok.encode(sample))
            rows.append({"vocab_size": tok.vocab_size, "bytes_per_token": ratio, "tokens": count})
            print(
                f"{tok.vocab_size:>8} {len(tok.merges):>8} {ratio:>13.3f} "
                f"{count:>12,} {elapsed:>9.1f}"
            )
        out = data_dir / "interim" / f"{args.corpus}-vocab-sweep.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {out}")
        print(
            "\nA larger vocabulary always compresses better, so the number that matters is\n"
            "the gain per embedding parameter. At d_model=384 each extra 1024 vocab entries\n"
            "cost ~393k parameters — compare that against the sequence-length saving before\n"
            "choosing (Decision 4)."
        )
        return

    banner(f"4. tokenizer — {'char' if args.char else f'byte-BPE, vocab {args.vocab_size}'}")
    started = time.perf_counter()
    if args.char:
        tokenizer = CharTokenizer.train(splits["train"])
        separator = None
    else:
        tokenizer = ByteBPETokenizer.train(
            splits["train"], vocab_size=args.vocab_size, special_tokens=[EOT], verbose=True
        )
        separator = tokenizer.special_tokens[EOT]
    print(f"vocab {tokenizer.vocab_size}, fingerprint {tokenizer.fingerprint()}")
    print(f"trained in {time.perf_counter() - started:.1f}s")

    sample = kept[0][:200]
    ids = tokenizer.encode(sample)
    print(f"\n  sample     {sample[:70]!r}")
    print(f"  tokens     {ids[:16]}...")
    print(f"  round-trip {'exact' if tokenizer.decode(ids) == sample else 'BROKEN'}")
    print(f"  bytes/token {tokenizer.compression_ratio(sample):.3f}")

    tokenizer_dir = data_dir / "processed" / args.corpus
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_dir / "tokenizer.json")

    # ----------------------------------------------------------------- shards
    banner("5. shards")
    for name in ("train", "val", "test"):
        if splits[name]:
            tokenize_to_shards(
                splits[name], tokenizer, tokenizer_dir, split=name, document_separator=separator
            )
    from gitai.data import ShardIndex

    index = ShardIndex.load(tokenizer_dir)
    print(index.describe())

    # ----------------------------------------------------------------- loader
    banner("6. loader")
    sampler = BatchSampler(tokenizer_dir, "train", seq_len=args.seq_len)
    x, y = sampler.batch(8, __import__("numpy").random.default_rng(0))
    print(f"batch {x.shape} {x.dtype}, targets shifted by one: {(x[:, 1:] == y[:, :-1]).all()}")
    speed = sampler.throughput(batch_size=8, steps=100)
    print(
        f"loader throughput {speed['tokens_per_sec']:,.0f} tokens/sec "
        f"({speed['ms_per_batch']:.2f} ms/batch)"
    )
    print(
        "\nCompare that against your model's step time from `make bench`. On CPU the\n"
        "loader is often the bottleneck, which is the opposite of the GPU intuition."
    )
    print(f"\ndone. shards in {tokenizer_dir}")


if __name__ == "__main__":
    main()
