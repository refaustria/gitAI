#!/usr/bin/env python3
"""Phase 5: the model-collapse experiment (lab-notebook E3).

    uv run python scripts/collapse_experiment.py --generations 3 --seeds 3

Three arms differing only in what generation *n* is trained on: real data
(control), the parent model's output (replace), or both (accumulate). Each
generation is a freshly initialised model, so what is inherited is the data, not
the weights.

Results are appended to ``runs/collapse/results.jsonl`` as they are produced, so
a partial run is still usable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import BatchSampler, ShardIndex, read_documents, tokenize_to_shards
from gitai.model import RUNGS, ModelConfig, Transformer
from gitai.selftrain import ARMS, Anchor, generate_corpus
from gitai.tokenizer import ByteBPETokenizer
from gitai.training import TrainConfig, train

ROOT = Path(__file__).resolve().parent.parent
EOT = "<|endoftext|>"


def build_model(vocab_size: int, args) -> Transformer:
    return Transformer(
        ModelConfig(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            **RUNGS["v6_modern"],
        )
    )


def shard(documents, tokenizer, work: Path, separator: int, seq_len: int) -> BatchSampler:
    """Tokenize a corpus into a throwaway shard directory and open a sampler."""
    if work.exists():
        shutil.rmtree(work)
    tokenize_to_shards(documents, tokenizer, work, split="train", document_separator=separator)
    return BatchSampler(work, "train", seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument("--parquet", default=str(ROOT / "data/interim/tinyshakespeare.parquet"))
    parser.add_argument("--out", default=str(ROOT / "runs/collapse"))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--gen-batch", type=int, default=256)
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated subset of arms to run (control is temperature-independent)",
    )
    parser.add_argument(
        "--real-fraction",
        type=float,
        default=None,
        help="run the fixed-pool anchor arm at this real-data fraction instead of a named arm",
    )
    args = parser.parse_args()

    if args.real_fraction is not None:
        selected = ["anchor"]
    else:
        selected = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = set(selected) - set(ARMS)
        if unknown:
            parser.error(f"unknown arm(s) {sorted(unknown)}; available: {sorted(ARMS)}")

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    work_root = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    index = ShardIndex.load(data_dir)
    tokenizer = ByteBPETokenizer.load(data_dir / "tokenizer.json")
    separator = tokenizer.special_tokens[EOT]

    real_documents = read_documents(args.parquet, split="train")
    real_tokens = index.split("train").tokens

    # The constant yardstick: held-out REAL data, identical for every arm,
    # generation and seed. A collapsing model gets better at predicting its own
    # output and worse at predicting reality; only this distinguishes them.
    val_sampler = BatchSampler(data_dir, "val", args.seq_len, index=index)

    results_path = out_dir / "results.jsonl"
    results = results_path.open("a", encoding="utf-8")

    def record(row: dict) -> None:
        # Every row carries the sampling regime it was produced under, so one
        # results file can hold several sweeps and stay unambiguous.
        row.setdefault("temperature", args.temperature)
        if args.real_fraction is not None:
            row.setdefault("real_fraction", args.real_fraction)
        row.setdefault("top_k", args.top_k)
        row.setdefault("steps", args.steps)
        results.write(json.dumps(row) + "\n")
        results.flush()

    print(f"real corpus: {len(real_documents):,} documents / {real_tokens:,} tokens")
    print(f"val: {val_sampler.total_tokens:,} tokens (real, held out, constant)")
    regime = f"temperature {args.temperature}" + (f", top-k {args.top_k}" if args.top_k else "")
    print(f"{args.generations} generations x {args.seeds} seeds x {len(selected)} arms")
    print(f"sampling: {regime}\n")

    started = time.perf_counter()
    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=min(100, args.steps // 5),
        eval_every=max(1, args.steps // 2),
        log_every=10**9,  # quiet; the experiment prints its own summary
    )

    for seed in range(args.seeds):
        # ---- generation 0: one model per seed, shared as the parent of every arm
        print(f"seed {seed} | generation 0 (shared, real data)")
        sampler = shard(
            real_documents, tokenizer, work_root / f"s{seed}-g0", separator, args.seq_len
        )
        model = build_model(index.vocab_size, args)
        config = TrainConfig(**{**train_config.__dict__, "seed": seed})
        outcome = train(model, sampler, val_sampler, config)
        print(f"          val BPB {outcome.best_bpb:.4f}   ({outcome.seconds:.0f}s)")
        record(
            {
                "arm": "generation0",
                "seed": seed,
                "generation": 0,
                "val_bpb": outcome.best_bpb,
                "train_loss": outcome.final_train_loss,
                "train_tokens": sampler.total_tokens,
                "synthetic_tokens": 0,
            }
        )

        # One synthetic corpus from the shared parent, used by every arm at gen 1.
        print("          generating synthetic corpus from generation 0 ...")
        gen0_documents, gen0_stats = generate_corpus(
            model,
            tokenizer,
            real_tokens,
            separator_id=separator,
            batch_size=args.gen_batch,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=seed,
        )
        print(f"          {gen0_stats}")
        record(
            {
                "arm": "generation0",
                "seed": seed,
                "generation": 0,
                "corpus_stats": gen0_stats.to_dict(),
            }
        )

        for arm_name in selected:
            arm = (
                Anchor(real_fraction=args.real_fraction, seed=seed)
                if arm_name == "anchor"
                else ARMS[arm_name]()
            )
            arm_name = arm.name
            history: list[list[str]] = [gen0_documents] if arm.needs_generation else []
            parent_stats = gen0_stats

            for generation in range(1, args.generations + 1):
                label = f"seed {seed} | {arm_name:<10} gen {generation}"
                corpus = arm.corpus(real_documents, history)
                sampler = shard(
                    corpus,
                    tokenizer,
                    work_root / f"s{seed}-{arm_name}-g{generation}",
                    separator,
                    args.seq_len,
                )

                model = build_model(index.vocab_size, args)
                config = TrainConfig(**{**train_config.__dict__, "seed": seed * 100 + generation})
                outcome = train(model, sampler, val_sampler, config)

                row = {
                    "arm": arm_name,
                    "seed": seed,
                    "generation": generation,
                    "val_bpb": outcome.best_bpb,
                    "train_loss": outcome.final_train_loss,
                    "train_tokens": sampler.total_tokens,
                    "documents": len(corpus),
                    "parent_distinct_3": parent_stats.distinct_3 if arm.needs_generation else None,
                    "parent_vocabulary_fraction": (
                        parent_stats.vocabulary_fraction if arm.needs_generation else None
                    ),
                    "seconds": outcome.seconds,
                }
                print(
                    f"{label}  val BPB {outcome.best_bpb:.4f}  "
                    f"train loss {outcome.final_train_loss:.4f}  "
                    f"({sampler.total_tokens:,} tokens, {outcome.seconds:.0f}s)"
                )

                # Generate this generation's output, unless nothing will consume it.
                if arm.needs_generation and generation < args.generations:
                    documents, stats = generate_corpus(
                        model,
                        tokenizer,
                        real_tokens,
                        separator_id=separator,
                        batch_size=args.gen_batch,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        seed=seed * 1000 + generation,
                    )
                    history.append(documents)
                    parent_stats = stats
                    row["own_corpus_stats"] = stats.to_dict()
                    print(
                        f"{' ' * len(label)}  generated: distinct_3 {stats.distinct_3:.4f}  "
                        f"vocab {stats.vocabulary_fraction:.1%}"
                    )

                record(row)

        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True, exist_ok=True)

    results.close()
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"\ndone in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
