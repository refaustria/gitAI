#!/usr/bin/env python3
"""Run the full evaluation suite on a trained checkpoint.

    uv run python scripts/evaluate.py                  # latest run
    uv run python scripts/evaluate.py --run runs/... --probes

Writes the result to the run's directory as ``eval.json``, stamped with the
eval-suite version so a later change to the suite cannot silently invalidate
the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import BatchSampler, ShardIndex
from gitai.eval import InductionProbe, evaluate, frequent_tokens
from gitai.model import BigramModel, ModelConfig, Transformer
from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*/checkpoints/best.safetensors"))
    if not runs:
        sys.exit("no trained runs found — run scripts/train.py first")
    return runs[-1].parent.parent


def load(run_dir: Path, data_dir: Path):
    from safetensors.torch import load_model

    meta = json.loads((run_dir / "checkpoints" / "best.json").read_text())
    spec = meta["model"]
    if spec.get("kind") == "bigram":
        model = BigramModel(ShardIndex.load(data_dir).vocab_size)
    else:
        model = Transformer(ModelConfig(**spec))
    load_model(model, str(run_dir / "checkpoints" / "best.safetensors"))
    model.eval()

    path = data_dir / "tokenizer.json"
    kind = json.loads(path.read_text()).get("kind")
    loader = CharTokenizer.load if kind == "char" else ByteBPETokenizer.load
    return model, loader(path), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probes", action="store_true", help="also run capability probes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run()
    data_dir = Path(args.data)
    model, tokenizer, meta = load(run_dir, data_dir)

    index = ShardIndex.load(data_dir)
    seq_len = getattr(getattr(model, "config", None), "seq_len", 128)
    sampler = BatchSampler(data_dir, args.split, seq_len, index=index)

    probes = []
    if args.probes and hasattr(model, "config"):
        # Sample the probe's random sequences from *frequent* tokens: drawing
        # uniformly over the vocabulary would mostly draw rare tokens whose
        # embeddings barely moved, measuring coverage rather than induction.
        pool = frequent_tokens(BatchSampler(data_dir, "train", seq_len, index=index), top=256)
        probes.append(InductionProbe(block_len=min(48, seq_len // 2), trials=32, token_pool=pool))

    print(f"run  {run_dir.name}   (checkpoint step {meta['step']})")
    result = evaluate(
        model,
        sampler,
        batch_size=args.batch_size,
        probes=probes,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    print()
    print(result)

    if result.probes:
        score = result.probes.get("induction_score_bits", 0.0)
        print()
        print(f"  induction: the second copy of a random sequence costs {score:.2f} bits")
        print("  fewer than the first. Near zero means no in-context copying at all;")
        print("  a large positive value means induction heads have formed.")

    if result.samples:
        print(f"\nfixed-prompt samples (seed {args.seed}):")
        for sample in result.samples[:1]:
            print("-" * 72)
            print(sample[:600])
            print("-" * 72)

    out = run_dir / "eval.json"
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
