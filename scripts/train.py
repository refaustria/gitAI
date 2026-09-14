#!/usr/bin/env python3
"""Train a model on prepared shards.

    uv run python scripts/train.py --steps 1500 --rung v6_modern

A working trainer with the parts that matter for reproducibility: resolved
config and run manifest written before the first step, metrics streamed to
JSONL, checkpoints in safetensors. The remaining Phase 3 work — resume,
gradient accumulation equivalence, checkpoint rotation — is marked in TODO.md.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import BatchSampler, ShardIndex
from gitai.interpret import bits_per_byte, surprisal_bits
from gitai.model import RUNGS, BigramModel, ModelConfig, Transformer
from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def load_tokenizer(path: Path):
    kind = json.loads(path.read_text(encoding="utf-8")).get("kind")
    return CharTokenizer.load(path) if kind == "char" else ByteBPETokenizer.load(path)


def lr_at(step: int, total: int, peak: float, warmup: int, floor_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay — the schedule almost everything uses."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, sampler: BatchSampler, batch_size: int, max_batches: int = 50) -> dict:
    """Held-out loss and bits-per-byte.

    Evaluated sequentially rather than by random sampling: random windows would
    score some tokens twice and skip others, adding variance to a number whose
    whole purpose is cross-run comparison.
    """
    model.eval()
    total_bits = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(sampler.sequential(batch_size)):
        if i >= max_batches:
            break
        logits, _ = model(torch.from_numpy(x), torch.from_numpy(y))
        bits = surprisal_bits(logits, torch.from_numpy(y))
        total_bits += float(bits.sum())
        total_tokens += bits.numel()
    model.train()

    bytes_per_token = sampler.utf8_bytes / sampler.total_tokens
    return {
        "val_loss": total_bits / total_tokens * math.log(2),
        "val_bits_per_token": total_bits / total_tokens,
        "val_bpb": bits_per_byte(total_bits, int(total_tokens * bytes_per_token)),
        "val_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument("--rung", default="v6_modern")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="split each effective batch into this many forward passes",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write a resumable checkpoint every N steps (0 = off)",
    )
    parser.add_argument(
        "--keep-last", type=int, default=2, help="how many step checkpoints to retain"
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help="resume from the latest checkpoint in this run directory",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    index = ShardIndex.load(data_dir)
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")

    if args.rung == "bigram":
        # The floor. A context-free lookup table: whatever loss it reaches is the
        # best any model without context can do, which is what turns "is 1.87
        # good?" from an opinion into a measurement.
        model = BigramModel(index.vocab_size)
        config = None
        print(f"bigram baseline: {model.num_parameters():,} parameters (all embedding)")
    else:
        config = ModelConfig(
            vocab_size=index.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            **RUNGS[args.rung],
        )
        model = Transformer(config)
        print(config.summary())

    train = BatchSampler(data_dir, "train", args.seq_len, index=index)
    val = BatchSampler(data_dir, "val", args.seq_len, index=index)
    print(f"\ntrain {train.total_tokens:,} tokens   val {val.total_tokens:,} tokens")
    epochs = args.steps * args.batch_size * args.seq_len / train.total_tokens
    print(f"{args.steps} steps x {args.batch_size} x {args.seq_len} = {epochs:.1f} epochs\n")

    # Weight decay on matrices only. Decaying norms, biases and embeddings is a
    # small, silent, universally-copied mistake.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{stamp}-{args.name or args.rung}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(
        json.dumps(
            {"model": asdict(config) if config else {"kind": "bigram"}, "training": vars(args)},
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git": git_state(),
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "tokenizer_fingerprint": index.tokenizer_fingerprint,
                "data_dir": str(data_dir),
                "train_tokens": train.total_tokens,
                "seeds": {"torch": args.seed, "numpy": args.seed},
                "platform": platform.platform(),
                "torch": torch.__version__,
                "parameters": (
                    config.parameter_count()
                    if config
                    else {"total": model.num_parameters(), "non_embedding": 0}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_file = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    model.train()
    started = time.perf_counter()
    best_bpb = float("inf")

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for group in optimiser.param_groups:
            group["lr"] = lr

        x, y = train.batch(args.batch_size, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - started
            tps = (step + 1) * args.batch_size * args.seq_len / elapsed
            record = {
                "step": step,
                "loss": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": tps,
                "elapsed": elapsed,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(
                f"  {step:>5}  loss {loss.item():.4f}  lr {lr:.2e}  "
                f"|g| {grad_norm.item():.2f}  {tps:,.0f} tok/s"
            )

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            stats = evaluate(model, val, args.batch_size)
            stats["step"] = step
            metrics_file.write(json.dumps(stats) + "\n")
            metrics_file.flush()
            print(f"        val loss {stats['val_loss']:.4f}   BPB {stats['val_bpb']:.4f}")
            if stats["val_bpb"] < best_bpb:
                best_bpb = stats["val_bpb"]
                # save_model, not save_file: with tied embeddings the head and
                # the token embedding are the *same tensor*, and save_file
                # refuses to write aliased storage. save_model records the
                # sharing and load_model restores it.
                from safetensors.torch import save_model

                save_model(model, str(run_dir / "checkpoints" / "best.safetensors"))
                (run_dir / "checkpoints" / "best.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            **stats,
                            "rung": args.rung,
                            "model": asdict(config) if config else {"kind": "bigram"},
                        },
                        indent=2,
                    )
                )

    metrics_file.close()

    print("\nsample (temperature 0.8):")
    prompt = torch.tensor([tokenizer.encode("First Citizen:\n")], dtype=torch.long)
    if args.rung == "bigram":
        sample = model.generate(prompt, 300, temperature=0.8)
    else:
        sample = model.generate(
            prompt,
            300,
            temperature=0.8,
            top_k=40,
            generator=torch.Generator().manual_seed(args.seed),
        )
    print("-" * 72)
    print(tokenizer.decode(sample[0].tolist()))
    print("-" * 72)
    print(f"\nbest val BPB {best_bpb:.4f}")
    print(f"run: {run_dir}")


if __name__ == "__main__":
    main()
