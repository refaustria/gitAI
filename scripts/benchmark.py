#!/usr/bin/env python3
"""Measure this machine's training throughput.

Run this on the laptop you will actually train on, then paste the output into
docs/hardware-baseline.md. Every scoping decision later in the project — how
large a model, how many tokens, how long a run — should be derived from these
numbers rather than from anyone's guess about your hardware.

    uv run python scripts/benchmark.py

Requires the `train` extra:  uv pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CONFIGS = [
    # (name, n_layer, d_model, n_head, seq_len, batch)
    ("tiny", 4, 128, 4, 256, 16),
    ("small", 6, 256, 8, 256, 16),
    ("medium", 8, 384, 6, 512, 8),
    ("large", 12, 512, 8, 512, 8),
]


def _require_torch():
    try:
        import torch
    except ImportError:
        sys.exit('torch is not installed. Run:  uv pip install -e ".[train]"')
    return torch


def pick_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_real_model(n_layer: int, d_model: int, n_head: int, seq_len: int, vocab: int = 4096):
    """The architecture you will actually train: v6_modern.

    The proxy below predates the real model and was never swapped out, so every
    number in docs/hardware-baseline.md described a transformer nobody trains.
    It was not a small discrepancy: the proxy's `nn.MultiheadAttention` and
    LayerNorm/GELU stack runs ~1.6x *slower* than the real RMSNorm/RoPE/SwiGLU
    model at the same dimensions, so the table overstated every training cost
    by that factor. A benchmark whose whole purpose is scoping real runs has to
    time the real thing.
    """
    from gitai.model import RUNGS, ModelConfig, Transformer

    config = ModelConfig(
        vocab_size=vocab,
        seq_len=seq_len,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        **RUNGS["v6_modern"],
    )
    return Transformer(config)


def build_model(torch, n_layer: int, d_model: int, n_head: int, vocab: int = 4096):
    """A throwaway pre-norm transformer, kept only for --proxy comparisons.

    Retained so the pre-2026-09-15 numbers in docs/hardware-baseline.md stay
    reproducible, not because it should be used for scoping.
    """
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
            )

        def forward(self, x):
            h = self.ln1(x)
            n = x.size(1)
            ones = torch.ones(n, n, device=x.device, dtype=torch.bool)
            mask = torch.triu(ones, 1)
            x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            return x + self.mlp(self.ln2(x))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.embed.weight  # tied, as the real model will be

        def forward(self, idx):
            x = self.embed(idx)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    return Model()


def benchmark_one(torch, cfg, device: str, steps: int, compile_model: bool, proxy: bool) -> dict:
    import torch.nn.functional as F

    name, n_layer, d_model, n_head, seq_len, batch = cfg
    torch.manual_seed(0)

    if proxy:
        model = build_model(torch, n_layer, d_model, n_head).to(device)
    else:
        model = build_real_model(n_layer, d_model, n_head, seq_len).to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as exc:
            return {"config": name, "error": f"torch.compile failed: {exc}"}

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 4096, (batch, seq_len), device=device)
    y = torch.randint(0, 4096, (batch, seq_len), device=device)

    params = sum(p.numel() for p in model.parameters())
    embed_params = 4096 * d_model

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        out = model(x)
        # The real model returns (logits, cache); the proxy returns logits.
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()

    for _ in range(3):  # warmup: allocator, kernel autotuning, compile
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = steps * batch * seq_len
    return {
        "config": name,
        "params_total": params,
        "params_non_embedding": params - embed_params,
        "seq_len": seq_len,
        "batch": batch,
        "sec_per_step": elapsed / steps,
        "tokens_per_sec": tokens / elapsed,
        "hours_per_100M_tokens": (100e6 / (tokens / elapsed)) / 3600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="also measure torch.compile")
    parser.add_argument("--json", metavar="PATH", help="write results as JSON")
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="time the throwaway pre-norm model instead of v6_modern; only for "
        "reproducing numbers recorded before 2026-09-15",
    )
    args = parser.parse_args()

    torch = _require_torch()
    device = pick_device(torch, args.device)

    print(f"platform      {platform.platform()}")
    print(f"processor     {platform.processor() or 'unknown'}")
    print(f"python        {platform.python_version()}")
    print(f"torch         {torch.__version__}")
    print(f"device        {device}")
    print(f"threads       {torch.get_num_threads()}")
    print()

    header = (
        f"{'config':<8} {'params':>10} {'non-emb':>10} {'s/step':>9} {'tok/s':>10} {'h/100M':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for cfg in CONFIGS:
        row = benchmark_one(
            torch, cfg, device, args.steps, compile_model=args.compile, proxy=args.proxy
        )
        results.append(row)
        if "error" in row:
            print(f"{row['config']:<8} {row['error']}")
            continue
        print(
            f"{row['config']:<8} {row['params_total']:>10,} {row['params_non_embedding']:>10,} "
            f"{row['sec_per_step']:>9.3f} {row['tokens_per_sec']:>10,.0f} "
            f"{row['hours_per_100M_tokens']:>9.1f}"
        )

    print()
    print("h/100M is wall-clock hours to see 100M tokens once.")
    print("Pick the largest config whose number you are willing to wait for, and")
    print("record it in docs/hardware-baseline.md.")

    if args.json:
        payload = {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": device,
            "threads": torch.get_num_threads(),
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
