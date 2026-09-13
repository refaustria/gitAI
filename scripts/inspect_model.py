#!/usr/bin/env python3
"""Look inside a trained model.

    uv run python scripts/inspect_model.py                    # latest run
    uv run python scripts/inspect_model.py --run runs/... --prompt "First Citizen:"

A model this size does not produce a reasoning trace — it predicts next tokens,
it does not deliberate. What it offers instead is more direct: at ~1M parameters
every intermediate is observable, so you can watch the actual computation rather
than read a narration of it.

Six views, running from correlational to causal. Only the last two support
claims about what *caused* a prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.data import BatchSampler, ShardIndex
from gitai.interpret import (
    ablate_heads,
    ablation_table,
    attention_heatmap,
    attribute_heads,
    attribute_logit,
    attribution_table,
    colour_by_surprisal,
    lens_table,
    lens_trajectory,
    logit_difference,
    patch_residual,
    predict_tokens,
    visible,
)
from gitai.model import ModelConfig, Transformer
from gitai.tokenizer import ByteBPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*/checkpoints/best.safetensors"))
    if not runs:
        sys.exit("no trained runs found — run scripts/train.py first")
    return runs[-1].parent.parent


def load(run_dir: Path, data_dir: Path) -> tuple[Transformer, object, dict]:
    from safetensors.torch import load_model

    meta = json.loads((run_dir / "checkpoints" / "best.json").read_text())
    model = Transformer(ModelConfig(**meta["model"]))
    load_model(model, str(run_dir / "checkpoints" / "best.safetensors"))
    model.eval()

    tokenizer_path = data_dir / "tokenizer.json"
    kind = json.loads(tokenizer_path.read_text()).get("kind")
    loader = CharTokenizer.load if kind == "char" else ByteBPETokenizer.load
    tokenizer = loader(tokenizer_path)
    return model, tokenizer, meta


def banner(n: int, title: str, subtitle: str = "") -> None:
    print(f"\n{'=' * 78}\n{n}. {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--data", default=str(ROOT / "data/processed/tinyshakespeare"))
    parser.add_argument(
        "--prompt", default="First Citizen:\nBefore we proceed any further, hear me"
    )
    parser.add_argument("--position", type=int, default=None, help="position to analyse in depth")
    parser.add_argument("--layer", type=int, default=0, help="layer for the attention map")
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run()
    data_dir = Path(args.data)
    model, tokenizer, meta = load(run_dir, data_dir)
    colour = None if not args.no_colour else False

    print(f"run   {run_dir.name}")
    print(f"step  {meta['step']}   val BPB {meta['val_bpb']:.4f}")
    print(model.config.summary())

    ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long)
    ids = ids[: model.config.seq_len]
    tokens = [tokenizer.decode([int(t)]) for t in ids]

    # ---------------------------------------------------------------- 1
    banner(
        1,
        "Per-token surprisal",
        "green = predictable, red = blindsided. This is the loss, "
        "decomposed to where it was incurred.",
    )
    predictions = predict_tokens(model, tokenizer, ids, top_k=4)
    print(colour_by_surprisal(predictions, colour=colour))

    # ---------------------------------------------------------------- 2
    banner(2, "What it expected, token by token")
    print(f"{'pos':>4} {'context':>10} {'actual':>10} {'bits':>7} {'rank':>5}   top predictions")
    print("-" * 78)
    for p in predictions:
        top = "  ".join(f"{visible(t)[:8]!r}:{q:.2f}" for t, q in p.top[:3])
        flag = " ←confident+wrong" if p.confident_and_wrong else ""
        print(
            f"{p.position:>4} {visible(p.context_token)[:10]:>10} "
            f"{visible(p.actual_token)[:10]:>10} {p.surprisal:>7.2f} {p.rank:>5}   {top}{flag}"
        )

    # pick the most interesting position: the one the model got most wrong
    position = (
        args.position
        if args.position is not None
        else max(predictions, key=lambda p: p.surprisal).position
    )
    target = int(ids[position + 1])
    print(
        f"\nanalysing position {position}: "
        f"{visible(tokenizer.decode([int(ids[position])]))!r} -> "
        f"{visible(tokenizer.decode([target]))!r}"
    )

    # ---------------------------------------------------------------- 3
    banner(
        3,
        "Logit lens — the prediction forming across depth",
        "decode the residual stream at each layer. Caveat: norm_f was trained on the "
        "final layer, so early rows may be noise.",
    )
    rows = lens_trajectory(model, tokenizer, ids, position=position, top_k=3)
    print(lens_table(rows, tokenizer.decode([target]), colour=colour))

    # ---------------------------------------------------------------- 4
    banner(
        4,
        f"Attention — layer {args.layer}",
        "rows are queries, columns keys. Suggestive, not explanatory.",
    )
    _, _, cache = model.run_with_cache(ids.unsqueeze(0))
    pattern = cache.attention(args.layer)[0]
    for head in range(min(2, model.config.n_head)):
        print(f"\n  head {head}:")
        print(attention_heatmap(pattern[head], tokens, max_tokens=22, colour=colour))

    # ---------------------------------------------------------------- 5
    banner(
        5,
        "Direct logit attribution",
        "exact decomposition — the parts sum to the logit, which is asserted in tests.",
    )
    contributions, offset = attribute_logit(model, cache, position, target)
    print(attribution_table(contributions, offset))
    print(f"\n  actual logit: {float(cache['logits'][0, position, target]):.3f}")

    heads = attribute_heads(model, cache, position, target)
    best = heads.abs().flatten().argmax()
    layer, head = divmod(int(best), model.config.n_head)
    print(f"  strongest single head: L{layer}H{head} ({float(heads[layer, head]):+.3f} logits)")

    # ---------------------------------------------------------------- 6
    banner(
        6,
        "Causal: head ablation",
        "zero each head, measure the loss increase. "
        "Unlike everything above, this is an intervention.",
    )
    index = ShardIndex.load(data_dir)
    val = BatchSampler(data_dir, "val", min(64, model.config.seq_len), index=index)
    x, y = val.batch(8, np.random.default_rng(0))
    deltas = ablate_heads(model, torch.from_numpy(x), torch.from_numpy(y))
    print(ablation_table(deltas, top=8))
    dead = int((deltas.abs() < 1e-4).sum())
    print(f"\n  {dead}/{deltas.numel()} heads change the loss by < 1e-4 when removed.")

    # ---------------------------------------------------------------- 7
    banner(
        7,
        "Causal: activation patching",
        "splice clean activations into a corrupted run. Where does the information live?",
    )
    # Corrupt an EARLIER position than the one measured. Corrupting the measured
    # position itself makes the answer trivial: patching it back recovers 100% at
    # every layer and the map shows nothing. Corrupting upstream forces the
    # information to travel, and the map then shows where it travels through.
    corrupted_at = max(0, position - 4)
    corrupt = ids.clone()
    corrupt[corrupted_at] = (corrupt[corrupted_at] + 7) % model.config.vocab_size
    # Contrast against the model's own runner-up: a logit difference isolates one
    # decision and is unaffected by the rest of the distribution shifting.
    other = int(cache["logits"][0, position].topk(2).indices[-1])
    metric = logit_difference(target, other, position=position)

    clean_score = metric(cache["logits"])
    with torch.no_grad():
        corrupt_logits, _ = model(corrupt.unsqueeze(0))
    corrupt_score = metric(corrupt_logits)

    sites = list(range(corrupted_at, position + 1))
    patched = patch_residual(model, ids.unsqueeze(0), corrupt.unsqueeze(0), metric, positions=sites)

    print(
        f"corrupted position {corrupted_at} "
        f"({visible(tokenizer.decode([int(ids[corrupted_at])]))!r} -> "
        f"{visible(tokenizer.decode([int(corrupt[corrupted_at])]))!r}), "
        f"measuring at position {position}"
    )
    print(
        f"metric: logit({visible(tokenizer.decode([target]))!r}) "
        f"- logit({visible(tokenizer.decode([other]))!r})"
    )
    print(f"  clean   {clean_score:+.3f}")
    print(f"  corrupt {corrupt_score:+.3f}\n")
    header = "layer  " + "".join(f"{p:>8}" for p in sites)
    print(header)
    print("-" * len(header))
    span = abs(clean_score - corrupt_score) or 1.0
    for layer in range(model.config.n_layer):
        cells = "".join(f"{float(patched[layer, c]):>8.2f}" for c in range(len(sites)))
        recovery = (float(patched[layer].max()) - corrupt_score) / span
        print(f"  L{layer}   {cells}   recovered {recovery:>6.0%}")
    print("\n  a cell near the clean score means that site carried the information.")

    print(f"\n{'=' * 78}\nNone of this is a reasoning trace. It is the computation itself.")


if __name__ == "__main__":
    main()
