"""The logit lens: watching a prediction assemble across depth.

The residual stream is the model's working memory — every block reads from it
and adds back, never overwrites. So the stream at layer 3 is a partial answer,
expressed in the same space as the final one. Decoding it with the output head
shows what the model *would* have predicted had it stopped there.

This is the closest thing available to watching the model think, and unlike a
written reasoning trace it cannot be unfaithful: it is the actual computation,
read directly.

**The caveat that must travel with it.** ``norm_f`` was trained on final-layer
activations, so applying it to an intermediate layer is an approximation. Early
layers often decode to noise, and in some models the lens is misleading
throughout. Treat a clean lens trajectory as evidence and a messy one as
uninformative rather than as proof of disorder — and confirm anything important
with :mod:`gitai.interpret.patching`, which is causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["LensRow", "lens_trajectory", "logit_lens"]


@dataclass
class LensRow:
    layer: int  # -1 is the embedding, before any block has run
    label: str
    top: list[tuple[str, float]]
    target_prob: float
    target_rank: int


@torch.no_grad()
def logit_lens(model, cache) -> torch.Tensor:
    """Decode every residual-stream state. Returns ``(n_layer + 1, batch, seq, vocab)``.

    Index 0 is the embedding output; index *i+1* is the output of block *i*.
    """
    stream = cache.residual_stream()
    flat = stream.reshape(-1, stream.shape[-1])
    decoded = model.head(model.norm_f(flat))
    return decoded.reshape(*stream.shape[:-1], -1)


@torch.no_grad()
def lens_trajectory(
    model, tokenizer, ids: torch.Tensor, position: int, top_k: int = 3
) -> list[LensRow]:
    """How the prediction at one position develops layer by layer.

    The interesting shape to look for: the correct token climbing the ranking at
    a specific depth. That layer is where the computation responsible for this
    prediction lives, and it is the natural place to point
    :mod:`gitai.interpret.patching` next.
    """
    if ids.dim() != 1:
        raise ValueError(f"expected a 1-D sequence, got {tuple(ids.shape)}")
    if not -ids.numel() <= position < ids.numel() - 1:
        raise ValueError(f"position {position} has no following token to predict")

    was_training = model.training
    model.eval()
    try:
        _, _, cache = model.run_with_cache(ids.unsqueeze(0))
    finally:
        model.train(was_training)

    lens = logit_lens(model, cache)[:, 0, position, :].float()
    target = int(ids[position + 1])

    rows: list[LensRow] = []
    for layer in range(lens.shape[0]):
        probs = lens[layer].softmax(dim=-1)
        top_probs, top_ids = probs.topk(min(top_k, probs.numel()))
        rows.append(
            LensRow(
                layer=layer - 1,
                label="embed" if layer == 0 else f"block {layer - 1}",
                top=[
                    (tokenizer.decode([int(t)]), float(p))
                    for p, t in zip(top_probs, top_ids, strict=True)
                ],
                target_prob=float(probs[target]),
                target_rank=int((probs > probs[target]).sum()) + 1,
            )
        )
    return rows
