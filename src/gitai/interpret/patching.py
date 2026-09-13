"""Causal interventions: ablation and activation patching.

Everything else in this package is correlational — it reports what the model
computed. These two change the computation and measure what breaks, which is the
only way to establish that a component *mattered*.

Activation patching is the sharper of the two. Run the model on a clean input,
run it on a corrupted one, then splice a single clean activation into the
corrupted run. If the output recovers, that activation carried the information.
Sweeping over every (layer, position) produces a map of where in the network the
relevant computation actually happens.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

__all__ = ["ablate_heads", "logit_difference", "loss_metric", "patch_residual"]

Metric = Callable[[torch.Tensor], float]


def logit_difference(correct: int, incorrect: int, position: int = -1) -> Metric:
    """Metric: how much more the model favours one token over another.

    Better than raw loss for patching, because it isolates a single decision and
    is unaffected by the rest of the distribution shifting around.
    """

    def metric(logits: torch.Tensor) -> float:
        row = logits[0, position].float()
        return float(row[correct] - row[incorrect])

    return metric


def loss_metric(targets: torch.Tensor) -> Metric:
    def metric(logits: torch.Tensor) -> float:
        return float(
            F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        )

    return metric


@torch.no_grad()
def ablate_heads(model, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Zero each attention head in turn; return the loss increase. ``(n_layer, n_head)``.

    A large positive value means the head was carrying its weight. A value near
    zero means it was not — at this scale, plenty of heads are genuinely doing
    nothing, and knowing which is useful before you go looking for a story in
    their attention patterns.

    Note the limitation: ablating one head at a time misses redundancy. Two heads
    computing the same thing will both look useless.
    """
    config = model.config
    was_training = model.training
    model.eval()
    try:
        full = torch.ones(config.n_layer, config.n_head)
        baseline = float(model(ids, targets, head_mask=full)[1])

        deltas = torch.zeros(config.n_layer, config.n_head)
        for layer in range(config.n_layer):
            for head in range(config.n_head):
                mask = torch.ones(config.n_layer, config.n_head)
                mask[layer, head] = 0.0
                deltas[layer, head] = float(model(ids, targets, head_mask=mask)[1]) - baseline
        return deltas
    finally:
        model.train(was_training)


@torch.no_grad()
def patch_residual(
    model,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    metric: Metric,
    positions: list[int] | None = None,
) -> torch.Tensor:
    """Patch clean activations into a corrupted run, one site at a time.

    Returns ``(n_layer, len(positions))`` of the metric after patching each
    residual-stream site. Compare against the clean and corrupt baselines the
    caller computes: a patched value near the clean baseline means that site was
    sufficient to restore the behaviour.

    Implemented with forward hooks rather than by threading a parameter through
    the model, so the model stays free of analysis machinery.
    """
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(
            f"clean and corrupt inputs must align position-for-position; "
            f"got {tuple(clean_ids.shape)} and {tuple(corrupt_ids.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        _, _, clean_cache = model.run_with_cache(clean_ids)
        seq = clean_ids.shape[1]
        sites = list(range(seq)) if positions is None else positions

        results = torch.zeros(model.config.n_layer, len(sites))
        for layer in range(model.config.n_layer):
            donor = clean_cache.resid_post(layer)
            for column, position in enumerate(sites):
                replacement = donor[:, position].clone()

                def hook(_module, _args, output, pos=position, value=replacement):
                    patched = output.clone()
                    patched[:, pos] = value
                    return patched

                handle = model.blocks[layer].register_forward_hook(hook)
                try:
                    logits, _ = model(corrupt_ids)
                    results[layer, column] = metric(logits)
                finally:
                    handle.remove()
        return results
    finally:
        model.train(was_training)
