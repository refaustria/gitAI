"""Direct logit attribution: which component caused this prediction?

The residual stream is *additive* — every block adds its output and none
overwrite — so the final state is exactly the sum of the embedding plus every
attention and MLP output. The final normalisation is linear in its input once
its per-position scale is fixed (which it is, on any actual forward pass), so
the final logit decomposes **exactly** into one number per component.

Exactly, not approximately: ``test_attribution_sums_to_the_logit`` asserts the
parts add back to the whole. An attribution method that does not reconstruct
what it claims to explain is telling you a story.

Correlational, though. This says a component *contributed* to a logit, not that
removing it would change the answer — components can cancel, and another may
step in. For causal claims use :mod:`gitai.interpret.patching`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gitai.model.layers import RMSNorm

__all__ = ["Contribution", "attribute_heads", "attribute_logit", "final_norm_scale"]


@dataclass
class Contribution:
    name: str
    logit: float

    @property
    def kind(self) -> str:
        if self.name.startswith("embed"):
            return "embedding"
        return "attention" if ".attn" in self.name else "mlp"


def final_norm_scale(model, resid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose the final norm into ``(scale, weight, bias)`` for this input.

    Both norms are affine in the residual once the per-position scale is taken
    from the real forward pass, which is what makes the decomposition exact
    rather than a linearisation.
    """
    resid = resid.float()
    norm = model.norm_f
    if isinstance(norm, RMSNorm):
        scale = torch.rsqrt(resid.pow(2).mean(-1, keepdim=True) + norm.eps)
        return scale, norm.weight.float(), torch.zeros_like(norm.weight.float())
    # LayerNorm: the mean subtraction is linear too, so components are centred
    # individually below and the decomposition stays exact.
    scale = torch.rsqrt(resid.var(-1, keepdim=True, unbiased=False) + norm.eps)
    bias = norm.bias.float() if norm.bias is not None else torch.zeros_like(norm.weight.float())
    return scale, norm.weight.float(), bias


def _components(model, cache) -> dict[str, torch.Tensor]:
    parts = {"embed": cache["embeddings"]}
    for i in range(model.config.n_layer):
        parts[f"L{i}.attn"] = cache.layer(i, "attn_out")
        parts[f"L{i}.mlp"] = cache.layer(i, "mlp_out")
    return parts


@torch.no_grad()
def attribute_logit(
    model, cache, position: int, token_id: int, batch: int = 0
) -> tuple[list[Contribution], float]:
    """Split one logit into per-component contributions.

    Returns the contributions and the constant offset from the final norm's bias,
    which belongs to no component. Contributions plus offset equal the logit.
    """
    resid = cache.resid_post(model.config.n_layer - 1)[batch, position]
    scale, weight, bias = final_norm_scale(model, resid)
    direction = model.head.weight[token_id].float()

    is_layernorm = not isinstance(model.norm_f, RMSNorm)
    contributions: list[Contribution] = []
    for name, tensor in _components(model, cache).items():
        component = tensor[batch, position].float()
        if is_layernorm:
            component = component - component.mean()
        contributions.append(
            Contribution(name, float((component * scale.squeeze() * weight) @ direction))
        )

    offset = float(bias @ direction)
    return contributions, offset


@torch.no_grad()
def attribute_heads(model, cache, position: int, token_id: int, batch: int = 0) -> torch.Tensor:
    """Per-head contribution to one logit. Returns ``(n_layer, n_head)``.

    Attention output is ``concat(heads) @ W_O``, and that matmul is a sum over
    head-sized slices — so each head's contribution is separable exactly.

    This is the tool for finding induction heads (Decision 12, question D): a
    head whose contribution spikes when the context contains a repeat is the
    signature.
    """
    config = model.config
    resid = cache.resid_post(config.n_layer - 1)[batch, position]
    scale, weight, _ = final_norm_scale(model, resid)
    direction = model.head.weight[token_id].float()
    is_layernorm = not isinstance(model.norm_f, RMSNorm)

    out = torch.zeros(config.n_layer, config.n_head)
    for layer in range(config.n_layer):
        z = cache.layer(layer, "attn_z")[batch, position].float()
        projection = model.blocks[layer].attn.proj.weight.float()  # (d_model, d_model)
        for head in range(config.n_head):
            lo, hi = head * config.head_dim, (head + 1) * config.head_dim
            component = z[lo:hi] @ projection[:, lo:hi].T
            if is_layernorm:
                component = component - component.mean()
            out[layer, head] = float((component * scale.squeeze() * weight) @ direction)
    return out
