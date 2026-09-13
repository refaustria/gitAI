"""Looking inside the model.

A model this size will not produce a reasoning trace — it predicts next tokens,
it does not deliberate. What it offers instead is better: at 1-50M parameters
the model is small enough to inspect *completely*, which is why mechanistic
interpretability research uses models this size. The constraint is the field's
preferred experimental condition.

Four levels, answering different questions:

- :mod:`~gitai.interpret.surprisal` — where was it confident, where surprised?
- :mod:`~gitai.interpret.lens` — how did the prediction form across depth?
- :mod:`~gitai.interpret.attribution` — which component contributed this logit?
- :mod:`~gitai.interpret.patching` — which component *caused* it? (causal)

The first three are correlational; only patching establishes that something
mattered. Prefer it whenever the claim matters.
"""

from .attribution import Contribution, attribute_heads, attribute_logit
from .lens import LensRow, lens_trajectory, logit_lens
from .patching import ablate_heads, logit_difference, loss_metric, patch_residual
from .render import (
    ablation_table,
    attention_heatmap,
    attribution_table,
    colour_by_surprisal,
    lens_table,
    visible,
)
from .surprisal import TokenPrediction, bits_per_byte, predict_tokens, surprisal_bits

__all__ = [
    "Contribution",
    "LensRow",
    "TokenPrediction",
    "ablate_heads",
    "ablation_table",
    "attention_heatmap",
    "attribute_heads",
    "attribute_logit",
    "attribution_table",
    "bits_per_byte",
    "colour_by_surprisal",
    "lens_table",
    "lens_trajectory",
    "logit_difference",
    "logit_lens",
    "loss_metric",
    "patch_residual",
    "predict_tokens",
    "surprisal_bits",
    "visible",
]
