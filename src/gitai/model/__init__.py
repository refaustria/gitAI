"""The transformer, instrumented from the first rung.

Every intermediate is optionally observable (see :mod:`gitai.model.cache`),
because instrumentation is free to design in and painful to retrofit — and
because at this scale the model is small enough to understand completely, which
is the main advantage of working here rather than at frontier scale.
"""

from .attention import CausalSelfAttention
from .bigram import BigramModel
from .cache import ActivationCache
from .config import RUNGS, ModelConfig
from .kvcache import KVCache
from .layers import MLP, RMSNorm, RotaryEmbedding, SwiGLU, apply_rope
from .transformer import Block, Transformer

__all__ = [
    "MLP",
    "RUNGS",
    "ActivationCache",
    "BigramModel",
    "Block",
    "CausalSelfAttention",
    "KVCache",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "Transformer",
    "apply_rope",
]
