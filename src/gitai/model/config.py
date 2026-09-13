"""Model configuration, and the ladder of architectural rungs.

The ladder from the roadmap is expressed as configuration rather than as seven
separate classes. Each rung is one flag away from the previous one, which makes
"did this change help?" a controlled experiment instead of a diff review, and
keeps a single forward pass to debug rather than seven.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

__all__ = ["RUNGS", "ModelConfig"]

NormKind = Literal["layernorm", "rmsnorm"]
PositionKind = Literal["none", "learned", "rope"]
MLPKind = Literal["gelu", "swiglu"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    seq_len: int = 256
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8

    norm: NormKind = "rmsnorm"
    position: PositionKind = "rope"
    mlp: MLPKind = "swiglu"
    bias: bool = False
    tie_embeddings: bool = True

    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_base: float = 10_000.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_head {self.n_head}")
        if self.position == "rope" and self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim; got {self.head_dim}")
        if self.vocab_size < 1 or self.n_layer < 1:
            raise ValueError("vocab_size and n_layer must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def hidden_dim(self) -> int:
        """MLP inner width.

        SwiGLU has three weight matrices where a GELU MLP has two, so the naive
        4x ratio would make it 1.5x larger for free and quietly invalidate any
        comparison. The 8/3 ratio restores parameter parity, which is why the
        SwiGLU papers use it.
        """
        if self.mlp == "swiglu":
            return int(self.mlp_ratio * 2 / 3 * self.d_model)
        return int(self.mlp_ratio * self.d_model)

    def with_(self, **changes) -> ModelConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    # -------------------------------------------------------- parameter count

    def parameter_count(self) -> dict[str, int]:
        """Hand-derived, so the test suite can catch a silent architecture change.

        Embedding and non-embedding counts are reported separately because at
        this scale they differ enormously: comparing a "10M model" that spends
        1.5M on embeddings against one that spends 6M is comparing nothing.
        """
        d, v = self.d_model, self.vocab_size
        norm_params = d if self.norm == "rmsnorm" else 2 * d
        bias = 1 if self.bias else 0

        embeddings = v * d
        if self.position == "learned":
            embeddings += self.seq_len * d

        attention = 3 * d * d + 3 * d * bias  # q, k, v
        attention += d * d + d * bias  # output projection

        if self.mlp == "swiglu":
            mlp = 3 * d * self.hidden_dim + (2 * self.hidden_dim + d) * bias
        else:
            mlp = 2 * d * self.hidden_dim + (self.hidden_dim + d) * bias

        per_layer = 2 * norm_params + attention + mlp
        non_embedding = self.n_layer * per_layer + norm_params
        head = 0 if self.tie_embeddings else d * v

        return {
            "embedding": embeddings,
            "non_embedding": non_embedding + head,
            "total": embeddings + non_embedding + head,
            "per_layer": per_layer,
        }

    def summary(self) -> str:
        counts = self.parameter_count()
        return (
            f"{self.n_layer}L x {self.d_model}d x {self.n_head}h "
            f"({self.head_dim} per head), seq {self.seq_len}, vocab {self.vocab_size}\n"
            f"  {self.norm} / {self.position} / {self.mlp} / "
            f"bias={self.bias} / tied={self.tie_embeddings}\n"
            f"  {counts['total']:,} parameters "
            f"({counts['non_embedding']:,} non-embedding, "
            f"{counts['embedding']:,} embedding)"
        )


def _rung(**overrides) -> dict:
    return overrides


# The roadmap's ladder. Climb it in order; tag each rung in git. The point is
# that every step is independently debuggable and independently measurable.
RUNGS: dict[str, dict] = {
    # v0 (bigram) is a separate model class, not a config — see bigram.py
    "v1_single_head": _rung(
        n_head=1, norm="layernorm", position="learned", mlp="gelu", bias=True, tie_embeddings=False
    ),
    "v2_gpt2": _rung(
        norm="layernorm", position="learned", mlp="gelu", bias=True, tie_embeddings=False
    ),
    "v3_rmsnorm": _rung(
        norm="rmsnorm", position="learned", mlp="gelu", bias=True, tie_embeddings=False
    ),
    "v4_rope": _rung(norm="rmsnorm", position="rope", mlp="gelu", bias=True, tie_embeddings=False),
    "v5_swiglu": _rung(
        norm="rmsnorm", position="rope", mlp="swiglu", bias=True, tie_embeddings=False
    ),
    "v6_modern": _rung(
        norm="rmsnorm", position="rope", mlp="swiglu", bias=False, tie_embeddings=True
    ),
}
