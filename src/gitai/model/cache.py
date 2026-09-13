"""Activation capture.

Every intermediate the forward pass computes, optionally kept.

Instrumentation is painful to retrofit and free to design in, so it is here from
the first rung rather than bolted on once something interesting happens. Because
we wrote the model ourselves there is no need for hooks or monkey-patching — the
forward pass simply hands over what it already has. When no cache is passed the
cost is a handful of ``is not None`` checks.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

__all__ = ["ActivationCache"]


class ActivationCache:
    """Keyed store of intermediate tensors from one forward pass.

    Keys follow ``blocks.<i>.<name>`` for per-layer values and bare names for
    model-level ones. Everything is detached by default: keeping the graph alive
    for a whole forward pass is a memory leak in disguise, and the analyses that
    need gradients ask for them explicitly with ``keep_grad=True``.
    """

    def __init__(self, keep_grad: bool = False) -> None:
        self.keep_grad = keep_grad
        self._store: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ write

    def put(self, key: str, value: torch.Tensor) -> None:
        self._store[key] = value if self.keep_grad else value.detach()

    def put_layer(self, layer: int, name: str, value: torch.Tensor) -> None:
        self.put(f"blocks.{layer}.{name}", value)

    # ------------------------------------------------------------------- read

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._store:
            raise KeyError(f"{key!r} was not cached; available: {sorted(self._store)}")
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def keys(self) -> list[str]:
        return sorted(self._store)

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    # -------------------------------------------------------- typed accessors

    @property
    def n_layer(self) -> int:
        layers = {int(k.split(".")[1]) for k in self._store if k.startswith("blocks.")}
        return max(layers) + 1 if layers else 0

    def layer(self, i: int, name: str) -> torch.Tensor:
        return self[f"blocks.{i}.{name}"]

    def attention(self, layer: int) -> torch.Tensor:
        """Attention weights, ``(batch, head, query, key)``.

        A caveat worth carrying: attention weights are suggestive, not
        explanatory. A head attending to a token does not prove that token
        caused the output — for that, use ``interpret.patching``.
        """
        return self.layer(layer, "attn_pattern")

    def resid_post(self, layer: int) -> torch.Tensor:
        return self.layer(layer, "resid_post")

    def residual_stream(self) -> torch.Tensor:
        """Every residual-stream state stacked: ``(n_layer + 1, batch, seq, d_model)``.

        Index 0 is the embedding output, index *i+1* is the output of block *i*.
        This is what the logit lens reads to show a prediction assembling across
        depth.
        """
        states = [self["embeddings"]]
        states += [self.resid_post(i) for i in range(self.n_layer)]
        return torch.stack(states)

    def summary(self) -> str:
        lines = [f"ActivationCache: {len(self._store)} tensors"]
        for key in self.keys():
            tensor = self._store[key]
            lines.append(f"  {key:<28} {tuple(tensor.shape)}  {tensor.dtype}")
        return "\n".join(lines)
