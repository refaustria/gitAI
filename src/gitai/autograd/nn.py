"""Minimal module system: parameters, layers, and composition."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .functional import relu, sigmoid, tanh
from .tensor import Tensor

__all__ = ["Linear", "Module", "ReLU", "Sequential", "Sigmoid", "Tanh"]


class Module:
    """Base class. Anything assigned to ``self`` that holds parameters is found
    automatically, so subclasses never maintain a parameter list by hand."""

    def parameters(self) -> list[Tensor]:
        seen: dict[int, Tensor] = {}
        for value in self._children():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    seen[id(value)] = value
            elif isinstance(value, Module):
                for p in value.parameters():
                    seen[id(p)] = p
        return list(seen.values())

    def _children(self) -> Iterator[object]:
        for value in vars(self).values():
            if isinstance(value, list | tuple):
                yield from value
            else:
                yield value

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """``y = x @ W + b``.

    Default init is Kaiming-uniform on fan-in, the sane choice for ReLU networks:
    it keeps activation variance roughly constant with depth. Initialisation is
    not a detail — a bad scheme makes a deep network untrainable in a way that
    looks exactly like a bug in the optimiser.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        bound = np.sqrt(1.0 / in_features)
        self.weight = Tensor(
            rng.uniform(-bound, bound, size=(in_features, out_features)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
