"""Reverse-mode automatic differentiation on NumPy.

This exists to be read, not to be fast. Every gradient in this file is written
out explicitly so that ``.backward()`` stops being magic. PyTorch does all of
this, faster, in C++ — the point is that after writing it once you can debug a
transformer instead of guessing at it.

The whole idea in three sentences:

1. Every operation records which tensors produced it (``_parents``) and a
   closure (``_backward``) that knows how to push gradient from its output back
   to its inputs.
2. ``backward()`` topologically sorts that graph and walks it in reverse, so
   every tensor's gradient is fully accumulated before it is used.
3. Gradients *accumulate* (``+=``) rather than overwrite, because a tensor used
   in several places receives gradient from each of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["Tensor", "ones", "randn", "tensor", "zeros"]


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape``, undoing NumPy broadcasting.

    Broadcasting in the forward pass is summation in the backward pass: if a
    value was reused across N positions, it receives the sum of N gradients.
    Getting this wrong is the single most common bug in a hand-written autograd
    engine, and it usually shows up as a silent shape error much later.
    """
    if grad.shape == shape:
        return grad
    # Axes that broadcasting prepended.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Axes that were size 1 and got stretched.
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _normalize_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


class Tensor:
    """An n-dimensional array that remembers how it was computed."""

    __slots__ = ("_backward", "_op", "_parents", "data", "grad", "requires_grad")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _parents: Sequence[Tensor] = (),
        _op: str = "",
        dtype=np.float64,
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._parents: tuple[Tensor, ...] = tuple(_parents)
        self._op = _op
        self._backward = lambda: None

    # ---------------------------------------------------------------- helpers

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self) -> str:
        head = f"Tensor(shape={self.shape}"
        if self._op:
            head += f", op={self._op!r}"
        if self.requires_grad:
            head += ", requires_grad=True"
        return head + ")"

    def __len__(self) -> int:
        return len(self.data)

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def detach(self) -> Tensor:
        """A tensor sharing this data but disconnected from the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def _coerce(other) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    def _child(self, data: np.ndarray, parents: Sequence[Tensor], op: str) -> Tensor:
        requires = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires, _parents=parents, _op=op)

    # ------------------------------------------------------------------- core

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Accumulate gradients of this tensor w.r.t. every ancestor."""
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    "backward() on a non-scalar requires an explicit gradient; "
                    f"this tensor has shape {self.shape}. Reduce it first (e.g. .sum())."
                )
            grad = np.ones_like(self.data)

        # Iterative post-order DFS. Deliberately not recursive: a 12-layer
        # transformer graph is deep enough to blow Python's recursion limit.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

        self.grad = self.grad + grad
        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------- binary ops

    def __add__(self, other) -> Tensor:
        other = self._coerce(other)
        out = self._child(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> Tensor:
        other = self._coerce(other)
        out = self._child(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other) -> Tensor:
        other = self._coerce(other)
        out = self._child(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            # d(A@B)/dA = grad @ B^T ; d(A@B)/dB = A^T @ grad
            # swapaxes(-1, -2) rather than .T so batched matmul works.
            self.grad += _unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.shape)
            other.grad += _unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        if not isinstance(exponent, int | float):
            raise TypeError("only scalar exponents are supported")
        out = self._child(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += out.grad * exponent * self.data ** (exponent - 1)

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        return self + (-self._coerce(other))

    def __truediv__(self, other) -> Tensor:
        return self * (self._coerce(other) ** -1.0)

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rsub__(self, other) -> Tensor:
        return self._coerce(other) + (-self)

    def __rtruediv__(self, other) -> Tensor:
        return self._coerce(other) * (self**-1.0)

    def __rmatmul__(self, other) -> Tensor:
        return self._coerce(other) @ self

    # --------------------------------------------------------------- unary ops

    def exp(self) -> Tensor:
        out = self._child(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.grad * out.data  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = self._child(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def sqrt(self) -> Tensor:
        return self**0.5

    def tanh(self) -> Tensor:
        out = self._child(np.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += out.grad * (1.0 - out.data**2)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = self._child(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._child(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- reducers

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        axes = _normalize_axes(axis, self.ndim)

        def _backward() -> None:
            g = out.grad
            if not keepdims:
                g = np.expand_dims(g, axes)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        axes = _normalize_axes(axis, self.ndim)
        n = int(np.prod([self.shape[a] for a in axes])) or 1
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self._child(self.data.max(axis=axis, keepdims=keepdims), (self,), "max")
        axes = _normalize_axes(axis, self.ndim)

        def _backward() -> None:
            g = out.grad
            peak = out.data
            if not keepdims:
                g = np.expand_dims(g, axes)
                peak = np.expand_dims(peak, axes)
            mask = (self.data == peak).astype(self.data.dtype)
            # Split gradient evenly across ties, which is what a finite-difference
            # check sees and keeps gradcheck honest.
            mask /= mask.sum(axis=axes, keepdims=True)
            self.grad += mask * g

        out._backward = _backward
        return out

    def var(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        mu = self.mean(axis=axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------- shape

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        original = self.shape
        out = self._child(self.data.reshape(shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(original)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        order = axes if axes else tuple(reversed(range(self.ndim)))
        out = self._child(self.data.transpose(order), (self,), "transpose")
        inverse = np.argsort(order)

        def _backward() -> None:
            self.grad += out.grad.transpose(inverse)

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, index) -> Tensor:
        out = self._child(self.data[index], (self,), "getitem")

        def _backward() -> None:
            # np.add.at, not `+=`, because repeated indices must accumulate
            # rather than overwrite (e.g. an embedding row used twice).
            np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out


# ------------------------------------------------------------------ factories


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(
    *shape: int, requires_grad: bool = False, rng: np.random.Generator | None = None
) -> Tensor:
    rng = rng or np.random.default_rng()
    return Tensor(rng.standard_normal(shape), requires_grad=requires_grad)
