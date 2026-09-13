"""Differentiable functions built on :class:`~gitai.autograd.tensor.Tensor`.

Two of these are written twice on purpose. ``log_softmax`` is *composed* from
primitive ops, so you can read the maths; ``cross_entropy`` is a *primitive*
with a hand-derived gradient, because composing it is numerically worse and
about four times slower. The test suite checks the two agree, which is the
honest way to earn the right to use the fast one.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["cross_entropy", "log_softmax", "mse_loss", "relu", "sigmoid", "softmax", "tanh"]


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Composed from primitives. Subtracting the max is what keeps exp() finite."""
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis=axis).exp()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean cross-entropy for ``logits`` of shape (N, C) and integer ``targets`` (N,).

    The gradient of softmax-cross-entropy collapses to ``softmax(x) - onehot(y)``,
    which is both simpler and better-conditioned than differentiating through the
    log and the exp separately. This is the single most worthwhile hand-derived
    gradient in a language model: it runs on every token of every step.
    """
    t = np.asarray(targets, dtype=np.int64)
    if t.ndim != 1 or logits.ndim != 2:
        raise ValueError(
            f"expected logits (N, C) and targets (N,), got {logits.shape} and {t.shape}"
        )
    if t.shape[0] != logits.shape[0]:
        raise ValueError(f"batch mismatch: {logits.shape[0]} logits vs {t.shape[0]} targets")

    x = logits.data
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    partition = exp.sum(axis=-1, keepdims=True)
    log_probs = shifted - np.log(partition)

    n = t.shape[0]
    rows = np.arange(n)
    loss = -log_probs[rows, t].mean()

    out = logits._child(np.asarray(loss), (logits,), "cross_entropy")

    def _backward() -> None:
        grad = exp / partition  # softmax(x)
        grad[rows, t] -= 1.0  # minus the one-hot target
        grad /= n  # because the forward took a mean
        logits.grad += grad * out.grad

    out._backward = _backward
    return out


def mse_loss(prediction: Tensor, target) -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target)
    return ((prediction - target) ** 2).mean()
