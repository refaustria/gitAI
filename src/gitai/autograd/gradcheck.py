"""Finite-difference gradient checking.

The ground truth for an autograd engine. If the analytic gradient disagrees with
a central difference, the analytic gradient is wrong — this check has no opinions
and no bugs to speak of, only a tolerance.

Central differences, ``(f(x+h) - f(x-h)) / 2h``, have error O(h^2) rather than
the O(h) of a forward difference. That is why everything here is float64: at
float32 the subtraction loses more precision than the smaller step size buys.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["gradcheck", "max_relative_error"]


def max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error, with a floor so that near-zero gradients don't blow up."""
    denominator = np.maximum(1e-8, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denominator))


def gradcheck(fn, inputs: list[Tensor], eps: float = 1e-6, tol: float = 1e-6) -> float:
    """Compare analytic against numeric gradients of ``fn(*inputs)``.

    ``fn`` must return a scalar Tensor (or one that is summed to a scalar).
    Returns the worst relative error, and raises AssertionError if it exceeds
    ``tol``.
    """
    for t in inputs:
        t.zero_grad()

    out = fn(*inputs)
    if out.size != 1:
        out = out.sum()
    out.backward()
    analytic = [t.grad.copy() for t in inputs]

    worst = 0.0
    for k, t in enumerate(inputs):
        numeric = np.zeros_like(t.data)
        for idx in np.ndindex(t.data.shape):
            original = t.data[idx]

            t.data[idx] = original + eps
            plus = fn(*inputs)
            plus = plus.sum() if plus.size != 1 else plus

            t.data[idx] = original - eps
            minus = fn(*inputs)
            minus = minus.sum() if minus.size != 1 else minus

            t.data[idx] = original
            numeric[idx] = (plus.item() - minus.item()) / (2.0 * eps)

        error = max_relative_error(analytic[k], numeric)
        worst = max(worst, error)
        if error > tol:
            raise AssertionError(
                f"gradcheck failed for input {k} (shape {t.shape}): "
                f"relative error {error:.3e} > tol {tol:.3e}\n"
                f"analytic:\n{analytic[k]}\nnumeric:\n{numeric}"
            )
    return worst
