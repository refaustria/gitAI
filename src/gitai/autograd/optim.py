"""Optimisers, written from the papers rather than from memory.

- SGD with (Nesterov) momentum
- Adam — Kingma & Ba, 2015
- AdamW — Loshchilov & Hutter, 2019

The Adam/AdamW difference is small in code and large in effect, so it is spelled
out explicitly below rather than hidden behind a flag.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["SGD", "Adam", "AdamW", "Optimizer"]


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimiser received an empty parameter list")
        self.lr = lr
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.step_count += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                g = g + self.momentum * self.velocity[i] if self.nesterov else self.velocity[i]
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam with L2 regularisation folded into the gradient (the original form)."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            # Bias correction. Without it the first steps are badly
            # under-scaled, because m and v start at zero.
            m_hat = self.m[i] / (1.0 - self.beta1**t)
            v_hat = self.v[i] / (1.0 - self.beta2**t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with *decoupled* weight decay.

    The only change from :class:`Adam` is where decay is applied. In Adam it
    enters the gradient, so it is scaled by the adaptive term ``1/sqrt(v)`` —
    parameters with small gradients get decayed far more than intended. AdamW
    applies it directly to the weights instead, which is what "weight decay"
    always meant. Use this one.
    """

    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        for i, p in enumerate(self.params):
            g = p.grad

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - self.beta1**t)
            v_hat = self.v[i] / (1.0 - self.beta2**t)

            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
