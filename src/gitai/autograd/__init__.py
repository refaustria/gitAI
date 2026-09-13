"""A small reverse-mode autograd engine on NumPy.

Phase 0 of the project: understand backpropagation by writing it, before
relying on PyTorch for everything afterwards. Deliberately depends on numpy
alone — it must not be able to lean on a framework.
"""

from . import functional, nn, optim
from .functional import cross_entropy, log_softmax, mse_loss, softmax
from .gradcheck import gradcheck, max_relative_error
from .tensor import Tensor, ones, randn, tensor, zeros

__all__ = [
    "Tensor",
    "cross_entropy",
    "functional",
    "gradcheck",
    "log_softmax",
    "max_relative_error",
    "mse_loss",
    "nn",
    "ones",
    "optim",
    "randn",
    "softmax",
    "tensor",
    "zeros",
]
