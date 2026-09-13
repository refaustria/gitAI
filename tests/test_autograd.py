"""Gradient correctness for every primitive, checked against finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from gitai.autograd import Tensor, gradcheck
from gitai.autograd import functional as F


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _t(rng: np.random.Generator, *shape: int, positive: bool = False) -> Tensor:
    data = rng.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5  # keep log/sqrt in their domain
    return Tensor(data, requires_grad=True)


# --------------------------------------------------------------- binary ops


def test_add_grad(rng):
    gradcheck(lambda a, b: (a + b).sum(), [_t(rng, 3, 4), _t(rng, 3, 4)])


def test_mul_grad(rng):
    gradcheck(lambda a, b: (a * b).sum(), [_t(rng, 3, 4), _t(rng, 3, 4)])


def test_sub_and_div_grad(rng):
    gradcheck(lambda a, b: (a - b).sum(), [_t(rng, 2, 3), _t(rng, 2, 3)])
    gradcheck(lambda a, b: (a / b).sum(), [_t(rng, 2, 3), _t(rng, 2, 3, positive=True)])


def test_matmul_grad(rng):
    gradcheck(lambda a, b: (a @ b).sum(), [_t(rng, 3, 4), _t(rng, 4, 2)])


def test_batched_matmul_grad(rng):
    """swapaxes(-1,-2) rather than .T in the matmul backward — this is the test that
    catches getting that wrong."""
    gradcheck(lambda a, b: (a @ b).sum(), [_t(rng, 2, 3, 4), _t(rng, 2, 4, 5)])


def test_pow_grad(rng):
    gradcheck(lambda a: (a**3).sum(), [_t(rng, 3, 3)])
    gradcheck(lambda a: (a**0.5).sum(), [_t(rng, 3, 3, positive=True)])


# -------------------------------------------------------------- broadcasting


@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((3, 4), (4,)),  # bias-style broadcast, the common case
        ((3, 4), (1, 4)),
        ((3, 4), (3, 1)),
        ((2, 3, 4), (4,)),
        ((2, 1, 4), (1, 3, 1)),
        ((5,), (1,)),
    ],
)
def test_broadcast_grad(rng, shape_a, shape_b):
    """Broadcasting forward is summation backward. Getting this wrong is the
    classic hand-written-autograd bug, so every shape pairing is checked."""
    gradcheck(lambda a, b: (a * b).sum(), [_t(rng, *shape_a), _t(rng, *shape_b)])
    gradcheck(lambda a, b: (a + b).sum(), [_t(rng, *shape_a), _t(rng, *shape_b)])


def test_broadcast_grad_shape_is_preserved(rng):
    a, b = _t(rng, 3, 4), _t(rng, 4)
    (a * b).sum().backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4,)


# ----------------------------------------------------------------- unary ops


def test_exp_log_sqrt_grad(rng):
    gradcheck(lambda a: a.exp().sum(), [_t(rng, 3, 3)])
    gradcheck(lambda a: a.log().sum(), [_t(rng, 3, 3, positive=True)])
    gradcheck(lambda a: a.sqrt().sum(), [_t(rng, 3, 3, positive=True)])


def test_tanh_sigmoid_grad(rng):
    gradcheck(lambda a: a.tanh().sum(), [_t(rng, 3, 3)])
    gradcheck(lambda a: a.sigmoid().sum(), [_t(rng, 3, 3)])


def test_relu_grad(rng):
    # Shifted away from 0 because ReLU is not differentiable there and a central
    # difference straddling the kink would legitimately disagree.
    a = Tensor(rng.standard_normal((4, 4)) + 1.5, requires_grad=True)
    gradcheck(lambda x: x.relu().sum(), [a])


# ------------------------------------------------------------------ reducers


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), -1])
def test_sum_grad(rng, axis):
    gradcheck(lambda a: a.sum(axis=axis).sum(), [_t(rng, 3, 4)])


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_mean_grad(rng, axis):
    gradcheck(lambda a: a.mean(axis=axis).sum(), [_t(rng, 3, 4)])


def test_mean_value_matches_numpy(rng):
    a = _t(rng, 3, 4)
    assert np.allclose(a.mean(axis=1).data, a.data.mean(axis=1))
    assert np.allclose(a.mean().data, a.data.mean())


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_max_grad(rng, axis):
    gradcheck(lambda a: a.max(axis=axis).sum(), [_t(rng, 4, 5)])


def test_var_grad(rng):
    gradcheck(lambda a: a.var(axis=-1).sum(), [_t(rng, 3, 4)])


# --------------------------------------------------------------------- shape


def test_reshape_grad(rng):
    gradcheck(lambda a: a.reshape(4, 3).sum(), [_t(rng, 3, 4)])


def test_transpose_grad(rng):
    gradcheck(lambda a: a.T.sum(), [_t(rng, 3, 4)])
    gradcheck(lambda a: a.transpose(1, 0, 2).sum(), [_t(rng, 2, 3, 4)])


def test_getitem_grad(rng):
    gradcheck(lambda a: a[1:3].sum(), [_t(rng, 5, 4)])


def test_getitem_repeated_index_accumulates(rng):
    """An embedding row used twice must receive both gradients. This is why the
    getitem backward uses np.add.at rather than +=."""
    a = Tensor(np.arange(12, dtype=np.float64).reshape(4, 3), requires_grad=True)
    a[[0, 0, 2]].sum().backward()
    assert a.grad[0].tolist() == [2.0, 2.0, 2.0]
    assert a.grad[2].tolist() == [1.0, 1.0, 1.0]
    assert a.grad[1].tolist() == [0.0, 0.0, 0.0]


# -------------------------------------------------------------- graph shape


def test_gradient_accumulates_over_reuse(rng):
    """A tensor used twice in one expression gets gradient from both paths."""
    a = Tensor([2.0], requires_grad=True)
    (a * a).backward()
    assert np.allclose(a.grad, [4.0])  # d(a^2)/da = 2a


def test_diamond_graph(rng):
    a = _t(rng, 3)
    gradcheck(lambda x: ((x * 2.0) * (x + 1.0)).sum(), [a])


def test_deep_graph_does_not_recurse(rng):
    """Backward is iterative, not recursive. A 5000-deep graph would blow the
    Python recursion limit if it were not."""
    x = Tensor([1.0], requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 0.001
    y.backward()
    assert np.allclose(x.grad, [1.0])


def test_backward_on_nonscalar_raises(rng):
    with pytest.raises(RuntimeError, match="non-scalar"):
        _t(rng, 3, 3).backward()


# ------------------------------------------------------------------- losses


def test_log_softmax_grad(rng):
    gradcheck(lambda a: F.log_softmax(a).sum(), [_t(rng, 4, 6)])


def test_softmax_sums_to_one(rng):
    probs = F.softmax(_t(rng, 4, 6)).data
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_softmax_is_stable_at_extreme_logits():
    """Without the max-subtraction this overflows to nan."""
    probs = F.softmax(Tensor([[1000.0, 1001.0, 999.0]])).data
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(), 1.0)


def test_cross_entropy_grad(rng):
    targets = np.array([0, 3, 1, 5])
    gradcheck(lambda a: F.cross_entropy(a, targets), [_t(rng, 4, 6)])


def test_cross_entropy_matches_composed_version(rng):
    """The fast hand-derived primitive must agree with the readable composed one,
    in both value and gradient. This is what earns the right to use the fast one."""
    targets = np.array([0, 3, 1, 5])
    logits_a = _t(rng, 4, 6)
    logits_b = Tensor(logits_a.data.copy(), requires_grad=True)

    fast = F.cross_entropy(logits_a, targets)
    fast.backward()

    log_probs = F.log_softmax(logits_b)
    slow = -log_probs[np.arange(4), targets].mean()
    slow.backward()

    assert np.allclose(fast.data, slow.data)
    assert np.allclose(logits_a.grad, logits_b.grad)


def test_cross_entropy_uniform_logits_equals_log_c():
    """Uniform logits over C classes must give exactly ln(C). This is the number
    every training run should start at — if step 0 loss is not ~ln(vocab_size),
    something is wrong before training even begins."""
    loss = F.cross_entropy(Tensor(np.zeros((8, 50))), np.zeros(8, dtype=int))
    assert np.isclose(loss.item(), np.log(50))


def test_cross_entropy_rejects_bad_shapes(rng):
    with pytest.raises(ValueError, match="batch mismatch"):
        F.cross_entropy(_t(rng, 4, 6), np.array([0, 1]))


def test_mse_loss_grad(rng):
    target = rng.standard_normal((3, 4))
    gradcheck(lambda a: F.mse_loss(a, target), [_t(rng, 3, 4)])
