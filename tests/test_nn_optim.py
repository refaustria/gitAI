"""Layers, optimisers, and the one test that matters most: does it learn?"""

from __future__ import annotations

import numpy as np
import pytest

from gitai.autograd import Tensor, gradcheck, nn, optim
from gitai.autograd import functional as F


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --------------------------------------------------------------------- layers


def test_linear_shapes_and_count(rng):
    layer = nn.Linear(4, 3, rng=rng)
    assert layer(Tensor(rng.standard_normal((7, 4)))).shape == (7, 3)
    assert layer.num_parameters() == 4 * 3 + 3


def test_linear_without_bias(rng):
    layer = nn.Linear(4, 3, bias=False, rng=rng)
    assert layer.bias is None
    assert layer.num_parameters() == 12


def test_linear_grad(rng):
    layer = nn.Linear(4, 3, rng=rng)
    x = Tensor(rng.standard_normal((5, 4)), requires_grad=True)
    gradcheck(lambda w, b, inp: (inp @ w + b).sum(), [layer.weight, layer.bias, x])


def test_sequential_collects_parameters_without_a_manual_list(rng):
    model = nn.Sequential(nn.Linear(4, 8, rng=rng), nn.ReLU(), nn.Linear(8, 2, rng=rng))
    assert len(model.parameters()) == 4
    assert model.num_parameters() == (4 * 8 + 8) + (8 * 2 + 2)


def test_parameters_are_deduplicated(rng):
    """Weight tying: the same tensor appearing twice must be returned once, or
    the optimiser would apply its update twice."""
    shared = nn.Linear(4, 4, rng=rng)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    assert len(model.parameters()) == 2


def test_zero_grad_clears(rng):
    model = nn.Linear(3, 2, rng=rng)
    F.mse_loss(model(Tensor(rng.standard_normal((5, 3)))), np.zeros((5, 2))).backward()
    assert np.abs(model.weight.grad).sum() > 0
    model.zero_grad()
    assert np.abs(model.weight.grad).sum() == 0


# ----------------------------------------------------------------- optimisers


def _quadratic_descent(optimizer_cls, steps: int = 200, **kwargs) -> float:
    """Minimise (x-3)^2 from x=0. Every optimiser should get close to 3."""
    x = Tensor([0.0], requires_grad=True)
    opt = optimizer_cls([x], **kwargs)
    for _ in range(steps):
        opt.zero_grad()
        ((x - 3.0) ** 2).sum().backward()
        opt.step()
    return float(x.data[0])


def test_sgd_converges():
    assert abs(_quadratic_descent(optim.SGD, lr=0.1) - 3.0) < 1e-3


def test_sgd_momentum_converges():
    assert abs(_quadratic_descent(optim.SGD, lr=0.05, momentum=0.9) - 3.0) < 1e-3


def test_adam_converges():
    assert abs(_quadratic_descent(optim.Adam, steps=800, lr=0.1) - 3.0) < 1e-3


def test_adamw_converges():
    assert abs(_quadratic_descent(optim.AdamW, steps=800, lr=0.1) - 3.0) < 1e-3


def test_adam_bias_correction_makes_first_step_full_size():
    """With bias correction Adam's first step is ~lr regardless of gradient
    magnitude. Without it the step would be ~lr*(1-beta1) = 10x smaller, which
    is a silent, slow-training bug rather than a crash."""
    x = Tensor([5.0], requires_grad=True)
    opt = optim.Adam([x], lr=0.1)
    opt.zero_grad()
    (x * 1.0).sum().backward()
    opt.step()
    assert np.isclose(5.0 - float(x.data[0]), 0.1, atol=1e-3)


def test_adamw_decouples_weight_decay():
    """With zero gradient, AdamW still decays; plain Adam with weight_decay
    routes decay through the adaptive term instead. They must differ."""
    x_w = Tensor([1.0], requires_grad=True)
    x_a = Tensor([1.0], requires_grad=True)
    opt_w = optim.AdamW([x_w], lr=0.1, weight_decay=0.5)
    opt_a = optim.Adam([x_a], lr=0.1, weight_decay=0.5)
    for opt, x in ((opt_w, x_w), (opt_a, x_a)):
        opt.zero_grad()
        (x * 0.0).sum().backward()
        opt.step()
    assert not np.isclose(x_w.data[0], x_a.data[0])
    assert np.isclose(x_w.data[0], 1.0 - 0.1 * 0.5)  # exactly lr * wd * w


def test_optimizer_rejects_empty_params():
    with pytest.raises(ValueError, match="empty parameter list"):
        optim.SGD([])


# ------------------------------------------------------------------- learning


def test_mlp_learns_xor():
    """XOR is not linearly separable, so a network that solves it has genuinely
    learned a nonlinear function rather than fitting a line. If the engine has a
    gradient bug this test fails and almost nothing else will."""
    rng = np.random.default_rng(1)
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    model = nn.Sequential(nn.Linear(2, 16, rng=rng), nn.Tanh(), nn.Linear(16, 2, rng=rng))
    opt = optim.AdamW(model.parameters(), lr=0.05)

    for _ in range(600):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.01
    assert (model(X).data.argmax(axis=-1) == y).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_overfit_a_single_batch(seed):
    """The single highest-value test in machine learning: a correct model with a
    correct optimiser must be able to memorise a small batch of *random* labels.
    If it cannot, there is a bug, and this finds it in under a second.

    This test is repeated in Phase 2 against the real transformer.
    """
    rng = np.random.default_rng(seed)
    X = Tensor(rng.standard_normal((32, 8)))
    y = rng.integers(0, 4, size=32)

    model = nn.Sequential(nn.Linear(8, 64, rng=rng), nn.ReLU(), nn.Linear(64, 4, rng=rng))
    opt = optim.AdamW(model.parameters(), lr=0.02)

    for _ in range(500):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.01, f"failed to overfit 32 examples: loss {loss.item():.4f}"


def _seeded_training_run(seed: int = 42, steps: int = 20) -> list[float]:
    rng = np.random.default_rng(seed)
    X = Tensor(rng.standard_normal((16, 4)))
    y = rng.integers(0, 3, size=16)
    model = nn.Sequential(nn.Linear(4, 8, rng=rng), nn.ReLU(), nn.Linear(8, 3, rng=rng))
    opt = optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_training_is_deterministic_given_a_seed():
    """Same seed, same loss curve. Without this, ablation results are noise and
    you cannot tell an improvement from a lucky init.

    **Not asserted bit-for-bit, and the reason matters.** On an Intel Mac two
    identically seeded runs of this function landed one ULP apart (3.6e-16
    relative); on Linux against OpenBLAS they are exactly equal. The mechanism
    is an unidentified floating-point-associativity difference in the platform's
    linear algebra — see docs/evaluation.md, which also records why the obvious
    Accelerate explanation is probably wrong.

    A tolerance of 1e-9 sits seven orders of magnitude above that noise
    (~4e-16 relative) and seven below what a genuine determinism bug produces
    (~1e-1 relative, measured against a different seed). The test therefore
    still fails loudly on unseeded RNG or shared state, which is what it is for
    — ``test_the_determinism_check_still_catches_real_nondeterminism`` proves it.
    """
    first, second = _seeded_training_run(), _seeded_training_run()
    assert first == pytest.approx(second, rel=1e-9)


def test_the_determinism_check_still_catches_real_nondeterminism():
    """A negative control for the relaxed tolerance above.

    Loosening an assertion is only safe if it still fails on the thing it was
    written to catch. Two different seeds must be far outside 1e-9.
    """
    with pytest.raises(AssertionError):
        assert _seeded_training_run(seed=42) == pytest.approx(
            _seeded_training_run(seed=43), rel=1e-9
        )
