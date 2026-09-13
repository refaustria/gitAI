#!/usr/bin/env python3
"""Train XOR with the hand-written engine. The 'it actually works' demo.

uv run python scripts/demo_autograd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.autograd import Tensor, nn, optim
from gitai.autograd import functional as F


def main() -> None:
    rng = np.random.default_rng(1)
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    model = nn.Sequential(nn.Linear(2, 16, rng=rng), nn.Tanh(), nn.Linear(16, 2, rng=rng))
    opt = optim.AdamW(model.parameters(), lr=0.05)

    print(f"XOR with a {model.num_parameters()}-parameter MLP, no PyTorch anywhere.")
    print(f"step 0 loss should be about ln(2) = {np.log(2):.4f}\n")

    for step in range(601):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:>4}   loss {loss.item():.6f}")

    predictions = model(X).data.argmax(axis=-1)
    print("\n  a  b  | target  predicted")
    for (a, b), target, pred in zip(X.data.astype(int), y, predictions, strict=True):
        print(f"  {a}  {b}  |   {target}         {pred}")
    print("\nsolved" if (predictions == y).all() else "\nFAILED")


if __name__ == "__main__":
    main()
