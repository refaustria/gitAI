"""The probe interface.

Loss is a summary statistic and it hides structure: two models at identical loss
can have very different abilities. Probes ask specific questions with
unambiguous answers.

Two kinds live here, and the difference matters:

- **Behavioural probes** (:mod:`~gitai.eval.probes.induction`) run on *any*
  trained model, including one trained on natural text. They measure a capability
  the model may have picked up incidentally.
- **Synthetic tasks** (:mod:`~gitai.eval.probes.synthetic`) generate their own
  data from a process you define exactly. A model has to be *trained* on them to
  be scored — but because you know the true generating function, you can ask
  whether it learned the algorithm or memorised a lookup table. That question is
  unanswerable on natural text, and it is nearly free in compute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Probe", "SyntheticTask"]


class Probe(ABC):
    """Something that can be measured on an already-trained model."""

    name: str

    @abstractmethod
    def run(self, model, **kwargs) -> dict[str, float]:
        """Return a flat dict of metric name -> value."""


class SyntheticTask(ABC):
    """A task with a known generating process.

    Provides a corpus to train on and an exact-match evaluation. The held-out set
    is generated with a different seed, which makes it held-out *by construction*
    rather than by splitting.

    A caveat that is itself interesting: for small task spaces (modular
    arithmetic with a small modulus, say) the evaluation samples will inevitably
    overlap the training ones, because the space is exhaustible. That is not a
    leak to fix — it is precisely the setting in which "did it learn the
    algorithm or memorise the table?" becomes a real, answerable question. Vary
    the space size to find out.
    """

    name: str

    @abstractmethod
    def sample(self, rng) -> tuple[str, str]:
        """One example as ``(prompt, answer)``. The answer is what gets scored."""

    def corpus(self, n: int, seed: int = 0) -> list[str]:
        import numpy as np

        rng = np.random.default_rng(seed)
        return [prompt + answer for prompt, answer in (self.sample(rng) for _ in range(n))]

    def examples(self, n: int, seed: int) -> list[tuple[str, str]]:
        import numpy as np

        rng = np.random.default_rng(seed)
        return [self.sample(rng) for _ in range(n)]
