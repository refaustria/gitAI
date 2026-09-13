"""Synthetic tasks with known generating processes.

The point of these is control. On natural text you can measure that a model got
something right; you cannot tell whether it learned a rule or memorised a
pattern. Here you wrote the rule, so you can ask directly — vary the size of the
task space and watch whether accuracy survives when memorisation stops being
possible.

They are also nearly free: the data is generated on demand, and models small
enough to train in minutes can genuinely solve several of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import SyntheticTask

__all__ = ["TASKS", "CopyTask", "DyckTask", "ModularArithmeticTask", "SortTask", "evaluate_task"]


@dataclass
class CopyTask(SyntheticTask):
    """``a b c | a b c`` — reproduce a sequence. The simplest induction test."""

    length: int = 8
    alphabet: int = 20
    name: str = "copy"

    def sample(self, rng) -> tuple[str, str]:
        items = rng.integers(0, self.alphabet, size=self.length)
        body = " ".join(str(int(x)) for x in items)
        return f"{body} | ", f"{body}\n"


@dataclass
class SortTask(SyntheticTask):
    """``5 1 4 -> 1 4 5`` — sort a list. Tests algorithmic generalisation."""

    length: int = 6
    maximum: int = 20
    name: str = "sort"

    def sample(self, rng) -> tuple[str, str]:
        items = rng.integers(0, self.maximum, size=self.length)
        prompt = " ".join(str(int(x)) for x in items)
        answer = " ".join(str(int(x)) for x in sorted(items))
        return f"{prompt} -> ", f"{answer}\n"


@dataclass
class ModularArithmeticTask(SyntheticTask):
    """``7 + 5 mod 13 = 12``.

    The classic memorisation-versus-algorithm probe. With modulus *m* there are
    only *m²* distinct problems, so a model with enough capacity can simply store
    the table. Shrink the training set or grow *m* and watch whether accuracy
    holds — that gap is the interesting measurement, not the accuracy itself.
    """

    modulus: int = 13
    operation: str = "+"
    name: str = "modular_arithmetic"

    def sample(self, rng) -> tuple[str, str]:
        a, b = int(rng.integers(0, self.modulus)), int(rng.integers(0, self.modulus))
        result = (a + b if self.operation == "+" else a * b) % self.modulus
        return f"{a} {self.operation} {b} mod {self.modulus} = ", f"{result}\n"


@dataclass
class DyckTask(SyntheticTask):
    """Balanced brackets — ``([]()) `` — testing recursive structure.

    The prompt is a prefix with unclosed brackets; the answer closes them in the
    right order. Requires maintaining a stack, which a fixed-depth transformer
    can only do up to a bounded depth. Where that bound sits, as a function of
    depth and width, is a real question.
    """

    max_depth: int = 4
    length: int = 12
    name: str = "dyck"

    PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))

    def sample(self, rng) -> tuple[str, str]:
        stack: list[str] = []
        prefix: list[str] = []
        for _ in range(self.length):
            can_open = len(stack) < self.max_depth
            can_close = bool(stack)
            if can_open and (not can_close or rng.random() < 0.6):
                opener, closer = self.PAIRS[int(rng.integers(0, len(self.PAIRS)))]
                prefix.append(opener)
                stack.append(closer)
            elif can_close:
                prefix.append(stack.pop())
        answer = "".join(reversed(stack))
        return "".join(prefix) + " ", f"{answer}\n"


TASKS = {
    "copy": CopyTask,
    "sort": SortTask,
    "modular_arithmetic": ModularArithmeticTask,
    "dyck": DyckTask,
}


@torch.no_grad()
def evaluate_task(
    task: SyntheticTask, model, tokenizer, n: int = 100, seed: int = 9999, max_new: int = 32
) -> dict[str, float]:
    """Exact-match accuracy by greedy decoding.

    Greedy, not sampled: the question is what the model believes, and sampling
    would add variance to a number meant for comparison. Generation stops at the
    newline that terminates every answer.
    """
    examples = task.examples(n, seed=seed)
    was_training = model.training
    model.eval()

    exact = 0
    prefix_hits = 0
    try:
        for prompt, answer in examples:
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            budget = min(max_new, len(tokenizer.encode(answer)) + 2)
            generated = model.generate(ids, budget, temperature=0.0)
            produced = tokenizer.decode(generated[0, ids.shape[1] :].tolist())
            produced = produced.split("\n")[0].strip()
            expected = answer.strip()
            if produced == expected:
                exact += 1
            if produced[: len(expected)] == expected[: len(produced)] and produced:
                prefix_hits += 1
    finally:
        model.train(was_training)

    return {
        f"{task.name}_exact_match": exact / n,
        f"{task.name}_prefix_agreement": prefix_hits / n,
    }
