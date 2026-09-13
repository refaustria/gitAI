"""The evaluation suite.

Requirements this is built to meet (from docs/evaluation.md):

**Fast.** It runs at every checkpoint. An eval you stop running because it is
slow leaves you flying blind.

**Deterministic.** Fixed seeds, fixed prompts, fixed order, sequential coverage.
Eval variance masquerading as model variance is a miserable bug to chase.

**Versioned.** ``EVAL_SUITE_VERSION`` is stamped into every result. Changing
what the suite measures invalidates comparison with earlier runs, and without
the stamp that invalidation is silent.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import torch

from gitai.data import BatchSampler
from gitai.interpret.surprisal import surprisal_bits

from .metrics import EVAL_SUITE_VERSION, EvalResult, accuracy_at_k, perplexity
from .probes.base import Probe

__all__ = ["evaluate", "frequent_tokens", "generate_samples"]

DEFAULT_PROMPTS = ("First Citizen:\n", "KING RICHARD III:\n", "To be, or not to be")


@torch.no_grad()
def evaluate(
    model,
    sampler: BatchSampler,
    batch_size: int = 16,
    max_batches: int | None = None,
    probes: Sequence[Probe] = (),
    tokenizer=None,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    sample_tokens: int = 160,
    seed: int = 0,
) -> EvalResult:
    """Score a model on one split.

    Evaluation is **sequential**, not randomly sampled: random windows would
    score some tokens twice and skip others, adding variance to a number whose
    entire purpose is comparison across runs.
    """
    started = time.perf_counter()
    was_training = model.training
    model.eval()

    total_bits = 0.0
    total_tokens = 0
    correct_1 = 0.0
    correct_5 = 0.0
    batches = 0

    try:
        for i, (x, y) in enumerate(sampler.sequential(batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            logits, _ = model(inputs, targets)

            bits = surprisal_bits(logits, targets)
            total_bits += float(bits.sum())
            total_tokens += bits.numel()
            correct_1 += accuracy_at_k(logits, targets, 1)
            correct_5 += accuracy_at_k(logits, targets, 5)
            batches += 1

        if total_tokens == 0:
            raise ValueError(f"split {sampler.split!r} produced no evaluable tokens")

        bits_per_token = total_bits / total_tokens
        # Scale the split's total byte count by the fraction of it actually
        # evaluated, so a truncated eval still reports a correct bits-per-byte.
        bytes_per_token = sampler.utf8_bytes / sampler.total_tokens
        evaluated_bytes = total_tokens * bytes_per_token

        result = EvalResult(
            suite_version=EVAL_SUITE_VERSION,
            split=sampler.split,
            bits_per_byte=total_bits / evaluated_bytes,
            loss=bits_per_token * math.log(2),
            perplexity=perplexity(bits_per_token * math.log(2)),
            bits_per_token=bits_per_token,
            accuracy_top1=correct_1 / batches,
            accuracy_top5=correct_5 / batches,
            tokens=total_tokens,
            utf8_bytes=int(evaluated_bytes),
            seconds=time.perf_counter() - started,
        )

        for probe in probes:
            result.probes.update(probe.run(model))

        if tokenizer is not None and prompts:
            result.samples = generate_samples(model, tokenizer, prompts, sample_tokens, seed)

        result.seconds = time.perf_counter() - started
        return result
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_samples(
    model, tokenizer, prompts: Sequence[str], max_new_tokens: int = 160, seed: int = 0
) -> list[str]:
    """Fixed prompts, fixed seed, every time.

    Numbers hide catastrophic failure modes; a page of generated text does not.
    Fixing the seed and the prompts is what makes two checkpoints' samples
    comparable — sampling differences would otherwise swamp model differences.
    """
    out: list[str] = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(
            ids, max_new_tokens, temperature=0.8, generator=torch.Generator().manual_seed(seed)
        )
        out.append(tokenizer.decode(generated[0].tolist()))
    return out


def frequent_tokens(sampler: BatchSampler, top: int = 512, batches: int = 40) -> object:
    """The most common token ids in a split.

    Used as the sampling pool for the induction probe. Drawing uniformly from
    the whole vocabulary instead would mostly draw rare tokens whose embeddings
    barely moved in training — that measures embedding coverage, not induction.
    """
    import numpy as np

    counts = np.zeros(sampler.vocab_size, dtype=np.int64)
    for i, (x, _) in enumerate(sampler.sequential(32)):
        if i >= batches:
            break
        counts += np.bincount(x.reshape(-1), minlength=sampler.vocab_size)
    return np.argsort(counts)[::-1][:top].copy()
