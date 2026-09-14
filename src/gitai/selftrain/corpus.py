"""Generating a synthetic corpus from a trained model, and measuring its diversity.

Two decisions here shape the whole experiment.

**Generation is unconditional.** Sequences start from the document separator,
never from a real prompt. Prompting with real text would leak the real corpus
into the synthetic one, and the entire question is what happens when a model
trains on *its own* output — a leak would quietly answer it in the wrong
direction.

**Diversity is measured at generation time.** Model collapse shows up as a
narrowing distribution long before it shows up in held-out loss, so the
distinct-n-gram ratios recorded here are an early indicator. They are also the
mechanism: collapse is the tails of the distribution disappearing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

__all__ = ["CorpusStats", "diversity", "generate_corpus"]


@dataclass
class CorpusStats:
    documents: int = 0
    tokens: int = 0
    utf8_bytes: int = 0
    seconds: float = 0.0
    tokens_per_sec: float = 0.0
    mean_document_tokens: float = 0.0
    vocabulary_used: int = 0
    vocabulary_fraction: float = 0.0
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    distinct_3: float = 0.0
    temperature: float = 1.0
    top_k: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.documents:,} docs / {self.tokens:,} tokens in {self.seconds:.0f}s "
            f"({self.tokens_per_sec:,.0f} tok/s)\n"
            f"  vocabulary {self.vocabulary_used:,} types "
            f"({self.vocabulary_fraction:.1%} of vocab)\n"
            f"  distinct  1-gram {self.distinct_1:.4f}  2-gram {self.distinct_2:.4f}  "
            f"3-gram {self.distinct_3:.4f}"
        )


def diversity(token_ids: list[int], vocab_size: int) -> dict[str, float]:
    """Distinct-n-gram ratios: unique n-grams divided by total n-grams.

    The standard cheap diversity measure. A collapsing model repeats itself, so
    these fall — and they fall *before* held-out loss rises, which is what makes
    them worth recording separately.
    """
    out: dict[str, float] = {}
    for n in (1, 2, 3):
        if len(token_ids) < n:
            out[f"distinct_{n}"] = 0.0
            continue
        grams = {tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)}
        out[f"distinct_{n}"] = len(grams) / (len(token_ids) - n + 1)
    out["vocabulary_used"] = len(set(token_ids))
    out["vocabulary_fraction"] = out["vocabulary_used"] / vocab_size
    return out


@torch.no_grad()
def generate_corpus(
    model,
    tokenizer,
    target_tokens: int,
    *,
    separator_id: int | None = None,
    batch_size: int = 256,
    length: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    progress: bool = False,
) -> tuple[list[str], CorpusStats]:
    """Sample ``target_tokens`` tokens of text from ``model``.

    Returns documents (split on the separator) and diversity statistics.

    Sampling temperature is a real experimental variable, not a detail: low
    temperature produces cleaner text that collapses *faster*, because the tails
    of the distribution are cut off at generation time rather than surviving into
    the next generation's training data.
    """
    config = model.config
    length = length or config.seq_len
    if separator_id is None:
        separator_id = getattr(tokenizer, "special_tokens", {}).get("<|endoftext|>")
    if separator_id is None:
        raise ValueError(
            "no document separator: pass separator_id, or use a tokenizer trained "
            "with <|endoftext|>. Without one the generated documents cannot be split."
        )

    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    collected: list[int] = []
    batches = 0

    while len(collected) < target_tokens:
        prompt = torch.full((batch_size, 1), separator_id, dtype=torch.long)
        produced = model.generate(
            prompt,
            length,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
            use_cache=True,
        )
        # Drop the seed separator from each row; keep the generated remainder.
        for row in produced[:, 1:].tolist():
            collected.extend(row)
        batches += 1
        if progress:
            print(f"    {len(collected):,}/{target_tokens:,} tokens", end="\r")

    collected = collected[:target_tokens]

    # Split into documents on the separator.
    documents: list[str] = []
    current: list[int] = []
    for token in collected:
        if token == separator_id:
            if current:
                documents.append(tokenizer.decode(current))
            current = []
        else:
            current.append(token)
    if current:
        documents.append(tokenizer.decode(current))

    elapsed = time.perf_counter() - started
    measures = diversity(collected, config.vocab_size)
    stats = CorpusStats(
        documents=len(documents),
        tokens=len(collected),
        utf8_bytes=sum(len(d.encode("utf-8")) for d in documents),
        seconds=elapsed,
        tokens_per_sec=len(collected) / elapsed if elapsed else 0.0,
        mean_document_tokens=len(collected) / len(documents) if documents else 0.0,
        vocabulary_used=int(measures["vocabulary_used"]),
        vocabulary_fraction=measures["vocabulary_fraction"],
        distinct_1=measures["distinct_1"],
        distinct_2=measures["distinct_2"],
        distinct_3=measures["distinct_3"],
        temperature=temperature,
        top_k=top_k,
    )
    return documents, stats
