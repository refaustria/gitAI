"""Byte-level Byte-Pair Encoding, written from scratch.

BPE in one paragraph: start with the 256 possible bytes as your vocabulary,
then repeatedly find the most frequent adjacent pair of tokens in the corpus and
merge it into a single new token. Do that N times and you have a vocabulary of
256 + N entries in which common sequences ("the", " said") are single tokens and
rare ones decompose into pieces. Starting from *bytes* rather than characters is
what makes it total: every possible input is representable, so there is no
out-of-vocabulary case and no unknown token, ever.

Three implementation decisions in here are worth more than the algorithm:

**Pre-tokenization.** The corpus is first split by a regex into word-ish chunks,
and merges are never allowed to cross a chunk boundary. Without this, BPE
happily learns tokens that span a word and its neighbour's punctuation, which
wastes vocabulary on artefacts of whitespace.

**Incremental pair counts.** The naive trainer recounts every pair in the corpus
after every merge, which is O(merges x corpus) and takes hours for an 8k vocab.
This keeps a running count plus an index from pair to the words containing it,
and touches only the affected words. Same result, minutes instead of hours.

**Deterministic tie-breaking.** When two pairs have equal frequency the winner is
chosen by a fixed rule, not by dict ordering. Training the same corpus twice must
produce byte-identical merges or nothing downstream is reproducible.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import regex

from .base import Tokenizer

__all__ = ["GPT2_SPLIT_PATTERN", "ByteBPETokenizer"]

# The GPT-2 pre-tokenization pattern. Keeps contractions together, attaches a
# leading space to a word (" the" is one token, distinct from "the"), and never
# lets letters, digits and punctuation merge into one chunk. Requires the
# `regex` module: `re` has no \p{L} Unicode property support.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_BYTE_VOCAB_SIZE = 256


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping left-to-right occurrence of ``pair``."""
    out: list[int] = []
    i = 0
    last = len(word) - 1
    while i < len(word):
        if i < last and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _pairs(word: list[int]):
    """Adjacent pairs: [a,b,c] -> (a,b), (b,c)."""
    return pairwise(word)


class ByteBPETokenizer(Tokenizer):
    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: dict[str, int] | None = None,
        pattern: str = GPT2_SPLIT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._compiled = regex.compile(pattern)
        self.merges: dict[tuple[int, int], int] = dict(merges or {})
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self._special_by_id = {i: s for s, i in self.special_tokens.items()}
        self.vocab = self._build_vocab()
        self._cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------ vocab

    def _build_vocab(self) -> dict[int, bytes]:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(_BYTE_VOCAB_SIZE)}
        # Merges are insertion-ordered and every merge's id exceeds both of its
        # parts, so a single forward pass resolves every token to bytes.
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        return vocab

    @property
    def vocab_size(self) -> int:
        return _BYTE_VOCAB_SIZE + len(self.merges) + len(self.special_tokens)

    # --------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (),
        pattern: str = GPT2_SPLIT_PATTERN,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> ByteBPETokenizer:
        specials = list(special_tokens)
        budget = vocab_size - _BYTE_VOCAB_SIZE - len(specials)
        if budget < 0:
            raise ValueError(
                f"vocab_size={vocab_size} leaves no room: 256 byte tokens plus "
                f"{len(specials)} special tokens already need "
                f"{_BYTE_VOCAB_SIZE + len(specials)}"
            )

        compiled = regex.compile(pattern)

        # 1. Pre-tokenize into chunks and count how often each distinct chunk
        #    occurs. Everything after this works on unique chunks weighted by
        #    frequency rather than on the raw corpus, which is the difference
        #    between seconds and hours.
        frequencies: Counter[bytes] = Counter()
        for text in texts:
            for chunk in compiled.findall(text):
                frequencies[chunk.encode("utf-8")] += 1

        words: list[list[int]] = [list(chunk) for chunk in frequencies]
        weights: list[int] = list(frequencies.values())

        # 2. Index every adjacent pair: how often it occurs, and which words
        #    contain it. The index is what makes updates incremental.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            for pair in _pairs(word):
                pair_counts[pair] += weights[i]
                pair_where[pair].add(i)

        merges: dict[tuple[int, int], int] = {}
        for step in range(budget):
            if not pair_counts:
                break
            # Deterministic tie-break: highest count, then lowest token ids.
            # Without the second key this depends on dict iteration order and
            # two runs on the same corpus can disagree.
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < min_frequency:
                break

            new_id = _BYTE_VOCAB_SIZE + step
            merges[best] = new_id

            for i in list(pair_where[best]):
                word, weight = words[i], weights[i]
                for pair in _pairs(word):
                    pair_counts[pair] -= weight
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_where[pair].discard(i)

                merged = _merge_word(word, best, new_id)
                words[i] = merged

                for pair in _pairs(merged):
                    pair_counts[pair] += weight
                    pair_where[pair].add(i)

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{budget}")

        tokenizer = cls(merges=merges, pattern=pattern)
        base = _BYTE_VOCAB_SIZE + len(merges)
        tokenizer.special_tokens = {name: base + k for k, name in enumerate(specials)}
        tokenizer._special_by_id = {i: s for s, i in tokenizer.special_tokens.items()}
        return tokenizer

    # --------------------------------------------------------------- encoding

    def _encode_chunk(self, data: bytes) -> list[int]:
        cached = self._cache.get(data)
        if cached is not None:
            return cached

        ids = list(data)
        while len(ids) >= 2:
            # Apply merges in the order they were *learned*, not greedily by
            # frequency: the lowest-rank applicable merge always goes first.
            # Encoding must replay training's decisions exactly, or the same
            # text tokenizes differently than it did during training.
            best = min(_pairs(ids), key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _merge_word(ids, best, self.merges[best])

        if len(data) <= 64:  # bound the cache: long chunks are rarely repeated
            self._cache[data] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode, treating special-token text as ordinary text."""
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode text, honouring special tokens unless told otherwise.

        ``allow_special=False`` matters for untrusted input: text containing the
        literal string ``<|endoftext|>`` should not be able to inject a document
        boundary into the training stream.
        """
        if not allow_special or not self.special_tokens:
            return self.encode_ordinary(text)

        splitter = "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
        out: list[int] = []
        for part in regex.split(splitter, text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self._special_by_id:
                parts.append(self._special_by_id[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"token id {i} is not in a vocabulary of {self.vocab_size}")
        # errors="replace" only matters for arbitrary id sequences (e.g. an
        # untrained model's samples), which can cut a multi-byte character in
        # half. Round-tripping real text is always exact.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- serialisation

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "byte_bpe",
            "version": 1,
            "pattern": self.pattern,
            # Lists, not a dict: JSON object keys cannot be tuples, and merge
            # ORDER is the thing that must survive the round trip.
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ByteBPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "byte_bpe":
            raise ValueError(f"not a byte-BPE tokenizer file: {path}")
        merges = {(a, b): idx for a, b, idx in payload["merges"]}
        return cls(
            merges=merges,
            special_tokens=payload.get("special_tokens", {}),
            pattern=payload.get("pattern", GPT2_SPLIT_PATTERN),
        )

    def _fingerprint_material(self) -> bytes:
        return json.dumps(
            {
                "pattern": self.pattern,
                "merges": [[a, b, i] for (a, b), i in self.merges.items()],
                "special_tokens": self.special_tokens,
            },
            sort_keys=False,
        ).encode("utf-8")
