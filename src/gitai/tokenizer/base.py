"""The tokenizer interface, and the one property every tokenizer must have."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["Tokenizer"]


class Tokenizer(ABC):
    """Text <-> integer ids.

    The contract every implementation here must satisfy is exact round-trip:
    ``decode(encode(s)) == s`` for *any* string, including emoji, CJK, and
    malformed-looking input. A tokenizer that silently mangles rare characters
    corrupts the training data in a way that never surfaces as an error — the
    model simply never learns those characters and nobody finds out why.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def save(self, path: Path | str) -> None: ...

    def compression_ratio(self, text: str) -> float:
        """UTF-8 bytes per token. Higher means a sequence covers more text.

        This is the number that decides whether a bigger vocabulary is worth its
        embedding parameters (Decision 4). A vocab that doubles the parameter
        count for a 3% compression gain is a bad trade at this scale.
        """
        tokens = len(self.encode(text))
        return len(text.encode("utf-8")) / tokens if tokens else 0.0

    def fingerprint(self) -> str:
        """Stable hash of the tokenizer's content.

        Goes into every shard's ``meta.json`` and every run manifest. Retokenising
        a corpus with a different tokenizer and comparing the losses is a silent,
        entirely invisible mistake; this is what makes it detectable.
        """
        return hashlib.sha256(self._fingerprint_material()).hexdigest()[:16]

    @abstractmethod
    def _fingerprint_material(self) -> bytes: ...
