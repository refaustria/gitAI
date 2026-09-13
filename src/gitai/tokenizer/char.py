"""Character-level tokenizer. The smoke test, not the destination.

Exists so that Phases 2 and 3 can be debugged before the BPE tokenizer is
finished, and so that every later comparison has a trivially-correct baseline.
On TinyShakespeare this gives a ~65-token vocabulary, which is small enough that
the embedding table is free and any loss curve is directly interpretable:
uniform prediction is exactly ln(65).
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Tokenizer

__all__ = ["CharTokenizer"]

UNKNOWN = "�"  # replacement character, also used as the OOV token


class CharTokenizer(Tokenizer):
    def __init__(self, characters: list[str]) -> None:
        chars = sorted(set(characters) | {UNKNOWN})
        self.itos: dict[int, str] = dict(enumerate(chars))
        self.stoi: dict[str, int] = {c: i for i, c in self.itos.items()}
        self._unk = self.stoi[UNKNOWN]

    @classmethod
    def train(cls, texts) -> CharTokenizer:
        seen: set[str] = set()
        for text in texts:
            seen.update(text)
        return cls(sorted(seen))

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self._unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, UNKNOWN) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        characters = [self.itos[i] for i in sorted(self.itos)]
        payload = {"kind": "char", "version": 1, "characters": characters}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> CharTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "char":
            raise ValueError(f"not a char tokenizer file: {path}")
        return cls(payload["characters"])

    def _fingerprint_material(self) -> bytes:
        return json.dumps([self.itos[i] for i in sorted(self.itos)], ensure_ascii=False).encode()
