"""Batch sampling from memory-mapped token shards.

Profile this before optimising anything in the model. On a CPU-only machine the
data loader is very often the bottleneck rather than the matmuls — the opposite
of the GPU intuition nearly everyone arrives with. ``BatchSampler.throughput``
exists to settle that question with a measurement instead of an argument.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .shards import ShardIndex

__all__ = ["BatchSampler"]


class BatchSampler:
    """Samples ``(inputs, targets)`` from random offsets across a split's shards.

    Targets are inputs shifted by one position: predicting token *t+1* from
    tokens *0..t* is the entire training objective of a causal language model.

    Shards are opened with ``np.memmap``, so nothing is read until it is
    touched and the OS page cache handles residency. A 2GB corpus works
    identically on a laptop with 8GB of RAM and one with 64GB — only the hit
    rate changes.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        seq_len: int,
        index: ShardIndex | None = None,
    ) -> None:
        self.index = index or ShardIndex.load(root)
        self.split = split
        self.seq_len = seq_len

        info = self.index.split(split)
        if not info.shards:
            raise ValueError(f"split {split!r} has no shards")

        self._maps: list[np.memmap] = []
        self._lengths: list[int] = []
        for path in self.index.shard_paths(split):
            mapped = np.memmap(path, dtype=self.index.np_dtype, mode="r")
            # +1 because the target sequence extends one token past the input.
            if len(mapped) < seq_len + 1:
                continue
            self._maps.append(mapped)
            self._lengths.append(len(mapped))

        if not self._maps:
            raise ValueError(
                f"no shard in split {split!r} holds {seq_len + 1} tokens; "
                f"largest is {max((s.tokens for s in info.shards), default=0)}. "
                "Reduce seq_len or add data."
            )

        # Sample shards in proportion to length, so every token is equally
        # likely regardless of how the corpus happened to be chunked.
        weights = np.asarray(self._lengths, dtype=np.float64)
        self._weights = weights / weights.sum()

    # ------------------------------------------------------------ properties

    @property
    def total_tokens(self) -> int:
        return int(sum(self._lengths))

    @property
    def vocab_size(self) -> int:
        return self.index.vocab_size

    @property
    def utf8_bytes(self) -> int:
        """Raw bytes of the underlying text — the denominator for bits-per-byte."""
        return self.index.split(self.split).utf8_bytes

    # -------------------------------------------------------------- sampling

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One batch. ``int64`` because that is what embedding lookups want."""
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)

        shard_ids = rng.choice(len(self._maps), size=batch_size, p=self._weights)
        for row, shard_id in enumerate(shard_ids):
            mapped = self._maps[shard_id]
            start = int(rng.integers(0, self._lengths[shard_id] - self.seq_len))
            window = mapped[start : start + self.seq_len + 1]
            x[row] = window[:-1]
            y[row] = window[1:]
        return x, y

    def batches(
        self, batch_size: int, steps: int, seed: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """A reproducible stream. Same seed, same batches, in the same order."""
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            yield self.batch(batch_size, rng)

    def sequential(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Every token exactly once, in order, no overlap.

        Use this for evaluation. Random sampling would score some tokens twice
        and others not at all, which adds variance to a number you are trying to
        compare across runs.
        """
        for mapped, length in zip(self._maps, self._lengths, strict=True):
            windows = (length - 1) // self.seq_len
            for start in range(0, windows * self.seq_len, batch_size * self.seq_len):
                rows = min(batch_size, (windows * self.seq_len - start) // self.seq_len)
                if rows <= 0:
                    break
                x = np.empty((rows, self.seq_len), dtype=np.int64)
                y = np.empty((rows, self.seq_len), dtype=np.int64)
                for row in range(rows):
                    offset = start + row * self.seq_len
                    window = mapped[offset : offset + self.seq_len + 1]
                    x[row] = window[:-1]
                    y[row] = window[1:]
                yield x, y

    # ------------------------------------------------------------ profiling

    def throughput(self, batch_size: int, steps: int = 50, seed: int = 0) -> dict[str, float]:
        """Measure loader-only tokens/sec. Compare against the model's step time:
        if this is the smaller number, the loader is your bottleneck."""
        import time

        rng = np.random.default_rng(seed)
        self.batch(batch_size, rng)  # warm the page cache

        start = time.perf_counter()
        for _ in range(steps):
            self.batch(batch_size, rng)
        elapsed = time.perf_counter() - start

        tokens = steps * batch_size * self.seq_len
        return {
            "tokens_per_sec": tokens / elapsed,
            "batches_per_sec": steps / elapsed,
            "ms_per_batch": 1000.0 * elapsed / steps,
        }
