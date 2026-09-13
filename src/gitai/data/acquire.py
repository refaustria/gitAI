"""Corpus acquisition with verified provenance.

Every download is recorded in ``MANIFEST.json`` with its SHA-256, size, licence
and retrieval date. That manifest is committed; the corpora themselves never
are. It is what lets anyone verify they have byte-identical inputs to yours —
and reproducibility that depends on a URL still serving the same bytes a year
later is not reproducibility.

Licences are recorded before the bytes land. Some corpora forbid
redistribution, which is a second reason corpus content stays out of git
regardless of size.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["REGISTRY", "Manifest", "Source", "fetch", "sha256_file", "verify"]

CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    licence: str
    description: str
    # None means "trust on first use": the hash is computed and recorded on the
    # first download. Stated explicitly rather than invented, because a wrong
    # checksum in a registry is worse than an absent one.
    sha256: str | None = None
    expected_bytes: int | None = None
    stage: int = 0


REGISTRY: dict[str, Source] = {
    "tinyshakespeare": Source(
        name="tinyshakespeare",
        url=(
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        ),
        sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        expected_bytes=1_115_394,
        licence="Public domain (Shakespeare); compilation MIT (karpathy/char-rnn)",
        description=(
            "~1MB of Shakespeare. The smoke test: trains in minutes, proves the loop works."
        ),
        stage=0,
    ),
    "tinystories": Source(
        name="tinystories",
        url=(
            "https://huggingface.co/datasets/roneneldan/TinyStories/"
            "resolve/main/TinyStoriesV2-GPT4-train.txt"
        ),
        sha256=None,  # recorded on first download
        licence="CDLA-Sharing-1.0 — check current terms at the dataset page before redistributing",
        description=(
            "Synthetic short stories in a 3-4 year old's vocabulary. The primary corpus: "
            "models of 1-35M parameters produce coherent English on it, which is what makes "
            "a no-GPU from-scratch LLM a real project rather than a toy."
        ),
        stage=1,
    ),
    "tinystories-valid": Source(
        name="tinystories-valid",
        url=(
            "https://huggingface.co/datasets/roneneldan/TinyStories/"
            "resolve/main/TinyStoriesV2-GPT4-valid.txt"
        ),
        sha256=None,
        licence=(
            "CDLA-Sharing-1.0 — check current terms at the dataset page before redistributing"
        ),
        description=(
            "TinyStories validation split (~20MB). Useful for pipeline testing at real scale."
        ),
        stage=1,
    ),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class Manifest:
    """``data/raw/MANIFEST.json`` — committed; the corpora it describes are not."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8")).get("sources", {})

    def record(self, source: Source, local_path: Path, digest: str, size: int) -> None:
        self.entries[source.name] = {
            **asdict(source),
            "sha256": digest,
            "actual_bytes": size,
            "local_path": local_path.name,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": 1, "sources": self.entries}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def digest_for(self, name: str) -> str | None:
        entry = self.entries.get(name)
        return entry.get("sha256") if entry else None


def verify(path: Path | str, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256


def fetch(
    name: str,
    dest_dir: Path | str = "data/raw",
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download a registered corpus, verify it, and record it in the manifest.

    Re-running is cheap: an existing file whose checksum matches is not
    re-downloaded. A file whose checksum does *not* match is an error, never a
    silent overwrite — the upstream data changed, and every result computed from
    the old bytes needs re-examining before you proceed.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown source {name!r}; registered: {sorted(REGISTRY)}")
    source = REGISTRY[name]

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{name}.txt"
    manifest = Manifest(dest_dir / "MANIFEST.json")

    expected = source.sha256 or manifest.digest_for(name)

    if target.exists() and not force:
        digest = sha256_file(target)
        if expected is None:
            manifest.record(source, target, digest, target.stat().st_size)
            return target
        if digest == expected:
            # Record even on a cache hit: the manifest is the reproducibility
            # record, and it must exist whether or not this run did the download.
            manifest.record(source, target, digest, target.stat().st_size)
            return target
        raise RuntimeError(
            f"{target} does not match the recorded checksum.\n"
            f"  expected {expected}\n  found    {digest}\n"
            "The upstream data changed, or the file is corrupt. Any result computed "
            "from the previous bytes needs re-examining. Delete the file and re-fetch "
            "deliberately if this is expected."
        )

    if progress:
        print(f"fetching {source.name} from {source.url}")
    tmp = target.with_suffix(".partial")
    with urllib.request.urlopen(source.url) as response, tmp.open("wb") as out:
        downloaded = 0
        while chunk := response.read(CHUNK):
            out.write(chunk)
            downloaded += len(chunk)
            if progress and downloaded % (32 * CHUNK) == 0:
                print(f"  {downloaded / 1024**2:,.0f} MiB", end="\r")

    digest = sha256_file(tmp)
    size = tmp.stat().st_size

    if expected and digest != expected:
        tmp.unlink()
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {expected}, got {digest}. "
            "Refusing to use data that is not what the registry describes."
        )
    if source.expected_bytes and size != source.expected_bytes:
        tmp.unlink()
        raise RuntimeError(
            f"size mismatch for {name}: expected {source.expected_bytes}, got {size}"
        )

    tmp.replace(target)
    manifest.record(source, target, digest, size)
    if progress:
        trust = " (trust-on-first-use)" if source.sha256 is None else ""
        print(f"  {size:,} bytes, sha256 {digest[:16]}...{trust}")
    return target
