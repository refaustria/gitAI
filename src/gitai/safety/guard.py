"""Containment for the loop's side effects: where it may write, what it may reach.

Scope, stated plainly so nobody over-trusts this module: these are guards
against *accident and drift*, not a sandbox against adversarial code. Anything
running in this interpreter can bypass them by not calling them. Real isolation
is an OS-level concern — containers, seccomp, an unprivileged user, no network
namespace — and belongs in the runbook, not in a Python import.

That said, the threat here is not an adversary. It is an unattended loop that
generates paths and configs and gets one wrong at 3am. Against that, a guard
that fails loudly at the boundary is exactly the right tool.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NetworkViolation", "PathGuard", "PathViolation", "deny_network"]


class PathViolation(Exception):
    """A write was attempted outside every allowed root."""


class NetworkViolation(Exception):
    """A socket was opened while network access was denied."""


@dataclass
class PathGuard:
    """Confines writes to an explicit allowlist of roots.

    Paths are fully resolved before the containment check, so ``..`` traversal
    and symlinks that point outside a root are both caught — a symlink escape
    resolves to its real target, which then fails containment.
    """

    allowed_roots: tuple[Path, ...]

    def __init__(self, allowed_roots: Iterable[Path | str]) -> None:
        roots = tuple(Path(r).resolve() for r in allowed_roots)
        if not roots:
            raise ValueError("PathGuard requires at least one allowed root")
        self.allowed_roots = roots

    def is_allowed(self, path: Path | str) -> bool:
        resolved = Path(path).resolve()
        return any(resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots)

    def check(self, path: Path | str) -> Path:
        """Return the resolved path, or raise :class:`PathViolation`."""
        resolved = Path(path).resolve()
        if not self.is_allowed(resolved):
            roots = ", ".join(str(r) for r in self.allowed_roots)
            raise PathViolation(f"write to {resolved} is outside the allowed roots: {roots}")
        return resolved

    def open_write(self, path: Path | str, mode: str = "w", **kwargs):
        if "r" in mode and "+" not in mode:
            raise ValueError("open_write is for writing; read directly instead")
        resolved = self.check(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved.open(mode, **kwargs)


@contextmanager
def deny_network() -> Iterator[None]:
    """Make socket creation raise for the duration of the block.

    Used to wrap the improvement loop. A training loop has no business making
    network calls; if one appears, that is either a dependency phoning home or a
    bug, and both should surface immediately rather than at the next invoice.

    Advisory only — see the module docstring.
    """
    original = socket.socket

    class _Denied(socket.socket):
        def __init__(self, *args, **kwargs):
            raise NetworkViolation(
                "network access is denied inside the improvement loop; "
                "fetch data in a separate, supervised step"
            )

    socket.socket = _Denied  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[misc]
