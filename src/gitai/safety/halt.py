"""The stop button, and the budget that makes stopping the default.

Two mechanisms, because "unattended" is the whole premise:

``HaltSwitch``  an out-of-band signal the loop watches for and cannot revoke.
``Budget``      a finite allowance of iterations, wall-clock and disk. Running
                out is normal termination, not an error.

The budget matters more than the switch. A stop button only helps someone who
is watching; a budget bounds a loop nobody is watching. The default posture is
therefore **stops unless renewed**, never **runs unless stopped**.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Budget", "HaltRequested", "HaltSwitch"]


class HaltRequested(BaseException):
    """Raised when the loop must stop.

    Inherits :class:`BaseException`, not :class:`Exception`, and that is the
    entire point. A long-running loop is full of ``except Exception`` handlers
    that retry and continue — exactly the code that would otherwise swallow a
    stop request and keep going. ``KeyboardInterrupt`` and ``SystemExit`` are
    built this way for the same reason.

    Never catch this to continue. Catch it to clean up, then re-raise.
    """


@dataclass
class HaltSwitch:
    """Halts when ``signal_path`` appears.

    The signal lives *outside* every root the loop is allowed to write to (see
    :class:`~gitai.safety.guard.PathGuard`), so the loop can observe it but has
    no sanctioned way to delete it. That asymmetry — readable by the loop,
    writable only by the operator — is what makes it a switch rather than a
    suggestion.

    This is an honest engineering boundary, not a security boundary: code that
    genuinely wanted to defeat it could simply not call ``check()``. Against the
    actual threat model here — a loop that optimises its way into somewhere
    unintended, or an operator who needs it to stop at 3am — it is sufficient.
    A model at this scale cannot author the code that would evade it.
    """

    signal_path: Path
    checks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.signal_path = Path(self.signal_path)

    def requested(self) -> bool:
        self.checks += 1
        return self.signal_path.exists()

    def reason(self) -> str:
        try:
            text = self.signal_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "halt requested (signal file unreadable)"
        return text or "halt requested (no reason given)"

    def check(self) -> None:
        """Raise if a halt has been requested. Call at every loop boundary."""
        if self.requested():
            raise HaltRequested(self.reason())

    def request(self, reason: str = "operator requested halt") -> None:
        """Operator-side helper. The loop must never call this on itself."""
        self.signal_path.parent.mkdir(parents=True, exist_ok=True)
        self.signal_path.write_text(reason, encoding="utf-8")

    def clear(self) -> None:
        """Operator-side helper. Deliberately not called anywhere in the loop."""
        self.signal_path.unlink(missing_ok=True)


@dataclass
class Budget:
    """A finite allowance. Exhaustion is a normal, expected stop."""

    max_iterations: int = 100
    max_wall_seconds: float = 3600.0
    max_disk_bytes: int = 20 * 1024**3
    started_at: float = field(default_factory=time.monotonic)
    iterations: int = field(default=0, init=False)

    def tick(self) -> None:
        self.iterations += 1

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> dict[str, float]:
        return {
            "iterations": self.max_iterations - self.iterations,
            "wall_seconds": max(0.0, self.max_wall_seconds - self.elapsed),
        }

    def check(self, disk_bytes_used: int = 0) -> None:
        if self.iterations >= self.max_iterations:
            raise HaltRequested(f"iteration budget exhausted ({self.max_iterations})")
        if self.elapsed >= self.max_wall_seconds:
            raise HaltRequested(f"wall-clock budget exhausted ({self.max_wall_seconds:.0f}s)")
        if disk_bytes_used >= self.max_disk_bytes:
            raise HaltRequested(
                f"disk budget exhausted ({disk_bytes_used / 1024**3:.1f} GiB "
                f"of {self.max_disk_bytes / 1024**3:.1f} GiB)"
            )
