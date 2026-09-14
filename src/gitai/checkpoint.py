"""Saving and restoring a training run, without pickle.

A run that cannot resume is a run you lose to a closed laptop lid, a full disk,
or an OOM at hour six. For the Phase 8 loop it is worse than an inconvenience:
an unattended process that cannot survive an interruption is fragile by
construction.

**Why this is more work than ``torch.save``.** Decision 6 says checkpoints are
safetensors, never pickle — pickle is brittle across versions and unsafe to load
from anywhere you do not control. But ``optimiser.state_dict()`` is nested
Python objects, which safetensors cannot hold. So optimiser state is flattened
into tensors with structured keys and the non-tensor remainder goes to JSON. The
alternative was to quietly exempt optimiser state from a rule the project had
already committed to.

**What a correct resume needs.** All five, or the resumed run diverges:

1. model weights
2. optimiser state — Adam's moment estimates, which take hundreds of steps to
   rebuild if lost
3. the step number — the LR schedule is a function of it
4. RNG state, both torch and numpy
5. the data sampler's position, which is the numpy generator's state

Dropping any one produces a run that *looks* fine and silently differs from an
uninterrupted one. ``test_resume_reproduces_an_uninterrupted_run`` is what
catches that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["ResumeState", "load_checkpoint", "rotate_checkpoints", "save_checkpoint"]

MODEL_FILE = "model.safetensors"
OPTIMISER_FILE = "optimiser.safetensors"
STATE_FILE = "state.json"


@dataclass
class ResumeState:
    """Everything needed to continue a run exactly where it stopped."""

    step: int
    best_bpb: float
    best_step: int
    rng_state: dict[str, Any]
    elapsed: float = 0.0
    metadata: dict[str, Any] | None = None


def _split_optimiser_state(optimiser) -> tuple[dict[str, torch.Tensor], dict]:
    """Separate optimiser state into tensors (safetensors) and the rest (JSON)."""
    state = optimiser.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, Any] = {}

    for param_id, entry in state["state"].items():
        for field, value in entry.items():
            key = f"{param_id}.{field}"
            if isinstance(value, torch.Tensor):
                # Adam's `step` is a 0-dim tensor in recent PyTorch and a plain
                # int in older versions; both round-trip through here.
                tensors[key] = value.detach().contiguous()
            else:
                scalars[key] = value

    return tensors, {"scalars": scalars, "param_groups": state["param_groups"]}


def _rebuild_optimiser_state(optimiser, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    state: dict[int, dict[str, Any]] = {}
    for key, tensor in tensors.items():
        param_id, field = key.split(".", 1)
        state.setdefault(int(param_id), {})[field] = tensor
    for key, value in meta["scalars"].items():
        param_id, field = key.split(".", 1)
        state.setdefault(int(param_id), {})[field] = value
    optimiser.load_state_dict({"state": state, "param_groups": meta["param_groups"]})


def save_checkpoint(
    directory: Path | str,
    model,
    optimiser,
    step: int,
    rng: np.random.Generator,
    best_bpb: float = float("inf"),
    best_step: int = 0,
    elapsed: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a complete, resumable checkpoint.

    Written to a temporary directory and moved into place, so an interruption
    mid-write cannot leave a half-written checkpoint that loads without error and
    resumes into nonsense.
    """
    from safetensors.torch import save_file, save_model

    directory = Path(directory)
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # save_model, not save_file: tied embeddings make the head and the token
    # embedding one aliased tensor, which save_file refuses to write.
    save_model(model, str(staging / MODEL_FILE))

    tensors, optimiser_meta = _split_optimiser_state(optimiser)
    # torch's RNG state is a uint8 tensor; it rides along with the optimiser
    # tensors rather than needing a file of its own.
    tensors["__torch_rng_state__"] = torch.get_rng_state().clone()
    save_file(tensors, str(staging / OPTIMISER_FILE))

    (staging / STATE_FILE).write_text(
        json.dumps(
            {
                "version": 1,
                "step": step,
                "best_bpb": best_bpb if best_bpb != float("inf") else None,
                "best_step": best_step,
                "elapsed": elapsed,
                "numpy_rng_state": rng.bit_generator.state,
                "optimiser": optimiser_meta,
                "metadata": metadata or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if directory.exists():
        import shutil

        shutil.rmtree(directory)
    staging.rename(directory)
    return directory


def load_checkpoint(
    directory: Path | str, model, optimiser, rng: np.random.Generator
) -> ResumeState:
    """Restore a run in place. ``model``, ``optimiser`` and ``rng`` are mutated."""
    from safetensors.torch import load_file, load_model

    directory = Path(directory)
    if not (directory / STATE_FILE).exists():
        raise FileNotFoundError(f"no checkpoint at {directory} (missing {STATE_FILE})")

    payload = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    load_model(model, str(directory / MODEL_FILE))

    tensors = load_file(str(directory / OPTIMISER_FILE))
    torch_rng = tensors.pop("__torch_rng_state__", None)
    if torch_rng is not None:
        torch.set_rng_state(torch_rng.to(torch.uint8))
    _rebuild_optimiser_state(optimiser, tensors, payload["optimiser"])

    # Restoring the generator's state is what puts the data loader back on the
    # same batch sequence. Without it the resumed run sees different data and
    # diverges while looking perfectly healthy.
    rng.bit_generator.state = payload["numpy_rng_state"]

    best = payload.get("best_bpb")
    return ResumeState(
        step=payload["step"],
        best_bpb=float("inf") if best is None else float(best),
        best_step=payload.get("best_step", 0),
        rng_state=payload["numpy_rng_state"],
        elapsed=payload.get("elapsed", 0.0),
        metadata=payload.get("metadata") or {},
    )


def rotate_checkpoints(directory: Path | str, keep: int = 2, pattern: str = "step-*") -> list[Path]:
    """Delete all but the ``keep`` most recent checkpoints.

    A long run writing a checkpoint every few hundred steps fills a laptop disk
    and dies at hour nine — which is both avoidable and infuriating. Returns the
    directories removed.
    """
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")

    directory = Path(directory)
    if not directory.exists():
        return []

    def step_of(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    checkpoints = sorted(
        (p for p in directory.glob(pattern) if p.is_dir() and step_of(p) >= 0),
        key=step_of,
    )
    removed: list[Path] = []
    for stale in checkpoints[:-keep]:
        import shutil

        shutil.rmtree(stale)
        removed.append(stale)
    return removed
