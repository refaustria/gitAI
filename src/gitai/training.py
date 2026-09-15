"""The training loop, as a reusable function.

Extracted from ``scripts/train.py`` so the CLI and the Phase 5 collapse
experiment run the *same* loop. Two copies of a training loop drift, and when
they do, a difference between two experiments can no longer be attributed to
the thing being varied.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gitai.checkpoint import (
    latest_checkpoint as _latest_checkpoint,
)
from gitai.checkpoint import (
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from gitai.data import BatchSampler
from gitai.interpret.surprisal import surprisal_bits

__all__ = ["TrainConfig", "TrainResult", "lr_at", "train"]


def lr_at(step: int, total: int, peak: float, warmup: int, floor_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 16
    accumulation_steps: int = 1
    lr: float = 3e-3
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50
    seed: int = 0
    log_every: int = 50
    checkpoint_every: int = 0
    keep_last: int = 2

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.batch_size % self.accumulation_steps:
            raise ValueError(
                f"batch_size {self.batch_size} is not divisible by accumulation_steps "
                f"{self.accumulation_steps}; the effective batch would not be what you asked for"
            )

    @property
    def micro_batch(self) -> int:
        """Rows per forward pass.

        ``batch_size`` is the *effective* batch — the number of examples each
        optimiser step learns from. Accumulation splits it into micro-batches to
        fit in memory, and must not change what the step computes.
        """
        return self.batch_size // self.accumulation_steps


@dataclass
class TrainResult:
    best_bpb: float = float("inf")
    best_step: int = 0
    final_train_loss: float = 0.0
    tokens_per_sec: float = 0.0
    seconds: float = 0.0
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate_bpb(model, sampler: BatchSampler, batch_size: int, max_batches: int = 50) -> dict:
    """Held-out loss and bits-per-byte, evaluated sequentially.

    Sequential rather than randomly sampled: random windows would score some
    tokens twice and skip others, adding variance to a number whose whole
    purpose is comparison across runs.
    """
    was_training = model.training
    model.eval()
    total_bits = 0.0
    total_tokens = 0
    try:
        for i, (x, y) in enumerate(sampler.sequential(batch_size)):
            if i >= max_batches:
                break
            logits, _ = model(torch.from_numpy(x), torch.from_numpy(y))
            bits = surprisal_bits(logits, torch.from_numpy(y))
            total_bits += float(bits.sum())
            total_tokens += bits.numel()
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError(f"split {sampler.split!r} produced no evaluable tokens")

    bytes_per_token = sampler.utf8_bytes / sampler.total_tokens
    return {
        "val_loss": total_bits / total_tokens * math.log(2),
        "val_bits_per_token": total_bits / total_tokens,
        "val_bpb": total_bits / (total_tokens * bytes_per_token),
        "val_tokens": total_tokens,
    }


def train(
    model,
    train_sampler: BatchSampler,
    val_sampler: BatchSampler,
    config: TrainConfig,
    on_event: Callable[[dict], None] | None = None,
    keep_best: bool = True,
    checkpoint_dir: Path | str | None = None,
    resume: bool = False,
) -> TrainResult:
    """Train, tracking the best held-out checkpoint.

    ``keep_best`` restores the best-scoring weights into ``model`` at the end, so
    the caller receives the checkpoint the metrics describe rather than whatever
    the last step happened to produce.

    ``checkpoint_dir`` with ``config.checkpoint_every`` writes resumable
    checkpoints; ``resume=True`` continues from the most recent one. A resumed
    run reproduces an uninterrupted one exactly — see
    ``test_resume_reproduces_an_uninterrupted_run``.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Weight decay on matrices only. Decaying norms, biases and embeddings is a
    # small, silent, universally-copied mistake.
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(0.9, 0.95),
    )

    rng = np.random.default_rng(config.seed)
    result = TrainResult()
    best_state: dict | None = None
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    start_step = 0
    elapsed_offset = 0.0

    if resume and checkpoint_root is not None:
        latest = _latest_checkpoint(checkpoint_root)
        if latest is not None:
            restored = load_checkpoint(latest, model, optimiser, rng)
            start_step = restored.step + 1
            result.best_bpb = restored.best_bpb
            result.best_step = restored.best_step
            elapsed_offset = restored.elapsed
            # The best weights live on disk, not in this process's memory. Without
            # reloading them, finishing a resumed run that never beats the old
            # best would silently return the latest weights instead.
            best_state = _load_best_weights(checkpoint_root, model)
            if on_event:
                on_event({"event": "resumed", "step": restored.step, "from": str(latest)})

    model.train()
    started = time.perf_counter()

    for step in range(start_step, config.steps):
        lr = lr_at(step, config.steps, config.lr, config.warmup)
        for group in optimiser.param_groups:
            group["lr"] = lr

        # Accumulate over micro-batches, then take one step. Each micro-batch's
        # loss is divided by the accumulation count so the summed gradient equals
        # the gradient of the mean over the full effective batch — without that
        # division the effective learning rate silently scales with accumulation.
        optimiser.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.accumulation_steps):
            x, y = train_sampler.batch(config.micro_batch, rng)
            _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
            (loss / config.accumulation_steps).backward()
            total_loss += loss.item() / config.accumulation_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        result.final_train_loss = total_loss

        if step % config.log_every == 0 or step == config.steps - 1:
            elapsed = elapsed_offset + time.perf_counter() - started
            done = step - start_step + 1
            per_step = (time.perf_counter() - started) / done
            result.tokens_per_sec = config.batch_size * train_sampler.seq_len / per_step
            record = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": result.tokens_per_sec,
                "elapsed": elapsed,
                "eta_seconds": (config.steps - step - 1) * per_step,
            }
            result.history.append(record)
            if on_event:
                on_event(record)

        if (step + 1) % config.eval_every == 0 or step == config.steps - 1:
            stats = evaluate_bpb(model, val_sampler, config.batch_size, config.eval_batches)
            stats["step"] = step
            result.history.append(stats)
            if on_event:
                on_event(stats)
            if stats["val_bpb"] < result.best_bpb:
                result.best_bpb = stats["val_bpb"]
                result.best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if checkpoint_root is not None:
                    _save_best_weights(checkpoint_root, model)

        if (
            checkpoint_root is not None
            and config.checkpoint_every
            and (step + 1) % config.checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_root / f"step-{step}",
                model,
                optimiser,
                step=step,
                rng=rng,
                best_bpb=result.best_bpb,
                best_step=result.best_step,
                elapsed=elapsed_offset + time.perf_counter() - started,
            )
            rotate_checkpoints(checkpoint_root, keep=config.keep_last)

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)

    result.seconds = elapsed_offset + time.perf_counter() - started
    return result


def _save_best_weights(root: Path, model) -> None:
    from safetensors.torch import save_model

    root.mkdir(parents=True, exist_ok=True)
    save_model(model, str(root / "best.safetensors"))


def _load_best_weights(root: Path, model) -> dict | None:
    """Read the best weights back without disturbing the live model."""
    from safetensors.torch import load_file

    path = root / "best.safetensors"
    if not path.exists():
        return None
    stored = load_file(str(path))
    # save_model drops one side of a tied pair; fill it back from the model's own
    # state dict so load_state_dict gets a complete mapping.
    complete = {k: v.detach().clone() for k, v in model.state_dict().items()}
    complete.update(stored)
    return complete
