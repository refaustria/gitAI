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

import numpy as np
import torch

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
    lr: float = 3e-3
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50
    seed: int = 0
    log_every: int = 50


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
) -> TrainResult:
    """Train, tracking the best held-out checkpoint.

    ``keep_best`` restores the best-scoring weights into ``model`` at the end, so
    the caller receives the checkpoint the metrics describe rather than whatever
    the last step happened to produce.
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
    model.train()
    started = time.perf_counter()

    for step in range(config.steps):
        lr = lr_at(step, config.steps, config.lr, config.warmup)
        for group in optimiser.param_groups:
            group["lr"] = lr

        x, y = train_sampler.batch(config.batch_size, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        result.final_train_loss = loss.item()

        if step % config.log_every == 0 or step == config.steps - 1:
            elapsed = time.perf_counter() - started
            result.tokens_per_sec = (step + 1) * config.batch_size * train_sampler.seq_len / elapsed
            record = {
                "step": step,
                "loss": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": result.tokens_per_sec,
                "elapsed": elapsed,
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
                if keep_best:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)

    result.seconds = time.perf_counter() - started
    return result
