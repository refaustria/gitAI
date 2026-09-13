"""Terminal rendering for the interpretability tools.

Numbers in a table are hard to read across a whole passage; colour is not. These
helpers exist so that "where is the model struggling?" is answerable at a glance
rather than by scanning a column of floats.

Everything degrades to plain text when ``colour=False`` or the output is not a
terminal, so the same functions work in a pipe or a log file.
"""

from __future__ import annotations

import sys

import torch

__all__ = [
    "ablation_table",
    "attention_heatmap",
    "attribution_table",
    "colour_by_surprisal",
    "lens_table",
    "visible",
]

RESET = "\033[0m"
BLOCKS = " ░▒▓█"

# Low surprisal (predictable) through high (blindsided).
SURPRISAL_COLOURS = [
    (1.0, 34),  # deep green
    (2.5, 40),  # green
    (4.0, 190),  # yellow-green
    (6.0, 220),  # amber
    (9.0, 208),  # orange
    (float("inf"), 196),  # red
]


def _use_colour(colour: bool | None) -> bool:
    return sys.stdout.isatty() if colour is None else colour


def visible(text: str) -> str:
    """Make whitespace visible, so a token boundary is never ambiguous."""
    return text.replace("\n", "⏎").replace("\t", "→").replace(" ", "·")


def _paint(text: str, code: int, colour: bool) -> str:
    return f"\033[38;5;{code}m{text}{RESET}" if colour else text


def colour_by_surprisal(predictions, colour: bool | None = None, width: int = 88) -> str:
    """Render a passage coloured by how surprised the model was at each token.

    Green is predictable, red is blindsided. Reading a page of this tells you
    more about a model's competence in five seconds than the loss value does —
    it shows *where* the loss was incurred, not just how much.
    """
    use = _use_colour(colour)
    out, line = [], 0
    for prediction in predictions:
        code = next(c for threshold, c in SURPRISAL_COLOURS if prediction.surprisal < threshold)
        token = visible(prediction.actual_token)
        if line + len(token) > width:
            out.append("\n")
            line = 0
        out.append(_paint(token, code, use))
        line += len(token)

    legend = "  ".join(
        _paint(label, code, use)
        for label, code in [
            ("0-1 bits", 34),
            ("1-2.5", 40),
            ("2.5-4", 190),
            ("4-6", 220),
            ("6-9", 208),
            ("9+", 196),
        ]
    )
    return "".join(out) + f"\n\nsurprisal: {legend}"


def attention_heatmap(
    pattern: torch.Tensor, tokens: list[str], max_tokens: int = 24, colour: bool | None = None
) -> str:
    """One head's attention as a text heatmap. Rows are queries, columns keys.

    Read it as: "when predicting at row *i*, how much did the model look at
    column *j*?" The lower triangle is all there is — the upper is masked.

    Remember that attention is suggestive, not explanatory: a head attending to a
    token does not prove that token caused the output.
    """
    if pattern.dim() != 2:
        raise ValueError(f"expected a (query, key) matrix, got {tuple(pattern.shape)}")
    use = _use_colour(colour)
    n = min(max_tokens, pattern.shape[0], len(tokens))
    grid = pattern[:n, :n].float()

    labels = [visible(t)[:6].rjust(6) for t in tokens[:n]]
    lines = []
    for row in range(n):
        cells = []
        for col in range(n):
            value = float(grid[row, col])
            glyph = BLOCKS[min(int(value * len(BLOCKS)), len(BLOCKS) - 1)]
            if use and value > 0.5:
                glyph = _paint(glyph, 220, True)
            cells.append(glyph)
        lines.append(f"{labels[row]} │{''.join(cells)}")
    footer = "       └" + "─" * n
    return "\n".join([*lines, footer])


def lens_table(rows, target_token: str, colour: bool | None = None) -> str:
    """The logit lens as a table: the prediction forming, layer by layer."""
    use = _use_colour(colour)
    label = "p(" + visible(target_token)[:8] + ")"
    header = f"{'layer':<10} {label:>12} {'rank':>6}   top predictions"
    lines = [header, "-" * (len(header) + 12)]
    for row in rows:
        top = "  ".join(f"{visible(t)[:10]!r}:{p:.2f}" for t, p in row.top)
        probability = f"{row.target_prob:.4f}"
        if use and row.target_rank == 1:
            probability = _paint(probability, 40, True)
        lines.append(f"{row.label:<10} {probability:>12} {row.target_rank:>6}   {top}")
    return "\n".join(lines)


def attribution_table(contributions, offset: float, top: int = 12) -> str:
    """Per-component contributions as a signed bar chart."""
    ordered = sorted(contributions, key=lambda c: -abs(c.logit))[:top]
    span = max((abs(c.logit) for c in ordered), default=1.0) or 1.0
    lines = [f"{'component':<12} {'logit':>9}   contribution", "-" * 56]
    for contribution in ordered:
        bar_width = int(abs(contribution.logit) / span * 22)
        bar = ("█" * bar_width).rjust(22) if contribution.logit < 0 else "█" * bar_width
        side = "◀" if contribution.logit < 0 else " "
        lines.append(f"{contribution.name:<12} {contribution.logit:>9.3f}  {side}{bar}")
    total = sum(c.logit for c in contributions) + offset
    lines.append("-" * 56)
    lines.append(f"{'sum':<12} {total:>9.3f}   (+ {offset:.3f} from the final norm bias)")
    return "\n".join(lines)


def ablation_table(deltas: torch.Tensor, top: int = 10) -> str:
    """Heads ranked by how much removing them hurts."""
    flat = [
        (float(deltas[layer, head]), layer, head)
        for layer in range(deltas.shape[0])
        for head in range(deltas.shape[1])
    ]
    flat.sort(key=lambda item: -item[0])
    lines = [f"{'head':<10} {'Δ loss':>9}   importance", "-" * 48]
    span = max((abs(d) for d, _, _ in flat), default=1.0) or 1.0
    for delta, layer, head in flat[:top]:
        lines.append(f"L{layer}H{head:<7} {delta:>9.4f}  {'█' * int(abs(delta) / span * 24)}")
    return "\n".join(lines)
