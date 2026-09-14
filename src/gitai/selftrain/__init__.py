"""Phase 5: does a self-improvement loop improve a small model, or collapse it?

The loop trains a model, has it generate a corpus, trains the next generation on
that corpus, and repeats. Three arms differ only in what the next generation is
trained on — real data (control), the parent's output (replace), or both
(accumulate).

Everything is scored on held-out **real** data, because a collapsing model gets
steadily better at predicting its own output while getting worse at predicting
reality, and only the real held-out set distinguishes those.
"""

from .arms import ARMS, Accumulate, Arm, Control, Replace
from .corpus import CorpusStats, diversity, generate_corpus

__all__ = [
    "ARMS",
    "Accumulate",
    "Arm",
    "Control",
    "CorpusStats",
    "Replace",
    "diversity",
    "generate_corpus",
]
