"""Phase 8: the bounded self-improvement loop.

What improves is the loop, not the model. See docs/self-improvement.md for the
design and docs/constitution.md for the constraints it runs under.
"""

from .loop import ImprovementLoop, IterationOutcome, LoopConfig, LoopResult
from .space import OutOfSpace, Proposal, SearchSpace

__all__ = [
    "ImprovementLoop",
    "IterationOutcome",
    "LoopConfig",
    "LoopResult",
    "OutOfSpace",
    "Proposal",
    "SearchSpace",
]
