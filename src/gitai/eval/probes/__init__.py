"""Capability probes: what can the model actually do?

Loss hides structure. Two models at identical loss can differ enormously in
capability, and these ask specific questions with unambiguous answers.
"""

from .base import Probe, SyntheticTask
from .induction import InductionProbe
from .synthetic import (
    TASKS,
    CopyTask,
    DyckTask,
    ModularArithmeticTask,
    SortTask,
    evaluate_task,
)

__all__ = [
    "TASKS",
    "CopyTask",
    "DyckTask",
    "InductionProbe",
    "ModularArithmeticTask",
    "Probe",
    "SortTask",
    "SyntheticTask",
    "evaluate_task",
]
