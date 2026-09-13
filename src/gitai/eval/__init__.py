"""Evaluation: turning training runs into comparable measurements.

The difference between "I trained a model" and "I produced a result" lives
here. Loss going down is not a finding.

The module that matters most is :mod:`~gitai.eval.stats`: small models are
noisy, and an effect smaller than the measured seed spread does not exist no
matter how clean the plot looks.
"""

from .harness import evaluate, frequent_tokens, generate_samples
from .index import build_index, connect
from .metrics import EVAL_SUITE_VERSION, EvalResult, accuracy_at_k, perplexity
from .probes import (
    TASKS,
    CopyTask,
    DyckTask,
    InductionProbe,
    ModularArithmeticTask,
    Probe,
    SortTask,
    SyntheticTask,
    evaluate_task,
)
from .report import compare_named, comparison_table, leaderboard, load_groups, noise_floor
from .stats import Comparison, SeedGroup, compare_groups, permutation_test

__all__ = [
    "EVAL_SUITE_VERSION",
    "TASKS",
    "Comparison",
    "CopyTask",
    "DyckTask",
    "EvalResult",
    "InductionProbe",
    "ModularArithmeticTask",
    "Probe",
    "SeedGroup",
    "SortTask",
    "SyntheticTask",
    "accuracy_at_k",
    "build_index",
    "compare_groups",
    "compare_named",
    "comparison_table",
    "connect",
    "evaluate",
    "evaluate_task",
    "frequent_tokens",
    "generate_samples",
    "leaderboard",
    "load_groups",
    "noise_floor",
    "permutation_test",
    "perplexity",
]
