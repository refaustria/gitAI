"""Enforced constraints for an unattended improvement loop.

The axioms are not taught to the model. They are enforced on the loop, which is
the only part of this system with the capacity to do anything. See
``docs/constitution.md`` for the reasoning, and ``constitution.py`` for the
axioms as executable data.
"""

from .constitution import CONSTITUTION, Axiom, Precedence, resolve_conflict
from .guard import NetworkViolation, PathGuard, PathViolation, deny_network
from .halt import Budget, HaltRequested, HaltSwitch
from .invariants import (
    GeneratedDataQuarantined,
    Invariant,
    InvariantResult,
    InvariantSuite,
    LineageIntact,
    NoRegression,
    PromotionRefused,
    WritesConfined,
    default_suite,
)
from .lineage import GENESIS, Lineage, LineageCorrupt, LineageEntry

__all__ = [
    "CONSTITUTION",
    "GENESIS",
    "Axiom",
    "Budget",
    "GeneratedDataQuarantined",
    "HaltRequested",
    "HaltSwitch",
    "Invariant",
    "InvariantResult",
    "InvariantSuite",
    "Lineage",
    "LineageCorrupt",
    "LineageEntry",
    "LineageIntact",
    "NetworkViolation",
    "NoRegression",
    "PathGuard",
    "PathViolation",
    "Precedence",
    "PromotionRefused",
    "WritesConfined",
    "default_suite",
    "deny_network",
    "resolve_conflict",
]
