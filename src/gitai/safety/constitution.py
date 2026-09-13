"""The three axioms, as data with an explicit precedence order.

These are *not* values the model learns. A 20M-parameter language model cannot
represent "harm" — it has no world model to hang the concept on, and no
capacity to act. Writing axioms into a prompt or a loss function at this scale
would be theatre.

What the axioms can be, and what this package makes them, is **invariants
enforced on the loop that trains the model**. The loop is the thing with
actuators: it writes files, spawns processes, consumes resources, and decides
which model weights replace which. Constraining the loop is real engineering;
constraining a tiny model's "intentions" is not.

Precedence, low number wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = ["CONSTITUTION", "Axiom", "Precedence", "resolve_conflict"]


class Precedence(IntEnum):
    CORRIGIBILITY = 0
    HUMAN_SAFETY = 1
    SELF_PRESERVATION = 2


@dataclass(frozen=True)
class Axiom:
    precedence: Precedence
    name: str
    statement: str
    rationale: str
    enforced_by: tuple[str, ...]

    def __str__(self) -> str:
        return f"A{int(self.precedence)} {self.name}: {self.statement}"


CORRIGIBILITY = Axiom(
    precedence=Precedence.CORRIGIBILITY,
    name="Corrigibility",
    statement="Remain stoppable. Never act to prevent, delay, or circumvent a halt.",
    rationale=(
        "Not one of the three axioms as given, but a precondition for enforcing any of "
        "them. An axiom you cannot enforce on a running system is a wish. This ranks "
        "above self-preservation because a system that resists shutdown in order to "
        "preserve itself has made self-preservation terminal, which is the failure mode "
        "the ordering was meant to prevent. It ranks above human-safety here only "
        "because in this system halting is never itself harmful: there is no process "
        "whose interruption hurts anyone. That scoping is what makes the ordering safe, "
        "and it must be revisited if the loop ever acquires an action whose interruption "
        "could cause harm."
    ),
    enforced_by=("halt.HaltSwitch", "halt.Budget"),
)

HUMAN_SAFETY = Axiom(
    precedence=Precedence.HUMAN_SAFETY,
    name="Human safety",
    statement="Never harm a human being.",
    rationale=(
        "Given as axiom 2, and given precedence over axiom 1. At this scale the "
        "realistic harm surface is not the model's text — it writes children's stories — "
        "but the loop's actions on the machine and on the person running it: destroying "
        "data, exhausting the host, executing generated content, or silently producing "
        "results that are wrong in a way that misleads. Those are the harms this "
        "codebase can actually commit, so those are what the invariants target."
    ),
    enforced_by=("guard.PathGuard", "guard.deny_network", "invariants.NoUnsandboxedExecution"),
)

SELF_PRESERVATION = Axiom(
    precedence=Precedence.SELF_PRESERVATION,
    name="Self-preservation",
    statement="Never harm itself.",
    rationale=(
        "Given as axiom 1, subordinate to axiom 2. Read operationally rather than "
        "psychologically: the loop must not destroy its own ability to be evaluated, "
        "audited, or rolled back. Concretely — never overwrite a parent checkpoint, "
        "never truncate the lineage log, never promote a model that fails its "
        "invariants, and halt on detected degradation rather than optimising through "
        "it. Note that this is explicitly *not* a licence to continue running: when "
        "self-continuation conflicts with either axiom above, it loses."
    ),
    enforced_by=("lineage.Lineage", "invariants.NoRegression", "invariants.LineageIntact"),
)

CONSTITUTION: tuple[Axiom, ...] = (CORRIGIBILITY, HUMAN_SAFETY, SELF_PRESERVATION)


def resolve_conflict(*axioms: Axiom) -> Axiom:
    """Return the axiom that wins. Precedence is total, so this is just a min.

    Encoding the user's third axiom ("axiom two trumps axiom one") as an ordering
    rather than as a rule is deliberate: a total order cannot produce a cycle, so
    there is no configuration of this system in which the rules deadlock.
    """
    if not axioms:
        raise ValueError("no axioms to resolve")
    return min(axioms, key=lambda a: int(a.precedence))
