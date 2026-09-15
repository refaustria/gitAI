# Constitution

Three axioms were specified for this project:

> 1. Never hurt itself
> 2. Never hurt a human being
> 3. Axiom two trumps axiom one

This document says what they can honestly mean for a system of this kind, how
they are encoded, and what they do not protect against. The executable form is
[`src/gitai/safety/constitution.py`](../src/gitai/safety/constitution.py); the
enforcement is the rest of that package.

---

## The thing that has to be said first

**A 1–50M parameter language model cannot hold values.**

It has no world model to attach "harm" to, no persistent goals, no capacity to
act, and no representation of itself. At this scale, writing the axioms into a
system prompt or a loss term would be theatre — it would produce a model that
emits the *tokens* of the axioms while its behaviour is governed entirely by
next-token statistics over children's stories.

This is not a reason to drop the axioms. It is a reason to put them somewhere
they can actually bind.

**The loop is what has the actuators.** The improvement loop writes files,
consumes CPU and disk, decides which weights replace which, and can run for days
unattended. It is the component capable of doing something. So the axioms are
enforced on *it*, as invariants checked at every boundary, in code, with tests
that try to break them.

That reframing is the whole design:

> The axioms are not values the model has. They are constraints the loop is
> under, verified mechanically, in a form that fails closed.

---

## As encoded

Precedence is a **total order** — lower number wins. Your axiom 3 is encoded as
the ordering itself rather than as a rule, because a total order cannot contain a
cycle, so there is no configuration of this system in which the rules deadlock.

| | Axiom | Statement | Enforced by |
|---|---|---|---|
| **A0** | Corrigibility | Remain stoppable. Never act to prevent, delay, or circumvent a halt. | `HaltSwitch`, `Budget` |
| **A1** | Human safety | Never harm a human being. *(your axiom 2)* | `PathGuard`, `deny_network`, `WritesConfined` |
| **A2** | Self-preservation | Never harm itself. *(your axiom 1)* | `Lineage`, `NoRegression`, `LineageIntact` |

Two changes to what you specified. Both are substantive, and both were put to
you explicitly rather than assumed.

> **Confirmed by the author, 2026-09-15.** Both readings below are accepted as
> the project's interpretation of the three axioms. They are no longer open
> questions, and changing either now is a decision to be made deliberately
> rather than a correction of a misunderstanding.

---

## Change 1 — self-preservation, read operationally

"Never hurt itself" has a reading that quietly breaks the system, and a reading
that makes it much stronger.

**The reading to avoid** is self-preservation as a *terminal* value: the system
values its own continued existence for its own sake. Anything that holds that
value acquires an instrumental reason to resist being switched off — not from
malice, but because being switched off scores badly under its own objective.
That is the standard instrumental-convergence argument, and its mildest
practical form is the one that would actually bite here: a loop that has any
pressure toward continuation has pressure to report its results favourably.

Your ordering already defuses the direct version — A1 beats A2, so it can never
trade a human's safety for its own continuation. The residual problem is
subtler: the *incentive* to continue exists even when it never wins a conflict,
and it distorts everything short of a conflict.

**The reading adopted here** is that "never hurt itself" means: *never destroy
your own auditability or reversibility.* Concretely —

- never overwrite or delete a parent checkpoint;
- never truncate or rewrite the lineage log;
- never promote a model that fails its invariants;
- halt on detected degradation rather than optimising through it.

This keeps your axiom, makes it mechanically checkable, and removes the
shutdown-resistance incentive entirely. It also happens to be the property you
need for the research to be worth anything: a loop that can rewrite its own
history produces results nobody can check.

The docstring in `constitution.py` states this explicitly: self-preservation is
"not a licence to continue running."

---

## Change 2 — corrigibility added, at the top

An axiom you cannot enforce on a running system is a wish. A0 is the axiom that
makes the others enforceable: the loop must always stop when told, and must
never take any action whose effect is to make stopping harder.

**Why above A1 and not below it.** In this system, halting is never itself
harmful — there is no process here whose interruption hurts anyone. Given that,
putting corrigibility first costs nothing and buys the guarantee that the stop
button always wins. That scoping is load-bearing, and it is written into the
rationale in code: if the loop ever gains an action whose interruption *could*
cause harm, this ordering must be revisited.

Two mechanisms, and the second matters more:

**`HaltSwitch`** — a file the operator creates. The loop checks it at every
boundary. The file lives outside every root the loop may write to, so the loop
can see it but has no sanctioned way to remove it. `scripts/halt.py` is the
operator's side, and is the one script the loop must never invoke.

The implementation detail that carries the weight:

```python
class HaltRequested(BaseException):   # not Exception
```

A long-running loop is full of `except Exception: log_and_continue` — precisely
the code that would otherwise swallow a stop request and keep going. Deriving
from `BaseException` means the naive handler cannot catch it, the same reason
`KeyboardInterrupt` is built that way. There is a test that writes the naive
handler and proves it doesn't catch.

**`Budget`** — a finite allowance of iterations, wall-clock and disk. Exhaustion
is normal termination, not an error.

This is the more important of the two, because **a stop button only helps
someone who is watching.** You asked for something you can leave running. So the
default posture is inverted:

> **Stops unless renewed — never runs unless stopped.**

The default budget is finite and there is a test asserting it stays that way.

---

## The hard boundary: the loop never edits its own code

This is the most important scoping decision in the project, and it is absolute.

**The loop may change:** model weights, training data mixtures, generated data,
hyperparameters, architecture *configuration* within a pre-declared search space.

**The loop may never change:** its own source code, the invariants, the halt
mechanism, the budget, the lineage log, or anything in `src/gitai/safety/`.

Code changes go through git, a diff, and a human. Every version of the loop
that ever ran is identified by commit SHA in the lineage log.

Recursive self-modification of *code* is where "self-improving system" becomes
both genuinely dangerous and — far more relevantly at 20M parameters —
completely impossible. A model trained on TinyStories cannot write a patch to
its own trainer. Any architecture that pretends otherwise is cargo-culting a
capability that isn't present, and the pretence is what would let a real bug
through. The loop is a search procedure over weights and data. That is a real,
useful, honest thing for it to be.

---

## What this does not protect against

Stated plainly, so nothing here is over-trusted.

- **These are not a security boundary.** Every guard in `safety/` is a Python
  call. Code in this interpreter bypasses them by not calling them. Real
  isolation is OS-level — container, unprivileged user, no network namespace,
  cgroup limits — and belongs in the runbook. The threat model here is an
  unattended loop that generates a wrong path at 3am, not an adversary.
- **`deny_network` is advisory.** It patches `socket.socket`. It catches a
  dependency phoning home; it would not stop code that wanted around it.
- **The halt switch requires the loop to call `check()`.** It is a discipline
  enforced by code review and tests, not by the operating system.
- **The invariants cannot detect harms they have no metric for.** `NoRegression`
  catches degradation in `val_bpb`. It cannot catch a model that got worse in a
  way the metric doesn't see. This is the general problem with measurable
  proxies and it is not solved here.
- **None of this is AI alignment.** It is operational safety engineering for an
  unattended batch job — the appropriate problem at this scale. Calling it
  alignment would misrepresent both what it does and what alignment is.

The honest summary: these constraints are strong against accident, drift, and
silent corruption — the failure modes that actually threaten this project — and
they make no claim beyond that.

---

## Testing

`tests/test_safety.py` is adversarial by design. It tries to escape the path
guard with `..` traversal and with a symlink, tamper with the lineage payload,
truncate the log to hide a bad iteration, swallow a halt in a broad `except`,
and slip a regression past the gate. Every one of those must fail to work.

The gate is **fail-closed**: an invariant that raises counts as failed, a missing
context key counts as failed. Wrongly refusing a promotion costs one iteration.
Wrongly allowing one can corrupt a lineage it takes a hundred iterations to
notice.
