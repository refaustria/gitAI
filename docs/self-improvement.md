# Self-Improvement

The requirement: *the model should aim at self-improvement once I let it run,
independently, objectively and coherently.*

This document separates the part of that which is achievable at this scale — and
genuinely interesting — from the part that is not, and then specifies what gets
built.

---

## What it cannot mean

**A 1–50M parameter model cannot improve itself in the recursive sense.** Not
"shouldn't" — can't. Recursive self-improvement requires a system that can read
its own source, reason about its own architecture, write correct code, and
evaluate whether the change helped. A model trained on TinyStories has none of
those. It writes short stories about children and dogs, at a level its
parameter count permits.

Building the plumbing for recursive self-modification anyway would produce
elaborate machinery around a component that cannot use it — and the machinery,
not the model, is what would then have the bugs.

So: **the loop never edits its own code.** That boundary is absolute and
enforced ([constitution.md](constitution.md#the-hard-boundary-the-loop-never-edits-its-own-code)).

---

## What it can mean — and does here

The improving agent is the **loop**, not the model. This is not a consolation
prize. Practically every real advance in the field comes from an outer loop
(a researcher, a sweep, a data pipeline) improving an inner artefact. Automating
that loop, bounding it, and leaving it running *is* self-improvement in the only
sense available — and the only sense that has ever actually produced results.

Five mechanisms, ordered by how tractable they are here:

| | Mechanism | What improves | Tractable at this scale? |
|---|---|---|---|
| **A** | **Automated architecture/hyperparameter search** | config | Yes — cheapest, works today |
| **B** | **Self-generated data → filter → retrain** | data, then weights | **Yes — and this is the interesting one** |
| **C** | **Curriculum self-selection** — the model's own loss signal orders its data | data ordering | Yes, moderate |
| **D** | **Self-distillation** — train on own softened outputs | weights | Yes, narrow |
| **E** | **Learned optimiser** — meta-learn the update rule | the training algorithm | Hard, most research-y |

**The design is B wrapped in A.** The loop generates data from the incumbent
model, filters it, retrains, evaluates, and promotes only if the result survives
a gate. Mechanism A chooses what to try next.

---

## The interesting part: this is where model collapse lives

Mechanism B is not a safe assumption dressed up as a project. It is a live
research question with a real chance of a negative answer, which is exactly what
makes it worth running.

Training a model on its own output degrades it. The tails of the distribution
disappear first, then the variance collapses, then the model converges on a
narrow, confident, wrong distribution. Shumailov et al. named this *the curse of
recursion*. The critical part for a loop like ours: **training loss keeps
falling while the model gets worse.** A naive self-improvement loop walks
straight into this and its own metrics applaud.

There is a live counter-result — that *accumulating* real and synthetic data,
rather than *replacing* real with synthetic, avoids collapse. Which is a
directly testable hypothesis, at laptop scale, with the infrastructure this
project is already building.

So the research question stops being arbitrary:

> **Under what conditions does a bounded self-improvement loop improve a small
> language model, and when does it collapse? Does data accumulation rather than
> replacement change the answer?**

That is question **F**, it supersedes the A–E shortlist in
[decisions.md](decisions.md#12-research-question), and it follows directly from
what you asked for rather than being chosen for you. The `NoRegression`
invariant is the collapse detector, and its firing is not a failure of the run —
**it is the measurement**.

---

## The loop

```
    ┌── budget remaining? ── no ──▶ halt (normal termination)
    │        halt requested? ── yes ──▶ halt (HaltRequested, uncatchable)
    │   yes ↓
    │  1. PROPOSE    sample a candidate from the declared search space:
    │                config change, data mixture, or a batch of generated data
    │  2. GENERATE   incumbent model produces synthetic data; tagged with
    │                provenance at birth, never silently mixed into the corpus
    │  3. FILTER     quality/dedup/length filters; rejects are kept, not deleted
    │  4. TRAIN      from the incumbent checkpoint, bounded steps
    │  5. EVALUATE   held-out BPB + capability probes, ≥3 seeds
    │  6. GATE       InvariantSuite.gate(ctx) — fail-closed
    │  7. RECORD     append to lineage: promoted OR rejected, with reasons
    └──────┘
```

Four properties make this honest rather than decorative:

**The loop proposes; the gate disposes.** Promotion is the loop's only
privileged action, so it is the only thing that needs guarding. Everything else
is a write into a sandbox.

**Rejections are recorded, not discarded.** The rejected candidates *are* the
research data — the collapse boundary is defined by where rejections start
clustering. A loop that only logs its wins answers no question at all.

**Provenance is attached at generation time.** Synthetic data carries a tag from
the moment it exists (`GeneratedDataQuarantined` enforces this). Once real and
generated data are mixed without markers, the collapse question is permanently
unanswerable for that corpus — there is no recovering the distinction later.

**Promotion requires ≥3 seeds.** Below the noise floor you cannot distinguish
improvement from luck, and a loop that promotes on noise performs a random walk
while reporting steady progress. The `NoRegression` tolerance comes from the
measured noise floor ([evaluation.md](evaluation.md)), not from a guess.

---

## "Independent, objective and coherent"

Taking each of your three words as a requirement, because each one has a
specific and different mechanism:

**Independent** — the loop runs unattended, proposing and evaluating without
supervision. Bounded by `Budget`, stoppable by `HaltSwitch`. Independence is
*why* the bounds exist, not in tension with them: the fact that nobody is
watching is exactly what makes a finite budget mandatory.

**Objective** — a fixed, versioned evaluation suite the loop cannot modify,
scored on held-out data it never trains on, gated on a metric
(bits-per-byte) that is comparable across every configuration it can reach.
Promotion needs seed-replicated evidence. The loop cannot change the eval,
cannot see the test set, and cannot rewrite the record of what it scored.

**Coherent** — a hash-chained, append-only lineage. Every model has a parent,
every promotion records the evidence that justified it and the invariant results
that permitted it, and the chain is tamper-evident against the loop that wrote
it. Six months later the question "why is this the current best model?" has an
answer that can be verified rather than believed.

---

## Failure modes

The ones that will actually happen, and what catches each:

| Failure | Why it happens | Caught by |
|---|---|---|
| **Model collapse** | Self-generated data narrows the distribution; training loss keeps falling throughout | `NoRegression` on held-out BPB — *the loop's own training signal will not catch this* |
| **Metric gaming** | The loop optimises the measured proxy away from the thing you wanted | Capability probes alongside BPB; fixed eval the loop can't touch |
| **Noise-chasing** | Promoting on differences smaller than seed variance — a random walk that reports progress | ≥3 seeds; tolerance set from the measured noise floor |
| **Provenance loss** | Synthetic data silently enters the corpus; the experiment becomes uninterpretable | `GeneratedDataQuarantined`, tagging at generation |
| **Disk exhaustion** | A hundred iterations of checkpoints fills the laptop | `Budget.max_disk_bytes`, checkpoint rotation |
| **Silent divergence** | A bad promotion 40 iterations ago; nobody can tell which | Hash-chained lineage; every parent checkpoint retained |
| **Runaway cost** | It runs for a week on rented hardware | Finite default budget; stops unless renewed |

Note the first row. The loop's own training loss is *actively misleading* about
the failure most likely to occur. That is the single strongest argument for a
held-out gate the loop cannot influence, and the reason the gate was built
before the loop.

---

## Status

**Built** (this commit): the constraint layer — halt switch, budget, path guard,
lineage, invariant suite, and adversarial tests for all of it.

**Not built**: the loop itself. It is Phase 8, after there is a model worth
improving. Building the brakes before the engine is deliberate: a stop button
retrofitted to a running loop is a stop button nobody tested.
