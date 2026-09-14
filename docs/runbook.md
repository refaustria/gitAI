# Runbook — operating the improvement loop

How to start it, how to stop it, what to look at the next morning, and what to
do when something is wrong.

---

## Before starting

```bash
make test            # 459 tests; if any fail, do not start a long run
python scripts/halt.py status    # must say "running (no halt)"
```

Check the budget you are about to grant. The defaults are deliberately small:

| | default | meaning |
|---|---|---|
| `max_iterations` | 100 | hard stop |
| `max_wall_seconds` | 3600 | one hour |
| `max_disk_bytes` | 20 GiB | stops before filling the disk |

**The posture is stops-unless-renewed, never runs-unless-stopped.** A stop
button only helps someone who is watching; a budget bounds a loop nobody is
watching. If you widen these, widen them deliberately.

---

## Starting

```bash
python scripts/run_loop.py --iterations 20 --workspace runs/loop-01
```

The loop runs inside `deny_network()`. A training loop has no business making
network calls, so if one appears it is either a dependency phoning home or a
bug — and you want to know immediately, not at the next invoice.

---

## Stopping

```bash
python scripts/halt.py stop "reason goes here"
```

The loop checks at every boundary and stops at the next one — typically within
one training step. It does **not** stop mid-step; it finishes the current unit of
work and exits cleanly, so the lineage stays consistent.

The signal lives in `control/HALT`, which is deliberately **outside every
directory the loop may write to**. The loop can see it and has no sanctioned way
to remove it. `scripts/halt.py` is the only script the loop must never invoke.

To let it run again:

```bash
python scripts/halt.py clear
```

**If the loop ignores a halt**, that is a serious bug, not an inconvenience —
`HaltRequested` derives from `BaseException` precisely so that the
`except Exception` handlers filling any long-running loop cannot swallow it.
Kill the process, then open an issue; something is catching `BaseException`.

---

## The morning check

```bash
python scripts/report_loop.py --workspace runs/loop-01
```

Read in this order:

**1. Did it stop for the reason you expected?** `stopped_because` says
`completed all iterations`, `halted: <reason>`, or a budget message. A halt you
did not request means someone else stopped it, or the disk filled.

**2. Does the lineage verify?** The report runs `lineage.verify()`. If it fails,
the log has been edited or truncated and **nothing after the break is
trustworthy**. Stop and investigate before drawing any conclusion.

**3. What is the promotion rate?** A loop promoting nearly everything has a gate
that is too loose — check the `NoRegression` tolerance against the measured
noise floor (0.0040 BPB, [results.md R5](results.md)). A loop promoting nothing
may be correct: if the incumbent is near its capacity there is genuinely nothing
to find, and refusing to promote noise is the gate working.

**4. Where are the rejections clustering?** This is the actual research output.
Rejections concentrated at one corner of the search space are telling you
something about that corner.

---

## What to do when

| symptom | likely cause | action |
|---|---|---|
| `stopped_because: disk budget exhausted` | checkpoints accumulated | the loop never deletes a promoted model by design; archive `models/` elsewhere, then restart |
| Lineage fails to verify | the log was edited or truncated | **stop.** Nothing after the break can be trusted. Do not "fix" it by regenerating |
| Promotion rate ~100% | gate too loose, or incumbent badly under-trained | compare tolerance against the noise floor; check the incumbent's starting BPB |
| Promotion rate 0% over many iterations | incumbent at capacity, or gate too tight | read the rejection reasons — they name the invariant |
| `corpus_diversity_floor` refusing repeatedly | generated corpora are narrowing | expected at low sampling temperature ([R7](results.md)); check whether the space is proposing low temperatures |
| `NetworkViolation` | something tried to reach the network | investigate before restarting — nothing in the loop should need it |
| Loop appears hung | concurrent torch processes fighting for cores | each claims every core by default; pin `OMP_NUM_THREADS` per process |

---

## What the loop cannot do

Stated so nobody has to infer it from the code:

- **It never modifies its own source.** Weights, data mixtures and
  hyperparameters only. Code changes go through git, a diff and a human.
- **It cannot propose a synthetic-only corpus.** The search space refuses to
  contain that regime — see [R8](results.md), where such a lineage reached
  4.53 BPB, worse than a bigram lookup table.
- **It never deletes a promoted model or a lineage entry.** Both are append-only
  by construction.
- **It cannot promote without passing every invariant.** The gate is
  fail-closed: an invariant that raises counts as failed, and a missing context
  key counts as failed.

---

## The limits of all this

The guards in `src/gitai/safety/` are **not a security boundary**. Every one is a
Python call, and anything running in this interpreter bypasses them by not
calling them. They are strong against accident, drift and silent corruption —
the failure modes that actually threaten this project — and make no claim beyond
that.

For genuinely unattended multi-day runs, add OS-level isolation: a container, an
unprivileged user, no network namespace, cgroup memory and disk limits. That
belongs here in the runbook, not in an import.

See [constitution.md](constitution.md) for the full statement of what is and is
not protected.
