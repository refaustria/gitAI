# Evaluation & Research Methodology

The difference between "I trained a model" and "I produced a result" is entirely
in this document. Loss going down is not a finding.

---

## Part 1 — Metrics

### Bits-per-byte is the primary metric

**Perplexity is measured per token, so it is not comparable across tokenizers.**
A model with a 4k vocab and one with an 8k vocab predict differently-sized units;
their perplexities cannot be meaningfully compared even on identical text.

Because vocabulary size is an explicit variable in this project
([Decision 4](decisions.md#4-tokenizer), and possibly the whole research question
under 12-B), reporting perplexity as the headline number would make your own
experiments incomparable **with each other**. That mistake has quietly invalidated
a lot of published small-model work.

Bits-per-byte normalises by raw UTF-8 bytes of the original text:

```
BPB = (total_cross_entropy_nats / ln(2)) / total_utf8_bytes
```

It is tokenizer-independent, comparable across every configuration you will run,
and comparable to numbers in the literature. Report perplexity alongside it if
you want — just never compare across vocab settings with it.

### The full metric set

| Metric | Why |
|---|---|
| **Bits-per-byte** (held-out) | Primary. Tokenizer-independent |
| Val loss / perplexity | Familiar; comparable only within one tokenizer |
| **Tokens/sec, FLOPs, wall-clock** | Compute-matched comparison is impossible without these |
| Peak RAM | Your binding constraint on a laptop |
| Parameters (embedding vs non-embedding, split out) | At small scale these differ enormously — always report both |

> Always report **non-embedding** parameter count separately. At a 4k vocab and
> `d_model=384`, embeddings are ~1.5M parameters. Comparing a "10M model" that
> spends 1.5M on embeddings against one that spends 6M is comparing nothing.

---

## Part 2 — Capability probes

Loss is a summary statistic and it hides structure. Two models at identical loss
can have very different abilities. Probe for specific capabilities:

### Synthetic probes (cheap, sharp, unambiguous)

| Probe | Measures |
|---|---|
| **Copying** — reproduce a random sequence seen earlier in context | Induction heads. The clearest known emergent circuit |
| **Sorting** — sort a short list of integers | Algorithmic generalisation |
| **Modular arithmetic** | Whether a real algorithm was learned or a lookup table memorised |
| **Dyck languages** — balanced brackets | Recursive/hierarchical structure |
| **Needle-in-haystack** at varying context length | Effective, as opposed to nominal, context |

These are cheap, have unambiguous correctness, and — unlike loss — tell you
*what* the model can do. They are the backbone of question D in
[Decision 12](decisions.md#12-research-question--open--needs-your-decision).

### Generation quality (TinyStories-style)

For natural-language runs, hold a fixed set of ~50 prompts, generate at fixed
temperature and seed, and score along three axes: **grammar**, **consistency**,
**creativity**. Score with a larger model as judge, or by hand for small batches.

Caveats, both real: LLM-as-judge is noisy and biased toward its own style, and
hand-scoring drifts as your expectations shift. Use it for coarse signal
("clearly better / clearly worse"), never to defend a small difference.

### Qualitative samples — keep them, always

Every eval writes a fixed-seed, fixed-prompt sample file. Numbers hide
catastrophic failure modes; a page of generated text does not. Reading samples
across a training run is also the single most motivating part of this project.

---

## Part 3 — Methodology (the part that makes it research)

### 1. Change one variable at a time

Obvious, universally violated under time pressure. A run differing in two
variables produces zero bits of information about either. Every experiment names
its single independent variable in its config; a sweep varies exactly one axis.

### 2. Establish the seed noise floor before claiming anything

**Do this before your first real experiment.** Train the same configuration with
3–5 different seeds and measure the spread in final validation loss. That spread
is your noise floor.

Any effect smaller than it does not exist, no matter how clean the plot looks.
Small models are *noisy*; this step will save you from at least one false
discovery, and probably several. Every reported result carries ≥3 seeds, with
mean and spread.

### 3. Compute-matched comparison, not step-matched

Comparing model A at 10k steps against model B at 10k steps is meaningless if B's
steps cost twice as much. Compare at equal **FLOPs** or equal **wall-clock** —
that is the comparison that answers "which should I actually use?"

This matters more here than at scale, because your architectural variations
(depth vs width especially) have very different cost profiles on CPU.

#### Never let wall-clock terminate an experimental arm

There is a trap in the sentence above. Comparing *at* equal wall-clock is sound.
Letting wall-clock decide *when an arm stops* is not, and the loop experiment
walked straight into it.

`Budget` bounds iterations, wall-clock and disk, and halts on whichever binds
first. That is exactly right as a safety property — an unattended loop must stop
even if an iteration hangs. But in the three-seed loop-vs-control run, the
1800s wall budget was the binding constraint for seed 0 and the iteration count
was binding for seed 1. The two seeds therefore ran different procedures:

| seed | stopped because | accumulated steps |
|---|---|---|
| 0 | wall-clock budget exhausted (1800s) | 2150 |
| 1 | completed all iterations | 2350 |

Machine load is now an uncontrolled variable in the treatment arm. Run the same
seed on an idle machine and it completes; run it next to two other training
processes and it gets cut short. That is not reproducible, and pooling seeds
across it means averaging over "how busy the laptop was", which no amount of
seeding fixes.

The rule: **for an experiment, set `max_wall_seconds` high enough that it never
binds, and let `max_iterations` be the thing that stops the run.** Wall-clock
stays in the budget as the backstop it was designed to be. Then assert after the
fact that every arm stopped for the intended reason — a run that halted on
wall-clock is not a datapoint, it is a truncated one, and the halt reason is
recorded in the lineage precisely so this is checkable rather than invisible.

(Note the interaction with the concurrency trap in TODO.md: running experiment
arms in parallel on a 4-core machine both slows them down *and*, through this
mechanism, silently changes what they measure.)

#### A seeded run that cannot be regenerated is not an audit trail

The loop was not reproducible at a fixed seed for its entire existence, and
nothing caught it, including the thirty tests written about it.

`scripts/run_loop.py` constructed its Transformer and only then called `train()`
-- which seeds *inside itself*, long after the weights were drawn. The incumbent
was therefore initialised from an unseeded generator, and since every candidate
is a clone of the incumbent, one missing line made every measurement in the
lineage unreproducible. `scripts/train.py` seeds before it builds anything,
which is why the controls reproduced and the loop did not, and why the
discrepancy took so long to notice: half the experiment was behaving.

Re-running seed 0 gave **identical proposals and different measurements**:

| iteration | first run | second run | decision |
|---|---|---|---|
| 0 | 2.159980 | 2.153856 | promoted both times |
| 12 | 1.868022 | 1.869494 | **promoted, then rejected** |

The divergence is above the 0.0040 noise floor from the first iteration, and by
iteration 12 it is enough to flip a promotion. Same seed, same proposals,
different lineage.

Three lessons, in order of how much they cost:

1. **Seed before you construct, not before you train.** Module `__init__` draws
   from the global generator. Seeding inside the training function is too late
   and looks correct at every call site.
2. **Assert on the measurements, not the plan.** The obvious reproducibility
   check -- do two runs propose the same things? -- passed throughout, because
   the search space has its own generator. Only the measured BPB and the
   promotion decisions diverged, so those are what a test must compare.
3. **A component handed its state cannot verify its own provenance.** The loop
   receives a built incumbent and has no way to know whether the caller seeded.
   It now records a sha256 of the starting weights in `models/model.json`, which
   makes "these two runs should have been identical" a question the artifacts
   answer, rather than one you answer three experiments later by noticing drift.

This is the failure the constitution's auditability reading is about. A lineage
that cannot be regenerated cannot be checked by anyone, including the system
that produced it.

### 4. Pre-register the prediction

Before each experiment, write down in `docs/lab-notebook.md`: the question, what
you predict, and what result would change your mind.

This is a discipline, not bureaucracy. It is the mechanism that prevents
retrofitting a story onto whatever came out — the most common failure mode in
solo research, precisely because no reviewer is there to catch it.

### 5. Negative results are results

"Depth didn't matter at this scale" is a finding, and often a more useful one
than the positive version. Record it with the same care. The temptation to bury
it and re-run until something looks interesting is how p-hacking happens to
honest people.

### 6. Reproducibility is a build target

Any run must be reproducible from `runs/<id>/` alone: resolved config, git SHA,
data hash, seeds. Test this for real — pick an old run, re-run it from its
manifest, confirm the curve matches. Do this early, while there are few runs, not
after you have 200.

### Reproducibility is not bitwise everywhere

The determinism test is written against a tolerance, not equality, and the
reason is worth recording because it qualifies a claim made loosely elsewhere in
these docs.

**What is measured.** On an Intel Mac, two identically seeded 20-step AdamW runs
of the same code in the same process produced `1.1013418974562068` and
`1.1013418974562064` — a relative difference of **3.6e-16**, one to two units in
the last place of a float64. The same test is exactly equal here on Linux
(NumPy 2.4 against OpenBLAS 0.3.31, Haswell kernels).

**What a real determinism bug looks like, for contrast.** Changing the seed on
that same test moves the loss by **9.7e-02 relative** — fourteen orders of
magnitude larger. There is an enormous empty gap between the two, and the test
now asserts `rel=1e-9` inside it: seven orders above the floating-point noise,
seven below the smallest failure worth catching. A companion test asserts that
this relaxed comparison *still fails* on a seed change, because loosening an
assertion is only defensible if you demonstrate it kept its teeth.

**What is not established: the mechanism.** The obvious story is that NumPy
delegates `matmul` to a BLAS whose reduction order varies between calls. I have
not verified that, and the first guess — Apple's Accelerate framework — is
probably wrong: Intel macOS is pinned to `numpy<2` here (see the torch pin in
`pyproject.toml`), and those wheels ship OpenBLAS, not Accelerate. The matrices
in this test are also tiny (16x4 @ 4x8), well below any threading threshold, so
a multithreaded-reduction explanation is weak too. Treat the cause as an
unidentified floating-point-associativity difference in the platform's linear
algebra until someone runs `python -c "import numpy; numpy.show_config()"` on
the affected machine and settles it. Naming a plausible mechanism is not the
same as having one.

**The practical consequence for the research.** Do not compare BPB across
machines beyond about 9 significant figures, and do not read a last-digit
difference between two runs of the same config as evidence of a bug. The
seed-to-seed noise floor in [results.md](results.md) is ~0.004 BPB, roughly
thirteen orders of magnitude above this, so every claim in the results stands
unchanged — but a future test that asserts exact equality on a float derived
from a matmul will be flaky off this machine, and should not be written.

---

## Part 4 — The eval harness

```
eval/
  metrics.py       # bpb, perplexity, token accuracy
  probes/          # copying, sorting, modular arithmetic, dyck
  generate.py      # fixed-prompt, fixed-seed sample generation
  judge.py         # optional LLM-as-judge scoring
  report.py        # runs/index.db → comparison tables + plots
```

Requirements:

- **Fast.** Runs at every checkpoint. If eval is slow you will stop running it,
  and then you're flying blind.
- **Deterministic.** Fixed seeds, fixed prompts, fixed order. Eval variance
  masquerading as model variance is a miserable bug to chase.
- **Versioned.** Changing the eval invalidates comparison with earlier runs.
  Stamp an eval-suite version into results and bump it on any change.
- **Reports from the database, not from memory.** `report.py` reads
  `runs/index.db` and emits the table or plot. Never assemble results by hand —
  that is where transcription errors enter.

---

## Part 5 — Honest reporting

Every write-up states, without being asked:

- Parameter count, **split into embedding and non-embedding**
- Training tokens, and tokens-per-parameter ratio
- Total compute (FLOPs) and wall-clock
- Exact data version and hash
- Number of seeds and the observed spread
- What you tried that **didn't** work

And keep the framing straight: the comparison class is *other small models
trained on similar data*, never a frontier model. A 20M parameter model is
roughly 0.001% the size of a frontier system. Comparing against ChatGPT is not
modesty or humility — it is a category error that obscures whether your actual
result is good.
