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
