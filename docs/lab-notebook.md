# Lab Notebook

One entry per experiment, **written before the runs finish**. Question,
prediction, and what result would change your mind.

This is a discipline, not bureaucracy. It is the mechanism that stops you
retrofitting a story onto whatever came out — the most common failure mode in
solo research, precisely because no reviewer is there to catch it. Writing down
"I expect X" and then getting Y is how you learn something; deciding afterwards
that Y is what you expected teaches you nothing.

Entries are append-only. When a prediction turns out wrong, **leave it** and add
the outcome underneath. The wrong predictions are the valuable part.

---

## E1 — What is the seed noise floor?

**Date:** 2026-09-13 · **Status:** prediction recorded before results

### Question

Train one configuration five times, changing nothing but the seed. How much does
final held-out BPB vary?

This number is the significance threshold for every later comparison in the
project. Until it exists, no claim that one configuration beats another means
anything.

### Setup

`v6_modern`, 4L × 128d × 4h, 1,049,216 parameters, 1000 steps, batch 16 × seq
128, lr 3e-3, TinyShakespeare (372,993 train tokens, vocab 2,048). Seeds 0–4.
The seed controls weight initialisation *and* batch order, so this measures the
combined effect — which is the right thing, since that is what differs between
any two runs you would actually compare.

### Prediction (recorded before the sweep completed)

- Standard deviation in val BPB: **0.005 – 0.02**.
- Spread (max − min) roughly 2–3× the std, so **0.01 – 0.06**.
- Reasoning: 6.6 epochs over a small corpus means each run sees essentially the
  same data, so the variation should come mostly from initialisation and batch
  order rather than from which data was seen. The model is also small enough to
  be far from capacity-limited on this corpus.

### What would change my mind

- **std > 0.05** would mean training is unstable at this lr/size, and the lr
  schedule or warmup needs revisiting before any experiment is worth running.
- **std < 0.001** would be suspicious rather than reassuring — it would suggest
  the seed is not actually varying the run, and I would go check that batch
  order really is seed-dependent.

### Result

*(filled in below once the sweep completed — see [results.md](results.md) R5)*

### What it means for the project

Whatever the number, it becomes the `NoRegression` tolerance in
`safety/invariants.py` for the Phase 8 improvement loop. A promotion gate set
tighter than the noise floor fires on chance; set much looser, it lets real
degradation through.

---

## E2 — Does the transformer beat the context-free floor? *(answered)*

**Date:** 2026-09-13

### Question

A bigram model is a pure lookup table: token *t* predicts token *t+1* with no
context at all. Its loss is the best any context-free model can do on this
corpus. Does the transformer beat it, and by how much?

Without this the headline number is uninterpretable — "1.87 BPB" is an opinion
until something anchors it.

### Prediction

The transformer should beat it clearly. A bigram on a 2,048-token vocabulary has
4.2M parameters — *four times* the transformer's 1.05M — and still cannot
condition on more than one token, so this is a test of whether context is worth
anything at all. If the transformer failed to beat it decisively, something would
be badly wrong.

### Result

| model | parameters | val BPB |
|---|---:|---:|
| Bigram (context-free floor) | 4,194,304 | **3.2406** |
| `v6_modern` transformer | 1,049,216 | **1.8732** |

**42% lower BPB with a quarter of the parameters.** Context is worth a great
deal, as expected. The transformer is not merely memorising unigram/bigram
statistics.

Incidentally: the bigram was trained twice by accident and produced
`3.240565592697779` both times — bit-identical. An unplanned confirmation that
the determinism guarantee holds end to end.

### Follow-up

A stronger floor would be an n-gram model with backoff, which would tell us how
much of the transformer's advantage is just longer-context statistics versus
something structural. Cheap to build; worth doing before claiming the model has
learned anything interesting.

---

## Template

```markdown
## E<n> — <question in one line>

**Date:** · **Status:** prediction recorded / complete

### Question
What do you want to know, and why does it matter for the project?

### Setup
Config, data, seeds, what varies and what is held fixed. One independent
variable.

### Prediction
What you expect, and roughly how strongly.

### What would change my mind
Concrete results that would falsify the prediction or invalidate the setup.

### Result
Numbers, with seed spread. Compare against the noise floor before concluding.

### What it means
Including "nothing — the effect was below the noise floor", which is a result.
```
