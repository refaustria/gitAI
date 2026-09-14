# Hardware baseline

Measured on the machine this project actually runs on, not estimated. Everything
in the roadmap that says "achievable" or "too slow" traces back to this table.

## The machine

```
platform      macOS-26.6.2-x86_64-i386-64bit
processor     i386
python        3.11.16
torch         2.2.2
device        cpu
threads       4
```

An Intel Mac, CPU only, four threads. Two consequences worth stating up front,
both already load-bearing elsewhere in these docs:

- torch is pinned to **2.2.2**, the last release with macOS x86_64 wheels, which
  in turn pins **numpy<2**. See the platform markers in `pyproject.toml`.
- this is the machine where identically seeded NumPy runs diverge by one ULP;
  see [evaluation.md](evaluation.md#reproducibility-is-not-bitwise-everywhere).

## Measured throughput

`make bench`, 2026-09-14:

| config | params | non-emb | s/step | tok/s | h/100M tokens |
|---|---:|---:|---:|---:|---:|
| tiny | 1,317,632 | 793,344 | 0.970 | 4,225 | 6.6 |
| small | 5,787,648 | 4,739,072 | 1.716 | 2,387 | 11.6 |
| medium | 15,769,344 | 14,196,480 | 3.605 | 1,136 | 24.5 |
| large | 39,926,784 | 37,829,632 | 8.019 | 511 | 54.4 |

Throughput falls sub-linearly with size — `large` has 30× the parameters of
`tiny` but only 8× the cost per step — so the per-token price of a bigger model
is better than the parameter count suggests. It is the *token budget* that
grows, and that is what makes the large configs impossible here.

## What that costs in practice

At the Chinchilla-ish heuristic of ~20 tokens per parameter:

| config | 20× tokens | one run | **a 3-seed, 2-arm experiment** | 1 epoch of TinyStories |
|---|---:|---:|---:|---:|
| tiny | 26M | **1.7 h** | **0.4 days** | 31 h |
| small | 116M | **13.4 h** | 3.4 days | 55 h |
| medium | 315M | 77 h | 19 days | 115 h |
| large | 799M | 434 h | 109 days | 256 h |

The middle column is the one that matters. **Research is not one run** — a claim
needs seeds, and a comparison needs two arms, so the real unit of work is six
runs and not one. That column is what turns "too slow" from an opinion into a
number.

## Recommendation

**Two configs, for two different jobs.**

- **`tiny` (1.3M) is the experimental workhorse.** A full 3-seed, 2-arm
  comparison finishes overnight. Every result in
  [results.md](results.md) was produced at roughly this scale, and the whole
  R6–R10 series is only possible because a run is cheap enough to repeat — which
  this project has now had to do twice.
- **`small` (5.8M) is the showcase model.** One overnight run at 20× tokens.
  This is the smallest size where TinyStories-style corpora are reported to
  produce genuinely coherent short stories, so it is the config to spend a night
  on once the experimental question is settled at `tiny`.

**`medium` is a single deliberate commitment**, not something to iterate on: 77
hours is a long weekend for *one* datapoint, with no seeds and no control.
**`large` is out of reach** on this machine — 109 days for one comparison — and
is the honest boundary where Decision 4's "rent a GPU?" question becomes real
rather than theoretical.

The instinct to reach for the biggest config that fits is the wrong one here.
Six `tiny` runs answer a question; one `medium` run produces an anecdote, and
this project has already demonstrated twice what an unreplicated anecdote is
worth.
