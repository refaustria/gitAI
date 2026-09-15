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

`make bench`, 2026-09-15, timing the **real v6_modern architecture** — the one
you actually train:

| config | params | non-emb | s/step | tok/s | h/100M tokens |
|---|---:|---:|---:|---:|---:|
| tiny | 1,311,360 | 787,072 | 0.439 | 9,340 | **3.0** |
| small | 5,767,424 | 4,718,848 | 1.169 | 3,503 | **7.9** |
| medium | 15,735,168 | 14,162,304 | 2.885 | 1,420 | **19.6** |
| large | 39,852,544 | 37,755,392 | 6.012 | 681 | **40.8** |

Throughput falls sub-linearly with size — `large` has 30× the parameters of
`tiny` but only 14× the cost per step — so the per-token price of a bigger model
is better than the parameter count suggests. It is the *token budget* that
grows, and that is what puts the large configs out of reach.

### The earlier numbers, and a correction that did not transfer

This table previously came from a benchmark that timed a *proxy* architecture —
a LayerNorm/GELU stack on `nn.MultiheadAttention`, written in Phase 1 before the
real model existed. Its own docstring said it was temporary ("Not the project's
model — that gets built properly in Phase 2") and it was never swapped, so every
scoping number in this project described a transformer nobody trains.

| config | proxy h/100M | real h/100M | ratio here | ratio on a Linux container |
|---|---:|---:|---:|---:|
| tiny | 6.6 | 3.0 | 0.45 | 0.20 |
| small | 11.6 | 7.9 | **0.68** | 0.34 |
| medium | 24.5 | 19.6 | 0.80 | 0.46 |
| large | 54.4 | 40.8 | 0.75 | 0.51 |

The last column is the part worth recording. Having measured the architecture
correction on a Linux container with torch 2.14, I predicted the real numbers
here would be 2–3× lower and called `small` "an evening, not an overnight".
**The ratio did not transfer.** On Intel macOS with torch 2.2.2 the real model
is only 1.3–1.5× faster than the proxy. An architecture-relative ratio still
rides on the BLAS, the torch version and the CPU; it is not a portable constant,
and `small` is 9.1 hours — a night, as originally stated.

Direction right, size wrong, in the optimistic direction: the fifth consecutive
instance of that pattern in this project (see [findings.md](findings.md)). The
correct move was the one stated and then talked past — re-run the benchmark on
the target machine and quote nothing until it lands.

## What that costs in practice

At the Chinchilla-ish heuristic of ~20 tokens per parameter:

| config | 20× tokens | one run | **a 3-seed, 2-arm experiment** |
|---|---:|---:|---:|
| tiny | 26M | **0.8 h** | **0.2 days** |
| small | 115M | **9.1 h** | 2.3 days |
| medium | 315M | 62 h | 15 days |
| large | 797M | 325 h | 81 days |

The last column is the one that matters. **Research is not one run** — a claim
needs seeds, and a comparison needs two arms, so the real unit of work is six
runs and not one. That column is what turns "too slow" from an opinion into a
number.

## Recommendation

**Two configs, for two different jobs.**

- **`tiny` (1.3M) is the experimental workhorse.** A full 3-seed, 2-arm
  comparison finishes in about five hours — a working day, not a night. Every result in
  [results.md](results.md) was produced at roughly this scale, and the whole
  R6–R10 series is only possible because a run is cheap enough to repeat — which
  this project has now had to do twice.
- **`small` (5.8M) is the showcase model.** One overnight run at 20× tokens —
  **9.1 hours**, genuinely a night.
  This is the smallest size where TinyStories-style corpora are reported to
  produce genuinely coherent short stories, so it is the config to spend a night
  on once the experimental question is settled at `tiny`.

**`medium` is a single deliberate commitment**, not something to iterate on: 62
hours is a long weekend for *one* datapoint, with no seeds and no control.
**`large` is out of reach** on this machine — 81 days for one comparison — and
is the honest boundary where Decision 4's "rent a GPU?" question becomes real
rather than theoretical.

The instinct to reach for the biggest config that fits is the wrong one here.
Six `tiny` runs answer a question; one `medium` run produces an anecdote, and
this project has already demonstrated twice what an unreplicated anecdote is
worth.
