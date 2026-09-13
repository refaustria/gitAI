# gitAI

Small-scale language model research. Building a decoder-only transformer from
scratch — tokenizer, architecture, training loop, evaluation harness — then
wrapping it in a bounded, unattended self-improvement loop to find out when such
a loop helps and when it collapses.

## What this project is

| | |
|---|---|
| **Goal** | Train a language model from scratch (own tokenizer, own architecture, own training loop) |
| **Purpose** | Research and experimentation — reproducible results, not a product |
| **Compute** | Laptop, CPU/MPS only, with occasional rented GPU for confirmation runs |
| **Scale** | 1M – 50M parameters, ~10M – 1B training tokens |
| **Domain** | Open — corpus chosen for tractability (see [docs/data-and-storage.md](docs/data-and-storage.md)) |
| **Question** | When does a bounded self-improvement loop improve a small LM, and when does it collapse? |

## What this project is not

- Not a GPT-4 competitor, or a competitor to anything you can download. A 20M
  parameter model is roughly 0.001% the size of a frontier model. Judge it
  against *small models trained on the same data*, never against ChatGPT.
- Not a fine-tune of open weights. We write the model.
- Not recursively self-modifying. The **loop** improves; the model is what gets
  improved. The loop may change weights, data and configuration, and may never
  change its own source code — see [docs/constitution.md](docs/constitution.md).
- Not a product. No users, no SLA, no serving infrastructure until and unless
  the research produces something worth serving.

## The enabling result

Training a useful LLM from scratch on a laptop sounds impossible, and it is if
you use a general web corpus. The escape hatch is corpus design:

> **TinyStories** (Eldan & Li, 2023) is a synthetic corpus of short stories using
> only vocabulary a 3–4 year old understands. Models of **1M–35M parameters**
> trained on it produce grammatical, coherent, consistent English — because the
> data distribution is narrow enough that a tiny model can actually cover it.

This is the single most important fact shaping the project. It converts
"impossible without a datacenter" into "a few hours per run on your laptop", and
it means every architectural and data decision here is made against a
small-model, narrow-corpus target rather than a general-purpose one.

## Quick start

```bash
make setup     # uv venv + dev install + pre-commit hooks
make test      # 96 tests, CPU-only, a few seconds
make demo      # train XOR with the hand-written engine — no PyTorch involved
make bench     # measure YOUR laptop (needs: uv pip install -e ".[train]")
make halt      # stop a running improvement loop
```

## Documents

Read in this order:

1. **[TODO.md](TODO.md)** — the phased roadmap. The working checklist.
2. **[docs/decisions.md](docs/decisions.md)** — the fundamental decisions:
   technology, infrastructure, tooling, storage. Each with options, trade-offs,
   a recommendation, and a status.
3. **[docs/data-and-storage.md](docs/data-and-storage.md)** — corpus strategy and
   the four distinct storage problems (this is where the "database" question is
   answered).
4. **[docs/evaluation.md](docs/evaluation.md)** — how we measure, and the research
   methodology that makes results mean something.
5. **[docs/self-improvement.md](docs/self-improvement.md)** — what self-improvement
   can and cannot mean at this scale, the loop design, and model collapse as the
   research question.
6. **[docs/constitution.md](docs/constitution.md)** — the three axioms, as
   invariants enforced on the loop rather than values taught to the model.
7. **[docs/adr/](docs/adr/)** — Architecture Decision Records.

## Layout

```
src/gitai/autograd/    hand-written reverse-mode autograd on NumPy (Phase 0)
src/gitai/safety/      halt switch, budget, path guard, lineage, invariant gate
scripts/benchmark.py   measure your hardware — run this first
scripts/halt.py        operator stop button
tests/                 96 tests: gradient checks, and adversarial safety tests
```

## Status

**Phase 0 complete.** The autograd engine works (`make demo` trains XOR with no
framework underneath it), 96 tests pass, CI is green on 3.11 and 3.12.

The constraint layer for the Phase 8 loop is also built — deliberately early.
Brakes before engine: a stop button retrofitted to a running loop is a stop
button nobody tested.

**Next:** run `make bench` on your own laptop to fill in
`docs/hardware-baseline.md`, then Phase 1 (data pipeline).
