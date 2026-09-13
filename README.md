# gitAI

Small-scale language model research. Building a decoder-only transformer from
scratch — tokenizer, architecture, training loop, evaluation harness — and using
it to answer a research question that is actually tractable without a GPU.

## What this project is

| | |
|---|---|
| **Goal** | Train a language model from scratch (own tokenizer, own architecture, own training loop) |
| **Purpose** | Research and experimentation — reproducible results, not a product |
| **Compute** | Laptop, CPU/MPS only, with occasional rented GPU for confirmation runs |
| **Scale** | 1M – 50M parameters, ~10M – 1B training tokens |
| **Domain** | Open — corpus chosen for tractability (see [docs/data-and-storage.md](docs/data-and-storage.md)) |

## What this project is not

- Not a GPT-4 competitor, or a competitor to anything you can download. A 20M
  parameter model is roughly 0.001% the size of a frontier model. Judge it
  against *small models trained on the same data*, never against ChatGPT.
- Not a fine-tune of open weights. We write the model.
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
5. **[docs/adr/](docs/adr/)** — Architecture Decision Records. Once a decision in
   `decisions.md` is settled, it becomes an immutable ADR.

## Status

**Planning.** No code yet. Nothing in `docs/decisions.md` marked *Proposed* has
been committed to. See [TODO.md](TODO.md) Phase 0 for the first work.
