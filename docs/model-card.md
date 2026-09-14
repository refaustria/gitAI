# Model card — gitAI TinyShakespeare

## What this is

A 1,049,216-parameter decoder-only transformer trained from scratch on
TinyShakespeare, produced as a **research artefact**. It exists to make
experiments about small language models possible, not to be used for anything.

**It is roughly 0.001% the size of a frontier model.** The correct comparison
class is other small models trained on similar data. Comparing it to a
general-purpose assistant is a category error, not modesty.

## Model details

| | |
|---|---|
| Architecture | Decoder-only transformer, pre-norm |
| Parameters | 1,049,216 (787,072 non-embedding, 262,144 embedding) |
| Layers / width / heads | 4 × 128 × 4 (32 per head) |
| Context length | 128 tokens |
| Normalisation | RMSNorm |
| Positions | RoPE |
| Feed-forward | SwiGLU at 8/3 width |
| Biases | none |
| Embeddings | input and output tied |
| Vocabulary | 2,048 (byte-level BPE, trained from scratch) |
| Precision | float32 |
| Checkpoint format | safetensors |

Everything — tokenizer, architecture, training loop, evaluation — was written
for this project rather than imported. The autograd engine in
`src/gitai/autograd/` is a NumPy implementation used for understanding
backpropagation; the model itself is PyTorch.

## Training data

**TinyShakespeare** — approximately 1.1 MB of Shakespeare, public domain, with
the compilation distributed under MIT as part of `karpathy/char-rnn`.

After curation: 7,018 documents, 372,993 training tokens. Split by content hash
into train / val / test (6,869 / 84 / 65 documents), which makes it structurally
impossible for a duplicate document to straddle the boundary. Measured leakage
between every split pair: **zero**.

Provenance, checksum and licence are recorded in `data/raw/MANIFEST.json`. The
corpus itself is not redistributed in the repository.

## Evaluation

| metric | value |
|---|---|
| **Held-out bits-per-byte** | **1.8732** |
| Validation loss | 3.6710 |
| Bigram baseline (context-free floor) | 3.2406 |
| Seed noise floor (1 std, 5 seeds) | 0.0040 |

Bits-per-byte is the headline metric rather than perplexity because perplexity is
per-token and therefore not comparable across tokenizers — and vocabulary size is
an experimental variable in this project.

**Capability probes:** the induction score is consistently **negative** (−0.52 to
−0.87 bits), meaning the model shows no in-context copying ability whatsoever.
It has not formed an induction circuit.

## What it can and cannot do

**Can:** reproduce the surface form of its corpus almost perfectly — speaker
names, colons, line breaks, document separators — and produce locally
grammatical English blank verse with plausible character names.

**Cannot:** anything else. It does not follow instructions, answer questions,
reason, retain information across a context, or generalise beyond its corpus. It
has no in-context learning ability, as measured.

Sample output at temperature 0.8:

```
First Citizen:
This is the house I do attend you to them?
First Senator:
If he become, thou cross,
To see what thou hast made it to wide?
```

## Intended use

Research on small language models: architecture ablations, data-composition
experiments, mechanistic interpretability, and self-training dynamics. It was
built as the instrument for [findings.md](findings.md).

## Out-of-scope use

Any deployed or user-facing application. It is not fit for one and would fail
immediately.

It should also not be used as evidence about the behaviour of large language
models. Several findings in this project were shown to depend on the sampling
regime and on having a weak parent generator; extrapolating them upward is
exactly the error this card exists to discourage.

## Limitations and biases

- **Trained on one author, one register, roughly 1 MB.** It reproduces the
  vocabulary and cadence of early-modern English drama, including whatever social
  attitudes that corpus carries. It has no exposure to anything else.
- **No safety training of any kind**, and none would be meaningful at this scale
  — the model has no capacity to represent the concepts such training targets.
- **128-token context.** It cannot attend past roughly one speech.
- **Deterministic given a seed**, which is a property of the harness rather than
  the model.

## Reproducing it

```bash
make quickstart     # setup, data, tests, train
```

Approximately two minutes of CPU training after data preparation. Each run writes
a manifest containing the git SHA, tokenizer fingerprint, data hash, hardware and
seeds.

## Licence

Code MIT. Training data public domain; see `data/LICENSES.md`.
