.PHONY: help setup test test-fast lint fmt typecheck bench demo data data-sweep halt clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create the venv and install everything
	uv venv --python 3.11
	uv pip install -e ".[dev]"
	.venv/bin/pre-commit install

test:  ## run the full suite
	.venv/bin/pytest -q

test-fast:  ## skip anything marked slow
	.venv/bin/pytest -q -m "not slow"

lint:  ## check style
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## fix style in place
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

typecheck:  ## static types
	.venv/bin/pyright

data:  ## raw corpus -> curated parquet -> tokenizer -> shards
	.venv/bin/python scripts/prepare_data.py --corpus tinyshakespeare --vocab-size 2048

data-sweep:  ## compare vocabulary sizes (the Decision 4 measurement)
	.venv/bin/python scripts/prepare_data.py --corpus tinyshakespeare --sweep

bench:  ## measure this machine (needs the train extra)
	.venv/bin/python scripts/benchmark.py --json docs/hardware-baseline.json

demo:  ## train XOR with the hand-written engine
	.venv/bin/python scripts/demo_autograd.py

halt:  ## stop a running improvement loop
	.venv/bin/python scripts/halt.py stop

clean:
	rm -rf .pytest_cache .ruff_cache .venv **/__pycache__ src/*.egg-info
