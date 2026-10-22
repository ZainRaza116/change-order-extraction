.DEFAULT_GOAL := help
.PHONY: help install lint format test run replay eval samples schema serve clean

help:  ## Show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies (incl. API extra) and git hooks
	uv sync --extra api
	uv run pre-commit install

lint:  ## Check style and formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix style and formatting
	uv run ruff check . --fix
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest -q

run:  ## Extract every sample into outputs/ (calls the LLM for uncached documents)
	uv run co-extract run samples/ --out outputs/

replay:  ## Same as run, from recorded responses only (no API key needed)
	uv run co-extract run samples/ --out outputs/ --replay

eval:  ## Score outputs/ against expected/ and write eval/report.md
	uv run co-extract eval

samples:  ## Regenerate the synthetic sample PDFs
	uv run python scripts/make_samples.py

schema:  ## Re-export the output JSON Schema
	uv run co-extract schema

serve:  ## Start the HTTP API on localhost:8000
	uv run co-extract serve

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
