# co-extract

Extracts structured fields from messy construction change orders (digital PDFs, scans, or plain text) into schema-validated JSON. Every field carries a confidence score and the verbatim source text it came from, and documents that need a human look are flagged `needs_review`.

> Work in progress. Results, examples and the writeup link will be filled in once the pipeline and evaluation are done.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and for scanned PDFs, Tesseract and Poppler (`brew install tesseract poppler` on macOS).

```bash
uv sync
cp .env.example .env        # add your ANTHROPIC_API_KEY
uv run co-extract run samples/ --out outputs/
```

## Development

```bash
uv run pytest
uv run pre-commit install   # ruff lint + format on commit
```

## Layout

```
src/co_extract/   pipeline stages: ingest → text_extraction → normalize → llm → validation → scoring
samples/          input change orders (synthetic)
expected/         hand-labelled ground truth, one JSON per sample
outputs/          pipeline results, committed so they can be read without running anything
schema/           exported JSON schema
scripts/          sample generation and evaluation
tests/            unit tests for deterministic code
```
