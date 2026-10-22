# co-extract

Extracts structured fields from messy construction change orders (digital PDFs, scans, or plain text) into schema-validated JSON. Every field carries a confidence score and the verbatim source text it came from, and documents that need a human look are flagged `needs_review`.

> Work in progress. Results, examples and the writeup link will be filled in once the pipeline and evaluation are done.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and for scanned PDFs, Tesseract and Poppler (`brew install tesseract poppler` on macOS).

```bash
uv sync
cp .env.example .env        # add your OPENAI_API_KEY (or set CO_EXTRACT_PROVIDER=anthropic)
uv run co-extract run samples/ --out outputs/
```

## Development

```bash
make install   # dependencies + git hooks
make test      # 205 tests, no network
make lint      # ruff
make replay    # run the pipeline from recorded LLM responses (no API key)
make eval      # score outputs/ against expected/
make help      # everything else
```

## Layout

```
src/co_extract/
  ingest/          documents.py (PDF/text/image -> pages), ocr.py (OpenCV + Tesseract)
  llm/             output.py (schema), prompt.py, providers.py (OpenAI, Anthropic),
                   cache.py (record/replay), extract.py (call + retry)
  normalize.py     money / date / day / percent / currency parsers
  assemble.py      raw LLM strings -> typed draft
  validate.py      domain rules
  confidence.py    per-field confidence
  result.py        ExtractionResult (output contract)
  pipeline.py      orchestration
  evaluate.py      accuracy, hallucinations, calibration
  cli.py, api.py   interfaces
tests/unit/        pure logic, no files or network
tests/integration/ samples, OCR, pipeline, CLI, API
samples/           input change orders        expected/  hand-labelled ground truth
outputs/           pipeline results           eval/      evaluation report
cassettes/         recorded LLM responses     schema/    output JSON Schema
scripts/           sample generation
```
