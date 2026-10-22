# co-extract

Extracts structured fields from messy construction change orders (digital PDFs, scans, faxes,
email text) into schema-validated JSON. Every field carries a confidence score, the verbatim
quote it came from and its page; documents that need a human look are flagged `needs_review`
with the reasons.

**[WRITEUP.md](WRITEUP.md)** covers the approach, results and failure modes.

## Results

4 documents with hand-labelled ground truth, `gpt-4o-mini` ([full report](eval/report.md)):

| Metric | Result |
|---|---|
| Field accuracy | 93% |
| Hallucinated values | 0 of 5 |
| Line-item amounts recovered | 14 of 15 |
| Documents with an error flagged for review | 2 of 2 |
| Cost / latency per document | $0.0015 / 26 s |

Confidence is calibrated in the right direction: 96% accuracy above 0.8, 86% at 0.5–0.8,
50% below 0.5. Four documents are a smoke test, not a benchmark.

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Tesseract for scans
(`brew install tesseract` or `apt-get install tesseract-ocr`).

**Without an API key**: the LLM responses for the samples are recorded in `cassettes/`, so
the whole pipeline replays offline:

```bash
uv sync
uv run co-extract run samples/ --replay    # writes outputs/*.json
uv run co-extract eval                     # writes eval/report.md
```

Scanned samples replay only with the same Tesseract version used to record them (5.5);
the text and digital samples replay anywhere.

**With an API key**, to process your own documents:

```bash
cp .env.example .env        # add OPENAI_API_KEY (or set CO_EXTRACT_PROVIDER=anthropic)
uv run co-extract run path/to/change_orders/ --out results/
```

**HTTP API**: `uv sync --extra api && uv run co-extract serve`, then
`curl -F "file=@samples/co_002_digital.pdf" localhost:8000/extract`.

## Output

One JSON per document ([schema](schema/extraction_result.schema.json)). Each field:

```json
"change_amount": {
  "value": "10763.06",
  "raw": "$10,763.06",
  "confidence": 0.973,
  "evidence": "TOTAL THIS CHANGE ORDER OK per mtg @ $10,763.06",
  "page": 1,
  "alternatives": [],
  "flags": []
}
```

Plus line items, markups and signatures (each with its own confidence), every validation
rule's outcome, `needs_review` with reasons, and run details (model, tokens, cost). For
example, the real scan in `samples/` is flagged with:

```
rule line_items_total failed: line items sum to 1,090.00, but subtotal is 9,640.00
rule status_consistency failed: approved, but unsigned: owner
```

## How it works

```
file ─► ingest ─► LLM (strict JSON schema) ─► parsers ─► validation rules ─► confidence ─► JSON
        text layer   values copied verbatim    money,     contract math,      evidence match,
        tables       + evidence quote          dates,     totals, signs,      validity, OCR
        OCR (scans)  + page                    days       dates, status       quality, ambiguity
```

The model only copies; deterministic code parses and checks. Details in [WRITEUP.md](WRITEUP.md).

## Configuration

All settings are environment variables; see [.env.example](.env.example). The main ones:

| Variable | Default | |
|---|---|---|
| `CO_EXTRACT_PROVIDER` | `openai` | `openai` or `anthropic` |
| `CO_EXTRACT_MODEL` | `gpt-4o-mini` | `claude-sonnet-5-5` for Anthropic |
| `CO_EXTRACT_LLM_MODE` | `auto` | `auto` (use recordings, else call), `replay`, `refresh` |
| `CO_EXTRACT_ESCALATE` | `0` | re-run flagged documents on a stronger model |
| `CO_EXTRACT_VISION_FALLBACK` | `0` | re-read low-confidence scans with a vision model |

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

## Scope

Built as a timeboxed take-home. Deliberately left out: a review UI, multi-CO PDFs, non-USD
handling, and a larger labelled set. The Anthropic provider and the vision fallback are
implemented and unit-tested, but the reported numbers come from the OpenAI runs.
