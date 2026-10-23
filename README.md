# co-extract

Extracts structured fields from messy construction change orders (digital or scanned PDFs,
and plain text such as emails) into schema-validated JSON. Every field carries a confidence
score, the verbatim quote it came from and its page; documents that need a human look are
flagged `needs_review` with the reasons.

**[WRITEUP.md](WRITEUP.md)** covers the approach, results and failure modes.

## Results

8 independently labelled documents (`testset/`, 9 inputs), one live run with `gpt-4o-mini`
([report](eval/testset/report.md)):

| Metric | Result |
|---|---|
| Accuracy, 16 structured fields | 99% (required fields 100%) |
| Invented values | 0 of 27 |
| Line items / markups recovered | 26 of 26 / 7 of 7 |
| Documents with an error flagged for review | 2 of 2 |
| Cost / LLM time per document | $0.0010 / 11 s |

Values scored above 0.8 confidence are right 97% of the time (150 values); lower buckets
hold too few values to judge. Free-text fields are paraphrased in the answers and reported
as similarity scores. The document set includes a prompt-injection line, which the pipeline
ignored. Details and failure modes in [WRITEUP.md](WRITEUP.md).

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Tesseract for scans
(`brew install tesseract` or `apt-get install tesseract-ocr`).

**Without an API key**: the LLM responses for the samples are recorded in `cassettes/`, so
the whole pipeline replays offline:

```bash
uv sync
make replay    # both sample sets -> outputs/ and testset/outputs/
make eval      # eval/report.md (dev set) and eval/testset/report.md
```

Scanned samples replay only with the same Tesseract version used to record them (5.5);
the text and digital samples replay anywhere.

**With an API key**, to process your own documents:

```bash
cp .env.example .env        # add OPENAI_API_KEY
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
  "evidence": "TOTAL THIS CHANGE ORDER OK per mtg @ $10,763.06 >",
  "page": 1,
  "alternatives": [],
  "flags": []
}
```

Plus line items, markups and signatures (each with its own confidence), every validation
rule's outcome, `needs_review` with reasons, and run details (model, tokens, cost). Values
filled by code rather than read from the page say so. On the faxed scan, a stamp hides line
item 1; it comes back recovered from the subtotal, and the document is flagged:

```json
{"value": {"quantity": "3", "unit": "EA", "unit_cost": "2850.00", "amount": "8550.00"},
 "confidence": 0.6,
 "flags": ["derived: amount derived: total less the other items (value unreadable in the document)"]}
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
| `OPENAI_API_KEY` | | needed only for documents without a recording |
| `CO_EXTRACT_MODEL` | `gpt-4o-mini` | any OpenAI model with structured outputs |
| `CO_EXTRACT_LLM_MODE` | `auto` | `auto` (use recordings, else call), `replay`, `refresh` |
| `CO_EXTRACT_REVIEW_THRESHOLD` | `0.7` | required fields below this flag the document |

## Development

```bash
make install   # dependencies + git hooks
make test      # 251 tests, no network
make lint      # ruff
make replay    # run the pipeline from recorded LLM responses (no API key)
make eval      # score both sets against their answer files
make help      # everything else
```

## Layout

```
src/co_extract/
  ingest/          documents.py (PDF/text -> pages), ocr.py (OpenCV + Tesseract)
  llm/             output.py (schema), prompt.py, providers.py (OpenAI),
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
samples/           dev documents              expected/  my labels for them
testset/           8 externally labelled documents (samples/, expected/, outputs/)
outputs/           dev results                eval/      evaluation reports
cassettes/         recorded LLM responses     schema/    output JSON Schema
scripts/           sample generation, test-set import
```

## Scope

Built as a timeboxed take-home, scoped to the task: PDFs (digital or scanned) and plain text
in, validated JSON with confidence out. Deliberately left out: image files, a review UI,
multi-CO PDFs, non-USD handling, and a larger labelled set.
