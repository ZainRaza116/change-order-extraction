"""co-extract: structured, validated JSON with per-field confidence from change orders.

Flow (pipeline.run_pipeline):

    ingest/      file -> per-page text (PDF text layer, tables, OCR for scans)
    llm/         one schema-constrained call -> raw values + verbatim evidence
    assemble     raw strings -> typed ChangeOrderDraft (normalize.py parsers)
    validate     11 domain rules (contract math, totals, signs, dates, status...)
    confidence   per-field scores from evidence, validity, OCR quality, ambiguity
    result       ExtractionResult: the public output contract

Interfaces: cli.py (Typer) and api.py (FastAPI). evaluate.py scores outputs against
hand-labelled ground truth.
"""

__version__ = "0.1.0"
