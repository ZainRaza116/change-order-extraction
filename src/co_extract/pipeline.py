"""End-to-end orchestration:

    file -> page text -> LLM fields -> typed draft -> validation rules -> confidence

run_pipeline() never raises for a bad document; it returns a result with status "error"
so a batch keeps going.
"""

import datetime as dt
import hashlib
import time
from pathlib import Path

from co_extract.assemble import assemble
from co_extract.confidence import score_document
from co_extract.config import Settings, load_settings
from co_extract.ingest import extract_text
from co_extract.llm import CachedProvider, LLMProvider, OpenAIProvider, extract_fields
from co_extract.result import ExtractionResult, RunInfo, build_result, error_result
from co_extract.validate import validate

# USD per million tokens (input, output), for the cost estimate in each result.
# OpenAI list prices as last known; check platform.openai.com/pricing.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def build_provider(settings: Settings, replay: bool = False) -> CachedProvider:
    """The OpenAI provider wrapped in the record/replay cache."""
    mode = "replay" if replay else settings.llm_mode
    inner = None if mode == "replay" else OpenAIProvider()
    return CachedProvider(inner, settings.cache_dir, mode)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 5)


def run_pipeline(
    path: Path,
    provider: LLMProvider,
    settings: Settings | None = None,
    *,
    today: dt.date | None = None,
) -> ExtractionResult:
    settings = settings or load_settings()
    path = Path(path)
    start = time.monotonic()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        pages = extract_text(path)
        llm = extract_fields(
            pages,
            provider,
            settings.model,
            temperature=settings.temperature,
            effort=settings.effort,
        )
    # RuntimeError covers LLM errors and unreadable files (pymupdf.FileDataError).
    except (RuntimeError, ValueError, OSError) as exc:
        return error_result(path, sha256, exc, time.monotonic() - start)

    assembled = assemble(llm.extraction, pages)
    rules = validate(assembled.draft, today=today)
    score = score_document(llm.extraction, assembled, pages, rules, settings.review_threshold)
    run = RunInfo(
        model=settings.model,
        attempts=llm.attempts,
        cached=all(r.cached for r in llm.responses),
        input_tokens=llm.input_tokens,
        output_tokens=llm.output_tokens,
        cost_usd=estimate_cost(settings.model, llm.input_tokens, llm.output_tokens),
        seconds=round(time.monotonic() - start, 2),
        llm_seconds=round(sum(r.latency_s for r in llm.responses), 2),
    )
    return build_result(path, sha256, pages, llm.extraction, assembled, rules, score, run)
