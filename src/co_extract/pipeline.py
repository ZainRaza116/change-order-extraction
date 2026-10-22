"""End-to-end orchestration:

    file -> page text -> LLM fields -> typed draft -> validation rules -> confidence

run_pipeline() never raises for a bad document; it returns a result with status "error"
so a batch keeps going.
"""

import datetime as dt
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from co_extract.assemble import Assembled, assemble
from co_extract.confidence import DocumentScore, score_document
from co_extract.config import Settings, load_settings
from co_extract.ingest import PageText, extract_text
from co_extract.llm import (
    AnthropicProvider,
    CachedProvider,
    LLMProvider,
    LlmResult,
    OpenAIProvider,
    extract_fields,
)
from co_extract.result import ExtractionResult, RunInfo, build_result, error_result
from co_extract.validate import RuleResult, validate

# USD per million tokens (input, output), for the cost estimate in each result.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # OpenAI list prices as last known; check platform.openai.com/pricing
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def build_provider(settings: Settings, replay: bool = False) -> CachedProvider:
    """The configured provider wrapped in the record/replay cache."""
    mode = "replay" if replay else settings.llm_mode
    if mode == "replay":
        inner = None
    elif settings.provider == "openai":
        inner = OpenAIProvider()
    else:
        inner = AnthropicProvider()
    return CachedProvider(inner, settings.cache_dir, mode)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 5)


@dataclass
class Attempt:
    """One model's pass over a document, from LLM call to confidence scores."""

    model: str
    llm: LlmResult
    assembled: Assembled
    rules: list[RuleResult]
    score: DocumentScore

    @property
    def rank(self) -> tuple[int, float]:
        """Higher is better: fewer failed rules, then higher overall confidence."""
        return (-sum(r.failed for r in self.rules), self.score.overall)

    @property
    def cost(self) -> float | None:
        return estimate_cost(self.model, self.llm.input_tokens, self.llm.output_tokens)


def _attempt(
    pages: list[PageText],
    provider: LLMProvider,
    model: str,
    settings: Settings,
    today: dt.date | None,
) -> Attempt:
    llm = extract_fields(
        pages, provider, model, temperature=settings.temperature, effort=settings.effort
    )
    assembled = assemble(llm.extraction)
    rules = validate(assembled.draft, today=today)
    score = score_document(llm.extraction, assembled, pages, rules, settings.review_threshold)
    return Attempt(model, llm, assembled, rules, score)


def run_pipeline(
    path: Path,
    provider: LLMProvider,
    settings: Settings | None = None,
    *,
    escalate: bool = False,
    today: dt.date | None = None,
) -> ExtractionResult:
    settings = settings or load_settings()
    path = Path(path)
    start = time.monotonic()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        pages = extract_text(path, vision_model=settings.ocr_vision_model)
        attempts = [_attempt(pages, provider, settings.model, settings, today)]
        if (
            escalate
            and attempts[0].score.needs_review
            and settings.escalation_model != settings.model
        ):
            # Cascade: re-run a doubtful document on the stronger model; keep the better one.
            attempts.append(_attempt(pages, provider, settings.escalation_model, settings, today))
    # RuntimeError covers LLM errors and unreadable files (pymupdf.FileDataError).
    except (RuntimeError, ValueError, OSError) as exc:
        return error_result(path, sha256, exc, time.monotonic() - start)

    best = max(attempts, key=lambda a: a.rank)
    costs = [a.cost for a in attempts]
    run = RunInfo(
        model=best.model,
        attempts=best.llm.attempts,
        cached=all(r.cached for r in best.llm.responses),
        escalated=len(attempts) > 1,
        input_tokens=sum(a.llm.input_tokens for a in attempts),
        output_tokens=sum(a.llm.output_tokens for a in attempts),
        cost_usd=None if None in costs else round(sum(costs), 5),  # every call is paid for
        seconds=round(time.monotonic() - start, 2),
        llm_seconds=round(sum(r.latency_s for a in attempts for r in a.llm.responses), 2),
    )
    return build_result(
        path, sha256, pages, best.llm.extraction, best.assembled, best.rules, best.score, run
    )
