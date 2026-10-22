"""End-to-end pipeline: file -> page text -> LLM fields -> typed draft -> rules -> confidence.

run_pipeline() never raises for a bad document; it returns a result with status "error"
so a batch keeps going.
"""

import datetime as dt
import hashlib
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from co_extract import __version__
from co_extract.assemble import Assembled, assemble
from co_extract.confidence import DocumentScore, FieldScore, score_document
from co_extract.config import Settings, load_settings
from co_extract.llm import (
    AnthropicProvider,
    CachedProvider,
    Extracted,
    LLMProvider,
    LlmResult,
    OpenAIProvider,
    extract_fields,
)
from co_extract.text_extract import PageText, extract_text
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


# --- output model ---------------------------------------------------------------------


class FieldResult(BaseModel):
    value: Any
    raw: str | None
    confidence: float
    evidence: str | None
    page: int | None
    alternatives: list[str] = []
    flags: list[str] = []


class ItemResult(BaseModel):
    value: dict[str, Any]
    confidence: float
    evidence: str | None
    page: int | None
    flags: list[str] = []


class RuleOutcome(BaseModel):
    rule: str
    outcome: str
    message: str


class PageInfo(BaseModel):
    page: int
    method: str
    char_count: int
    ocr_confidence: float | None = None


class RunInfo(BaseModel):
    model: str | None = None
    attempts: int = 0
    cached: bool = False
    escalated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0  # this run, wall clock (tiny when replayed from the cache)
    llm_seconds: float = 0.0  # original API latency, preserved in recordings
    pipeline_version: str = __version__


class ExtractionResult(BaseModel):
    source_file: str
    file_sha256: str
    status: Literal["ok", "error"]
    error: str | None = None
    pages: list[PageInfo] = []
    fields: dict[str, FieldResult] = {}
    line_items: list[ItemResult] = []
    markups: list[ItemResult] = []
    signatures: list[ItemResult] = []
    overall_confidence: float = 0.0
    needs_review: bool = True
    review_reasons: list[str] = []
    validation: list[RuleOutcome] = []
    model_notes: list[str] = []
    run: RunInfo = RunInfo()


# --- wiring -----------------------------------------------------------------------------


def build_provider(settings: Settings, replay: bool = False) -> CachedProvider:
    mode = "replay" if replay else settings.llm_mode
    if mode == "replay":
        inner = None
    elif settings.provider == "openai":
        inner = OpenAIProvider()
    else:
        inner = AnthropicProvider()
    return CachedProvider(inner, settings.cache_dir, mode)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 5)


def _field(item: Extracted, value: Any, score: FieldScore) -> FieldResult:
    raw = item.value.value if hasattr(item.value, "value") else item.value
    return FieldResult(
        value=value,
        raw=raw,
        confidence=score.confidence,
        evidence=item.evidence,
        page=item.page,
        alternatives=item.alternatives,
        flags=score.caps,
    )


def _items(raw_entries, typed_entries, kept: list[int], scores: list[FieldScore]):
    """One ItemResult per raw entry. Entries assemble() dropped (unreadable amount) keep
    their raw text with amount None, so every other entry still gets its typed value."""
    typed_by_raw = dict(zip(kept, typed_entries, strict=True))
    out = []
    for i, (raw, score) in enumerate(zip(raw_entries, scores, strict=True)):
        if i in typed_by_raw:
            value = typed_by_raw[i].model_dump(mode="json")
        else:
            value = raw.model_dump(mode="json", exclude={"evidence", "page"})
            value["raw_amount"], value["amount"] = value.get("amount"), None
        out.append(
            ItemResult(
                value=value,
                confidence=score.confidence,
                evidence=raw.evidence,
                page=raw.page,
                flags=score.caps,
            )
        )
    return out


def _compose(
    path: Path,
    sha: str,
    pages: list[PageText],
    llm: LlmResult,
    assembled: Assembled,
    rules: list[RuleResult],
    score: DocumentScore,
    run: RunInfo,
) -> ExtractionResult:
    x, draft = llm.extraction, assembled.draft.model_dump(mode="json")
    fields = {
        name: _field(item, draft.get(name), score.fields[name])
        for name, item in x
        if isinstance(item, Extracted)
    }
    return ExtractionResult(
        source_file=path.name,
        file_sha256=sha,
        status="ok",
        pages=[
            PageInfo(
                page=p.page,
                method=p.method,
                char_count=p.char_count,
                ocr_confidence=p.ocr_confidence,
            )
            for p in pages
        ],
        fields=fields,
        line_items=_items(
            x.line_items,
            assembled.draft.line_items,
            assembled.kept["line_items"],
            score.items["line_items"],
        ),
        markups=_items(
            x.markups, assembled.draft.markups, assembled.kept["markups"], score.items["markups"]
        ),
        signatures=_items(
            x.signatures,
            assembled.draft.signatures,
            assembled.kept["signatures"],
            score.items["signatures"],
        ),
        overall_confidence=score.overall,
        needs_review=score.needs_review,
        review_reasons=score.review_reasons,
        validation=[RuleOutcome(rule=r.rule, outcome=r.outcome, message=r.message) for r in rules],
        model_notes=x.notes,
        run=run,
    )


def _extract_once(pages, provider, model, settings, today):
    llm = extract_fields(
        pages, provider, model, temperature=settings.temperature, effort=settings.effort
    )
    assembled = assemble(llm.extraction)
    rules = validate(assembled.draft, today=today)
    score = score_document(llm.extraction, assembled, pages, rules, settings.review_threshold)
    return llm, assembled, rules, score


def _failed_count(rules: list[RuleResult]) -> int:
    return sum(r.failed for r in rules)


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
    sha = _sha256(path)
    try:
        pages = extract_text(path, vision_model=settings.vision_model)
        model = settings.model
        llm, assembled, rules, score = _extract_once(pages, provider, model, settings, today)
        spent = [(model, llm)]
        escalated = False
        if escalate and score.needs_review and settings.escalation_model != model:
            # Cascade: re-run a doubtful document on the stronger model and keep whichever
            # result fails fewer rules (ties go to the higher overall confidence).
            second = _extract_once(pages, provider, settings.escalation_model, settings, today)
            spent.append((settings.escalation_model, second[0]))
            first_key = (-_failed_count(rules), score.overall)
            second_key = (-_failed_count(second[2]), second[3].overall)
            if second_key > first_key:
                llm, assembled, rules, score = second
                model = settings.escalation_model
            escalated = True
    # RuntimeError covers LLM errors and unreadable files (pymupdf.FileDataError).
    except (RuntimeError, ValueError, OSError) as exc:
        return ExtractionResult(
            source_file=path.name,
            file_sha256=sha,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            review_reasons=["extraction failed"],
            run=RunInfo(seconds=round(time.monotonic() - start, 2)),
        )
    costs = [_cost(m, r.input_tokens, r.output_tokens) for m, r in spent]
    run = RunInfo(
        model=model,
        attempts=llm.attempts,
        cached=all(r.cached for r in llm.responses),
        escalated=escalated,
        input_tokens=sum(r.input_tokens for _, r in spent),
        output_tokens=sum(r.output_tokens for _, r in spent),
        cost_usd=None if None in costs else round(sum(costs), 5),  # all calls, incl. escalation
        seconds=round(time.monotonic() - start, 2),
        llm_seconds=round(sum(x.latency_s for _, r in spent for x in r.responses), 2),
    )
    return _compose(path, sha, pages, llm, assembled, rules, score, run)
