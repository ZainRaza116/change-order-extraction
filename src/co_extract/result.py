"""The public output: one ExtractionResult per document (CLI JSON files and API responses).

`build_result` maps the pipeline's internal objects onto this model, so internal types can
change without changing the output contract.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from co_extract import __version__
from co_extract.assemble import Assembled, Derivation
from co_extract.confidence import DocumentScore, FieldScore
from co_extract.ingest import PageText
from co_extract.llm import LlmExtraction, raw_value
from co_extract.validate import RuleResult


class FieldResult(BaseModel):
    value: Any  # typed and normalised (Decimal as string, ISO date, int, enum value)
    raw: str | None  # exactly as the model copied it from the document
    confidence: float
    evidence: str | None
    page: int | None
    alternatives: list[str] = []
    flags: list[str] = []  # why the confidence was capped, if it was


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


def _items(
    group: str,
    raw_entries,
    typed_entries,
    kept: list[int],
    scores: list[FieldScore],
    derived: dict[str, Derivation],
) -> list[ItemResult]:
    """One ItemResult per raw entry. Entries that assemble() dropped (unreadable amount)
    keep their raw text with amount None, so every other entry still gets its typed value."""
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
                evidence=raw.evidence or getattr(derived.get(f"{group}[{i}]"), "evidence", None),
                page=raw.page or getattr(derived.get(f"{group}[{i}]"), "page", None),
                flags=score.caps,
            )
        )
    return out


def build_result(
    path: Path,
    sha256: str,
    pages: list[PageText],
    extraction: LlmExtraction,
    assembled: Assembled,
    rules: list[RuleResult],
    score: DocumentScore,
    run: RunInfo,
) -> ExtractionResult:
    draft = assembled.draft.model_dump(mode="json")
    fields = {}
    for name, item in extraction.scalars():
        d = assembled.derived.get(name)
        fields[name] = FieldResult(
            value=draft.get(name),
            raw=d.raw if d else raw_value(item),
            confidence=score.fields[name].confidence,
            evidence=d.evidence if d else item.evidence,
            page=d.page if d else item.page,
            alternatives=item.alternatives,
            flags=score.fields[name].caps,
        )
    groups = {
        group: _items(
            group,
            getattr(extraction, group),
            getattr(assembled.draft, group),
            assembled.kept[group],
            score.items[group],
            assembled.derived,
        )
        for group in ("line_items", "markups", "signatures")
    }
    return ExtractionResult(
        source_file=path.name,
        file_sha256=sha256,
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
        **groups,
        overall_confidence=score.overall,
        needs_review=score.needs_review,
        review_reasons=score.review_reasons,
        validation=[RuleOutcome(rule=r.rule, outcome=r.outcome, message=r.message) for r in rules],
        model_notes=extraction.notes,
        run=run,
    )


def error_result(path: Path, sha256: str, exc: Exception, seconds: float) -> ExtractionResult:
    return ExtractionResult(
        source_file=path.name,
        file_sha256=sha256,
        status="error",
        error=f"{type(exc).__name__}: {exc}",
        review_reasons=["extraction failed"],
        run=RunInfo(seconds=round(seconds, 2)),
    )
