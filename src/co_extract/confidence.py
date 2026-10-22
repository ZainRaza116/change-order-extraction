"""Stage 6 - per-field confidence from measurable signals, not from the model's self-report.

Signals (each 0-1) and weights:
    evidence_match     0.35  the quoted evidence really appears in the source text (fuzzy)
    value_in_evidence  0.20  the value actually appears inside its own evidence quote
    validity           0.20  the value parsed and no validation rule failed on it
    source_quality     0.15  1.0 for digital text, OCR mean word confidence for scans
    ambiguity          0.10  1.0 if one candidate; 1/(1+n) with n competing alternatives

Caps applied after the weighted sum:
    no value                      -> 0
    evidence not found in source  -> at most 0.3 (likely hallucination)
    a cross-field rule failed     -> at most 0.6 (below the review threshold)
    filled by a default           -> 0.5
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from co_extract.assemble import Assembled
from co_extract.llm import Extracted, LlmExtraction
from co_extract.schema import REQUIRED_FIELDS
from co_extract.text_extract import PageText
from co_extract.validate import RuleResult, failed_fields

WEIGHTS = {
    "evidence_match": 0.35,
    "value_in_evidence": 0.20,
    "validity": 0.20,
    "source_quality": 0.15,
    "ambiguity": 0.10,
}
EVIDENCE_THRESHOLD = 0.85  # fuzzy match below this = evidence not found in the source
CAP_EVIDENCE_NOT_FOUND = 0.3
CAP_RULE_FAILED = 0.6
DEFAULTED_CONFIDENCE = 0.5
VISION_SOURCE_QUALITY = 0.6  # no word confidences from a vision transcript; assume middling
DEFAULT_REVIEW_THRESHOLD = 0.7
MAPPED_FIELDS = {"reason", "status"}


@dataclass
class FieldScore:
    name: str
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    caps: list[str] = field(default_factory=list)  # why the score was capped, if it was


@dataclass
class DocumentScore:
    fields: dict[str, FieldScore]
    items: dict[str, list[FieldScore]]  # line_items / markups / signatures, per entry
    overall: float
    needs_review: bool
    review_reasons: list[str]


# --- signals (pure) -----------------------------------------------------------------


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def page_text(pages: list[PageText], page: int | None) -> str:
    chosen = [p for p in pages if p.page == page] or pages
    return "\n".join(p.text for p in chosen)


def evidence_match(evidence: str | None, source: str) -> float:
    """Best fuzzy alignment of the evidence quote anywhere in the source, 0-1."""
    if not evidence or not source:
        return 0.0
    return fuzz.partial_ratio(_squash(evidence), _squash(source)) / 100


def value_in_evidence(raw: str | None, evidence: str | None) -> float:
    """1.0 if the value is in its quote; 0.8 if only its digits are (e.g. '$2,250' vs
    '2250.00' formatting); 0.0 otherwise."""
    if not raw or not evidence:
        return 0.0
    if _squash(raw) in _squash(evidence):
        return 1.0
    digits = _digits(raw)
    if digits and digits in _digits(evidence):
        return 0.8
    return 0.0


def source_quality(pages: list[PageText], page: int | None) -> float:
    chosen = [p for p in pages if p.page == page] or pages
    scores = []
    for p in chosen:
        if p.method == "text":
            scores.append(1.0)
        elif p.method == "vision":
            scores.append(VISION_SOURCE_QUALITY)
        else:
            scores.append(p.ocr_confidence if p.ocr_confidence is not None else 0.0)
    return min(scores) if scores else 0.0


def ambiguity(alternatives: list[str], value: str | None) -> float:
    others = {_squash(a) for a in alternatives if a and _squash(a) != _squash(value or "")}
    return 1.0 / (1 + len(others))


def combine(signals: dict[str, float]) -> float:
    return sum(WEIGHTS[name] * signals[name] for name in WEIGHTS)


def score_entry(
    name: str,
    raw: str | None,
    evidence: str | None,
    page: int | None,
    alternatives: list[str],
    pages: list[PageText],
    *,
    parsed_ok: bool,
    rule_failed: bool,
    defaulted: bool = False,
    mapped: bool = False,
) -> FieldScore:
    """mapped=True for enum fields (reason, status): the value is a label chosen from the
    quote, not copied from it, so value_in_evidence falls back to the evidence match."""
    if defaulted:
        return FieldScore(name, DEFAULTED_CONFIDENCE, caps=["filled by default"])
    if raw is None or not str(raw).strip():
        return FieldScore(name, 0.0, caps=["no value"])
    found = evidence_match(evidence, page_text(pages, page))
    signals = {
        "evidence_match": found,
        "value_in_evidence": found if mapped else value_in_evidence(raw, evidence),
        "validity": 1.0 if parsed_ok and not rule_failed else 0.0,
        "source_quality": source_quality(pages, page),
        "ambiguity": ambiguity(alternatives, raw),
    }
    confidence, caps = combine(signals), []
    if signals["evidence_match"] < EVIDENCE_THRESHOLD:
        confidence = min(confidence, CAP_EVIDENCE_NOT_FOUND)
        caps.append("evidence not found in source")
    if rule_failed:
        confidence = min(confidence, CAP_RULE_FAILED)
        caps.append("validation rule failed")
    if not parsed_ok:
        caps.append("value could not be parsed")
    return FieldScore(name, round(confidence, 3), signals, caps)


# --- document -----------------------------------------------------------------------


def _row_raw(group: str, entry) -> str | None:
    """The value an item's evidence should contain: its amount, or for a signature the
    signer's name (an unsigned line has no name, so its own quote stands in)."""
    if group == "signatures":
        return entry.name or entry.evidence
    return entry.amount


def score_document(
    extraction: LlmExtraction,
    assembled: Assembled,
    pages: list[PageText],
    rules: list[RuleResult],
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> DocumentScore:
    failed = failed_fields(rules)
    errors = assembled.parse_errors

    fields = {}
    for name, item in extraction:
        if not isinstance(item, Extracted):
            continue
        raw = item.value.value if hasattr(item.value, "value") else item.value
        fields[name] = score_entry(
            name,
            raw,
            item.evidence,
            item.page,
            item.alternatives,
            pages,
            parsed_ok=name not in errors,
            rule_failed=name in failed,
            defaulted=name in assembled.defaulted,
            mapped=name in MAPPED_FIELDS,
        )

    items: dict[str, list[FieldScore]] = {}
    for group in ("line_items", "markups", "signatures"):
        entries = getattr(extraction, group)
        items[group] = [
            score_entry(
                f"{group}[{i}]",
                _row_raw(group, e),
                e.evidence,
                e.page,
                [],
                pages,
                parsed_ok=not any(k.startswith(f"{group}[{i}]") for k in errors),
                rule_failed=group in failed,
            )
            for i, e in enumerate(entries)
        ]

    required = [fields[name].confidence for name in REQUIRED_FIELDS]
    overall = round(sum(required) / len(required), 3)

    reasons = [
        f"{name} confidence {fields[name].confidence:.2f} < {review_threshold}"
        for name in REQUIRED_FIELDS
        if fields[name].confidence < review_threshold
    ]
    reasons += [f"rule {r.rule} failed: {r.message}" for r in rules if r.failed]
    reasons += [f"could not parse {path}: {raw!r}" for path, raw in errors.items()]
    return DocumentScore(fields, items, overall, bool(reasons), reasons)
