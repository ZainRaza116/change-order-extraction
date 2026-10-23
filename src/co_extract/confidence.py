"""Per-field confidence from measurable signals, not from the model's self-report.

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
    present but unparseable       -> at most 0.3 (the typed value is empty)
    filled by a default           -> 0.5
    issue date with a competing labelled date in the document -> at most 0.65
"""

import datetime as dt
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from co_extract.assemble import Assembled, Derivation
from co_extract.ingest import PageText
from co_extract.llm import LlmExtraction, raw_value
from co_extract.normalize import find_dates
from co_extract.schema import REQUIRED_FIELDS
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
CAP_UNPARSEABLE = 0.3
CAP_AMBIGUOUS_DATE = 0.65
PROXIMITY_SLACK = 15  # extra tokens allowed between the words of a stitched quote
DEFAULTED_CONFIDENCE = 0.5
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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _smallest_window(page: list[str], needed: set[str]) -> int | None:
    """Length of the shortest run of page tokens that contains every needed token."""
    counts: dict[str, int] = {}
    have, best, left = 0, None, 0
    for right, token in enumerate(page):
        if token in needed:
            counts[token] = counts.get(token, 0) + 1
            have += counts[token] == 1
        while have == len(needed):
            size = right - left + 1
            best = size if best is None else min(best, size)
            dropped = page[left]
            if dropped in needed:
                counts[dropped] -= 1
                have -= counts[dropped] == 0
            left += 1
    return best


def stitched_match(evidence: str, source: str) -> float:
    """1.0 if every word of the quote occurs on the page within a short span, else 0.0.

    Catches quotes the model stitched together from a multi-column layout (label in one
    column, value in another), which no contiguous match finds. The span limit (the
    quote's own length + PROXIMITY_SLACK tokens) keeps a quote assembled from words
    scattered across the page from passing; a number not on the page always fails.
    """
    needed = set(_tokens(evidence))
    if not needed:
        return 0.0
    window = _smallest_window(_tokens(source), needed)
    return 1.0 if window is not None and window <= len(needed) + PROXIMITY_SLACK else 0.0


def evidence_match(evidence: str | None, source: str) -> float:
    """How well the evidence quote is supported by the source text, 0-1: the better of a
    contiguous fuzzy match and a stitched match across nearby words."""
    if not evidence or not source:
        return 0.0
    contiguous = fuzz.partial_ratio(_squash(evidence), _squash(source)) / 100
    return max(contiguous, stitched_match(evidence, source))


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
        confidence = min(confidence, CAP_UNPARSEABLE)
        caps.append("value could not be parsed")
    return FieldScore(name, round(confidence, 3), signals, caps)


_DATE_LABEL = re.compile(
    r"(?:(\w+)\s+)?\b(date|dated|sent|issued)\b[\s:,-]*(?:[a-z]+,?\s*)?$"  # "Sent: Thursday, "
)
# "Contract Date:", "Substantial Completion Date" etc. name some other date
_OTHER_DATE_WORDS = {"contract", "completion", "substantial", "due", "bid", "start", "effective"}
_SUPERSEDED = re.compile(r"supersed|\bvoid|replac", re.I)


def competing_issue_dates(
    text: str, chosen: dt.date, co_number: str | None, other_dates: set[dt.date]
) -> set[dt.date]:
    """Labelled dates in the document that could also be this change order's issue date.

    Counts dates after a "Date:", "Sent:" or "Issued" label, and "dated X" when this CO's
    number sits just before it. Skips other kinds of date (contract, completion), dates of
    superseded documents, and `other_dates` (signature dates). Evidence matching cannot see
    this kind of error: a wrong date that is really on the page still passes every check.
    """
    co_digits = re.search(r"\d+", co_number or "")
    found = set()
    for start, date in find_dates(text):
        before = text[max(0, start - 40) : start].split("\n")[-1].lower()
        label = _DATE_LABEL.search(before)
        if not label or label[1] in _OTHER_DATE_WORDS or date in other_dates:
            continue
        if label[2] == "dated":
            mentions_co = co_digits and re.search(rf"\b0*{int(co_digits[0])}\b", before)
            if not mentions_co or _SUPERSEDED.search(text[max(0, start - 60) : start]):
                continue
        found.add(date)
    return found - {chosen}


def _score_derived(
    name: str, d: Derivation, pages: list[PageText], rule_failed: bool
) -> FieldScore:
    """A value filled by code: scored on the evidence it came from, capped by its kind."""
    score = score_entry(name, d.raw, d.evidence, d.page, [], pages, parsed_ok=True,
                        rule_failed=rule_failed)  # fmt: skip
    score.confidence = min(score.confidence, d.cap)
    score.caps.append(f"derived: {d.note}")
    return score


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
    draft = assembled.draft
    rivals: set[dt.date] = set()
    if draft.date_issued:
        signed_on = {sig.date for sig in draft.signatures if sig.date}
        text = "\n".join(p.text for p in pages)
        rivals = competing_issue_dates(
            text, draft.date_issued, draft.change_order_number, signed_on
        )

    derived = assembled.derived
    for name, item in extraction.scalars():
        if name in derived:
            fields[name] = _score_derived(name, derived[name], pages, name in failed)
            continue
        alternatives = list(item.alternatives)
        if name == "date_issued":
            alternatives += [d.isoformat() for d in sorted(rivals)]
        fields[name] = score_entry(
            name,
            raw_value(item),
            item.evidence,
            item.page,
            alternatives,
            pages,
            parsed_ok=name not in errors,
            rule_failed=name in failed,
            defaulted=name in assembled.defaulted,
            mapped=name in MAPPED_FIELDS,
        )
    if rivals:
        date_score = fields["date_issued"]
        date_score.confidence = min(date_score.confidence, CAP_AMBIGUOUS_DATE)
        listed = ", ".join(d.isoformat() for d in sorted(rivals))
        date_score.caps.append(f"other candidate issue dates in the document: {listed}")

    items: dict[str, list[FieldScore]] = {}
    for group in ("line_items", "markups", "signatures"):
        entries = getattr(extraction, group)
        items[group] = [
            _score_derived(f"{group}[{i}]", derived[f"{group}[{i}]"], pages, group in failed)
            if f"{group}[{i}]" in derived
            else score_entry(
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
    reasons += [f"{path}: {d.note}" for path, d in derived.items() if d.cap < review_threshold]
    return DocumentScore(fields, items, overall, bool(reasons), reasons)
