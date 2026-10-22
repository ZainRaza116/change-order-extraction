"""Compare pipeline outputs with hand-labelled ground truth.

Reports field accuracy (all / required), hallucination rate, line-item and markup
recall, a calibration table (accuracy per confidence bucket), how many errors were
flagged for review, and cost and latency per document.

Usage: co-extract eval [--outputs outputs] [--expected expected]
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from co_extract.schema import REQUIRED_FIELDS

BUCKETS = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.0001))
REVIEW_THRESHOLD = 0.7
NAME_MIN = 90  # rapidfuzz ratio for party names after normalisation
TEXT_MIN = 80  # token_sort_ratio for free text

# --- comparators -------------------------------------------------------------------------

_NAME_SUFFIXES = {"inc", "llc", "ltd", "co", "corp", "corporation", "company", "aia", "pc", "pllc"}


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def same_money(a: Any, b: Any) -> bool:
    try:
        return abs(Decimal(str(a)) - Decimal(str(b))) <= Decimal("0.01")
    except InvalidOperation:
        return False


def same_co_number(a: Any, b: Any) -> bool:
    """'007' == 'CO-007' == 'CO #7'; '7A' != '7'."""

    def key(v: Any) -> tuple[int, str] | None:
        m = re.search(r"(\d+)\s*([a-z]?)\s*$", str(v).lower())
        return (int(m[1]), m[2]) if m else None

    return key(a) is not None and key(a) == key(b)


def same_name(a: Any, b: Any) -> bool:
    def norm(v: Any) -> str:
        return " ".join(w for w in _words(str(v)) if w not in _NAME_SUFFIXES)

    return fuzz.ratio(norm(a), norm(b)) >= NAME_MIN


def same_text(a: Any, b: Any) -> bool:
    return fuzz.token_sort_ratio(" ".join(_words(str(a))), " ".join(_words(str(b)))) >= TEXT_MIN


def same_exact(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


COMPARATORS = {
    "change_order_number": same_co_number,
    "project_name": same_name,
    "project_number": same_exact,  # codes: a dropped digit or hyphen is an error
    "date_issued": same_exact,
    "contractor": same_name,
    "owner": same_name,
    "architect_engineer": same_name,
    "description": same_text,
    "reason": same_exact,
    "reason_text": same_text,
    "subtotal": same_money,
    "change_amount": same_money,
    "currency": same_exact,
    "original_contract_sum": same_money,
    "previous_changes_total": same_money,
    "revised_contract_sum": same_money,
    "schedule_impact_days": same_exact,
    "status": same_exact,
}


# --- per-document scoring -------------------------------------------------------------------


@dataclass
class FieldCheck:
    document: str
    field: str
    outcome: str  # correct | correct_null | wrong | missed | hallucinated
    confidence: float
    predicted: Any = None
    expected: Any = None

    @property
    def correct(self) -> bool:
        return self.outcome in ("correct", "correct_null")


def check_field(document: str, name: str, predicted: Any, expected: Any, conf: float) -> FieldCheck:
    if expected is None and predicted is None:
        outcome = "correct_null"
    elif expected is None:
        outcome = "hallucinated"
    elif predicted is None:
        outcome = "missed"
    else:
        outcome = "correct" if COMPARATORS[name](predicted, expected) else "wrong"
    return FieldCheck(document, name, outcome, conf, predicted, expected)


def _amount_matches(predicted: list[Any], expected: list[Any]) -> tuple[int, list[bool]]:
    """Multiset match of amounts: how many expected amounts were found, and for each
    predicted entry whether it matched an expected one."""
    remaining = Counter(f"{Decimal(str(a)):.2f}" for a in expected)
    hits = []
    for a in predicted:
        try:
            key = f"{Decimal(str(a)):.2f}"
        except (InvalidOperation, TypeError):
            hits.append(False)
            continue
        if remaining[key] > 0:
            remaining[key] -= 1
            hits.append(True)
        else:
            hits.append(False)
    return sum(hits), hits


@dataclass
class ItemCheck:
    document: str
    group: str
    expected: int
    predicted: int
    matched: int
    entries: list[tuple[float, bool]] = field(default_factory=list)  # (confidence, correct)


def check_items(document: str, group: str, result: dict, truth: dict) -> ItemCheck:
    predicted = result.get(group, [])
    expected = truth.get(group, [])
    amounts = []
    for entry in predicted:
        value = entry["value"].get("amount")
        try:
            amounts.append(Decimal(str(value)) if value is not None else None)
        except InvalidOperation:
            amounts.append(None)
    matched, hits = _amount_matches(amounts, [e["amount"] for e in expected])
    entries = [(e["confidence"], hit) for e, hit in zip(predicted, hits, strict=True)]
    return ItemCheck(document, group, len(expected), len(predicted), matched, entries)


# --- report ------------------------------------------------------------------------------


@dataclass
class Report:
    documents: list[str]
    fields: list[FieldCheck]
    items: list[ItemCheck]
    runs: list[dict]
    errors: list[str]  # documents whose pipeline run failed

    def accuracy(self, names: tuple[str, ...] | None = None) -> float:
        checks = [c for c in self.fields if names is None or c.field in names]
        return sum(c.correct for c in checks) / len(checks) if checks else 0.0

    def hallucinations(self) -> tuple[int, int]:
        null_truth = [c for c in self.fields if c.expected is None]
        return sum(c.outcome == "hallucinated" for c in null_truth), len(null_truth)

    def calibration(self) -> list[dict]:
        points = [(c.confidence, c.correct) for c in self.fields if c.predicted is not None]
        points += [p for i in self.items for p in i.entries]
        rows = []
        for low, high in BUCKETS:
            inside = [ok for conf, ok in points if low <= conf < high]
            rows.append(
                {
                    "bucket": f"{low:.1f}-{min(high, 1.0):.1f}",
                    "n": len(inside),
                    "accuracy": sum(inside) / len(inside) if inside else None,
                }
            )
        return rows

    def errors_flagged(self) -> tuple[int, int]:
        """Wrong/missed/hallucinated fields that were flagged (confidence < threshold)."""
        bad = [c for c in self.fields if not c.correct]
        return sum(c.confidence < REVIEW_THRESHOLD for c in bad), len(bad)

    def to_dict(self) -> dict:
        h, n = self.hallucinations()
        flagged, bad = self.errors_flagged()
        return {
            "documents": self.documents,
            "field_accuracy": round(self.accuracy(), 4),
            "required_field_accuracy": round(self.accuracy(REQUIRED_FIELDS), 4),
            "hallucinated": h,
            "null_fields": n,
            "errors_flagged": flagged,
            "errors_total": bad,
            "calibration": self.calibration(),
            "items": [asdict(i) for i in self.items],
            "fields": [asdict(c) | {"correct": c.correct} for c in self.fields],
            "runs": self.runs,
            "failed_documents": self.errors,
        }


def evaluate(outputs: Path, expected: Path) -> Report:
    fields, items, runs, errors, documents = [], [], [], [], []
    for truth_path in sorted(Path(expected).glob("*.json")):
        name = truth_path.stem
        result_path = Path(outputs) / f"{name}.json"
        if not result_path.exists():
            continue
        truth = json.loads(truth_path.read_text())
        result = json.loads(result_path.read_text())
        documents.append(name)
        runs.append({"document": name, "needs_review": result["needs_review"], **result["run"]})
        if result["status"] != "ok":
            errors.append(name)
        for field_name in COMPARATORS:
            got = result.get("fields", {}).get(field_name, {})
            fields.append(
                check_field(
                    name,
                    field_name,
                    got.get("value"),
                    truth.get(field_name),
                    got.get("confidence", 0.0),
                )
            )
        for group in ("line_items", "markups"):
            items.append(check_items(name, group, result, truth))
    return Report(documents, fields, items, runs, errors)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def render_markdown(report: Report) -> str:
    h, n = report.hallucinations()
    flagged, bad = report.errors_flagged()
    li = [i for i in report.items if i.group == "line_items"]
    mk = [i for i in report.items if i.group == "markups"]
    li_exp, li_hit = sum(i.expected for i in li), sum(i.matched for i in li)
    mk_exp, mk_hit = sum(i.expected for i in mk), sum(i.matched for i in mk)
    costs = [r["cost_usd"] for r in report.runs if r.get("cost_usd") is not None]
    latency = [r["llm_seconds"] for r in report.runs if r.get("llm_seconds")]
    models = sorted({r["model"] for r in report.runs if r.get("model")})

    lines = [
        f"Evaluated {len(report.documents)} documents with {', '.join(models) or 'n/a'}.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Field accuracy (all {len(COMPARATORS)} fields) | {_pct(report.accuracy())} |",
        f"| Required-field accuracy | {_pct(report.accuracy(REQUIRED_FIELDS))} |",
        f"| Hallucinated values (value where truth is empty) | {h} of {n} |",
        f"| Line-item amounts recovered | {li_hit} of {li_exp} |",
        f"| Markup amounts recovered | {mk_hit} of {mk_exp} |",
        f"| Wrong fields flagged (confidence < {REVIEW_THRESHOLD}) | {flagged} of {bad} |",
        f"| Documents flagged for review | "
        f"{sum(r['needs_review'] for r in report.runs)} of {len(report.runs)} |",
    ]
    if costs:
        lines.append(f"| Cost per document (mean) | ${sum(costs) / len(costs):.4f} |")
    if latency:
        lines.append(f"| LLM latency per document (mean) | {sum(latency) / len(latency):.1f} s |")
    if report.errors:
        lines.append(f"| Failed documents | {', '.join(report.errors)} |")

    lines += ["", "**Calibration** (does higher confidence mean more often right?)", ""]
    lines += ["| Confidence | Values | Accuracy |", "|---|---|---|"]
    lines += [f"| {r['bucket']} | {r['n']} | {_pct(r['accuracy'])} |" for r in report.calibration()]

    wrong = [c for c in report.fields if not c.correct]
    if wrong:
        lines += [
            "",
            "**Errors**",
            "",
            "| Document | Field | Outcome | Confidence | Got | Expected |",
        ]
        lines += ["|---|---|---|---|---|---|"]
        for c in wrong:
            got, exp = str(c.predicted)[:40], str(c.expected)[:40]
            lines.append(
                f"| {c.document} | {c.field} | {c.outcome} | {c.confidence:.2f} | {got} | {exp} |"
            )
    return "\n".join(lines) + "\n"
