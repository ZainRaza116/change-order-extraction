"""Turn the LLM's raw strings into a typed ChangeOrderDraft using the deterministic parsers.

A value that fails to parse becomes None in the draft and is recorded in `parse_errors`,
so scoring can see the difference between "absent" and "present but unreadable".

A few gaps are filled by code rather than the model, each recorded as a `Derivation` with
its source, evidence and confidence cap:
- currency from the symbol printed with the amounts ("$" -> USD)
- schedule impact 0 from wording such as "no time extension" or "Unchanged (0 days)"
- one unreadable line-item amount from the subtotal minus the other items
"""

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from co_extract.ingest import PageText
from co_extract.llm import LlmExtraction
from co_extract.normalize import (
    parse_currency,
    parse_date,
    parse_days,
    parse_money,
    parse_percent,
)
from co_extract.schema import ChangeOrderDraft, LineItem, Markup, Signature

MONEY_FIELDS = (
    "subtotal",
    "change_amount",
    "original_contract_sum",
    "previous_changes_total",
    "revised_contract_sum",
)
TEXT_FIELDS = (
    "change_order_number",
    "project_name",
    "project_number",
    "contractor",
    "owner",
    "architect_engineer",
    "description",
    "reason_text",
)
DEFAULT_CURRENCY = "USD"
CENT = Decimal("0.01")

# Derived values: reliable normalisations stay above the review threshold; an inferred
# amount stays below it so a person checks it.
CAP_NORMALISED = 0.9
CAP_INFERRED = 0.6

_CURRENCY_MARK = re.compile(r"US\$|C\$|A\$|[$€£]|\b(?:USD|CAD|AUD|EUR|GBP)\b")
_NO_TIME_CHANGE = re.compile(
    r"no (?:time )?extension"
    r"|no (?:change|increase) (?:in|to) (?:the )?(?:contract )?time"
    r"|(?:contract )?time[^.\n]{0,40}\bunchanged\b"
    r"|unchanged\s*\(\s*0\s*days?\s*\)"
    r"|\b0\s*(?:calendar |working )?days\b",
    re.I,
)


@dataclass
class Derivation:
    """A value filled by code, not copied by the model."""

    raw: str  # the value as text
    evidence: str | None  # the text it was derived from
    page: int | None
    note: str  # how it was derived, shown in the output flags
    cap: float  # highest confidence it may get


@dataclass
class Assembled:
    draft: ChangeOrderDraft
    parse_errors: dict[str, str] = field(default_factory=dict)  # field path -> raw value
    defaulted: set[str] = field(default_factory=set)  # fields filled by a default, not the doc
    # for each group, the raw index of every entry kept in the draft (unreadable amounts
    # are dropped, so draft position i is not always raw position i)
    kept: dict[str, list[int]] = field(default_factory=dict)
    derived: dict[str, Derivation] = field(default_factory=dict)  # "currency", "line_items[0]"


def _text(raw: str | None) -> str | None:
    if raw is None:
        return None
    return " ".join(raw.split()) or None


def _parsed(path: str, raw: str | None, parser, errors: dict[str, str]):
    if raw is None or not raw.strip():
        return None
    value = parser(raw)
    if value is None:
        errors[path] = raw
    return value


def _currency_from_amounts(x: LlmExtraction) -> Derivation | None:
    """Currency printed with the amounts, e.g. "$10,763.06" -> USD."""
    for name in ("change_amount", "revised_contract_sum", "original_contract_sum", "subtotal"):
        item = getattr(x, name)
        mark = _CURRENCY_MARK.search(item.value or "")
        if mark and (code := parse_currency(mark[0])):
            return Derivation(
                code, item.evidence, item.page, f"currency from '{mark[0]}' in {name}",
                CAP_NORMALISED,
            )  # fmt: skip
    return None


def _no_time_change(pages: list[PageText]) -> Derivation | None:
    """A stated "no time extension" means 0 days, not unknown."""
    for page in pages:
        if m := _NO_TIME_CHANGE.search(page.text):
            line_start = page.text.rfind("\n", 0, m.start()) + 1
            line_end = page.text.find("\n", m.end())
            line = page.text[line_start : line_end if line_end != -1 else None].strip()
            return Derivation("0", line, page.page, f"0 days from '{m[0]}'", CAP_NORMALISED)
    return None


def assemble(x: LlmExtraction, pages: list[PageText] | None = None) -> Assembled:
    errors: dict[str, str] = {}
    defaulted: set[str] = set()
    derived: dict[str, Derivation] = {}
    data: dict = {name: _text(getattr(x, name).value) for name in TEXT_FIELDS}

    for name in MONEY_FIELDS:
        data[name] = _parsed(name, getattr(x, name).value, parse_money, errors)
    data["date_issued"] = _parsed("date_issued", x.date_issued.value, parse_date, errors)
    data["schedule_impact_days"] = _parsed(
        "schedule_impact_days", x.schedule_impact_days.value, parse_days, errors
    )
    if x.schedule_impact_days.value is None and pages and (d := _no_time_change(pages)):
        data["schedule_impact_days"] = 0
        derived["schedule_impact_days"] = d
    data["reason"] = x.reason.value
    data["status"] = x.status.value

    # The model's currency counts only with a quote; otherwise the amounts' symbol decides.
    quoted = x.currency.value if x.currency.evidence else None
    currency = _parsed("currency", quoted, parse_currency, errors)
    if currency is None and "currency" not in errors:
        if d := _currency_from_amounts(x):
            currency = d.raw
            derived["currency"] = d
        else:
            currency = DEFAULT_CURRENCY
            defaulted.add("currency")
    data["currency"] = currency

    markups, kept_markups = [], []
    for i, m in enumerate(x.markups):
        amount = _parsed(f"markups[{i}].amount", m.amount, parse_money, errors)
        if amount is None:
            continue
        rate = _parsed(f"markups[{i}].rate_percent", m.rate_percent, _percent, errors)
        kept_markups.append(i)
        markups.append(
            Markup(description=_text(m.description) or "", rate_percent=rate, amount=amount)
        )
    data["markups"] = markups

    items: dict[int, LineItem] = {}
    unreadable: list[int] = []
    for i, it in enumerate(x.line_items):
        amount = _parsed(f"line_items[{i}].amount", it.amount, parse_money, errors)
        if amount is None:
            unreadable.append(i)
            continue
        items[i] = LineItem(
            description=_text(it.description) or "",
            quantity=_parsed(f"line_items[{i}].quantity", it.quantity, parse_money, errors),
            unit=_text(it.unit),
            unit_cost=_parsed(f"line_items[{i}].unit_cost", it.unit_cost, parse_money, errors),
            amount=amount,
        )
    if len(unreadable) == 1:
        i = unreadable[0]
        if recovered := _recover_item(x, i, items, data, markups):
            items[i], derived[f"line_items[{i}]"] = recovered
    data["line_items"] = [items[i] for i in sorted(items)]
    items_total = sum((it.amount for it in items.values()), Decimal(0))
    if (
        data["subtotal"] is None
        and markups
        and items
        and pages
        and (d := _printed_subtotal(pages, items_total))
    ):
        data["subtotal"] = parse_money(d.raw)
        derived["subtotal"] = d

    data["signatures"] = [
        Signature(
            party=s.party.value,
            name=_text(s.name),
            date=_parsed(f"signatures[{i}].date", s.date, parse_date, errors),
            signed=s.signed,
        )
        for i, s in enumerate(x.signatures)
    ]
    kept = {
        "line_items": sorted(items),
        "markups": kept_markups,
        "signatures": list(range(len(x.signatures))),
    }
    return Assembled(ChangeOrderDraft(**data), errors, defaulted, kept, derived)


_MONEY_TOKEN = re.compile(r"\(?-?\$?\d[\d,]*\.\d{2}\)?")


def _printed_subtotal(pages: list[PageText], items_total: Decimal) -> Derivation | None:
    """A printed figure equal to the line items' sum is the subtotal, whatever its label
    ("Subtotal", "Net of items", "Total of items")."""
    for page in pages:
        for line in page.text.splitlines():
            for token in _MONEY_TOKEN.findall(line):
                if parse_money(token) == items_total:
                    note = f"subtotal: printed figure equal to the line items' sum ({token})"
                    return Derivation(token, line.strip(), page.page, note, CAP_NORMALISED)
    return None


def _recover_item(
    x: LlmExtraction, i: int, items: dict[int, LineItem], data: dict, markups: list[Markup]
) -> tuple[LineItem, Derivation] | None:
    """The one unreadable amount = subtotal (or change amount less markups) minus the other
    items. Only when every other item and the total are readable."""
    if len(items) != len(x.line_items) - 1:
        return None
    if data["subtotal"] is not None:
        base, source = data["subtotal"], x.subtotal
    elif data["change_amount"] is not None:
        base = data["change_amount"] - sum((m.amount for m in markups), Decimal(0))
        source = x.change_amount
    else:
        return None
    amount = base - sum((item.amount for item in items.values()), Decimal(0))
    raw = x.line_items[i]
    quantity = parse_money(raw.quantity) if raw.quantity else None
    unit_cost = parse_money(raw.unit_cost) if raw.unit_cost else None
    if unit_cost is None and quantity:
        unit_cost = (abs(amount) / quantity).quantize(CENT, rounding=ROUND_HALF_UP)
    item = LineItem(
        description=_text(raw.description) or "",
        quantity=quantity,
        unit=_text(raw.unit),
        unit_cost=unit_cost,
        amount=amount,
    )
    note = "amount derived: total less the other items (value unreadable in the document)"
    return item, Derivation(f"{amount:.2f}", source.evidence, source.page, note, CAP_INFERRED)


def _percent(raw: str) -> Decimal | None:
    # the model may copy "10%" or just "10"
    return parse_percent(raw) or parse_money(raw)
