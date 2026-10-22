"""Turn the LLM's raw strings into a typed ChangeOrderDraft using the deterministic parsers.

A value that fails to parse becomes None in the draft and is recorded in `parse_errors`,
so scoring can see the difference between "absent" and "present but unreadable".
"""

from dataclasses import dataclass, field
from decimal import Decimal

from co_extract.llm import Extracted, LlmExtraction
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


@dataclass
class Assembled:
    draft: ChangeOrderDraft
    parse_errors: dict[str, str] = field(default_factory=dict)  # field path -> raw value
    defaulted: set[str] = field(default_factory=set)  # fields filled by a default, not the doc
    # for each group, the raw index of every entry kept in the draft (unreadable amounts
    # are dropped, so draft position i is not always raw position i)
    kept: dict[str, list[int]] = field(default_factory=dict)


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


def assemble(x: LlmExtraction) -> Assembled:
    errors: dict[str, str] = {}
    defaulted: set[str] = set()
    data: dict = {name: _text(getattr(x, name).value) for name in TEXT_FIELDS}

    for name in MONEY_FIELDS:
        data[name] = _parsed(name, getattr(x, name).value, parse_money, errors)
    data["date_issued"] = _parsed("date_issued", x.date_issued.value, parse_date, errors)
    data["schedule_impact_days"] = _parsed(
        "schedule_impact_days", x.schedule_impact_days.value, parse_days, errors
    )
    data["reason"] = x.reason.value
    data["status"] = x.status.value

    currency = _parsed("currency", x.currency.value, parse_currency, errors)
    if currency is None and "currency" not in errors:
        currency = DEFAULT_CURRENCY
        defaulted.add("currency")
    data["currency"] = currency

    items, kept_items = [], []
    for i, it in enumerate(x.line_items):
        amount = _parsed(f"line_items[{i}].amount", it.amount, parse_money, errors)
        if amount is None:
            continue  # an item without a readable amount can't be used in any total
        kept_items.append(i)
        items.append(
            LineItem(
                description=_text(it.description) or "",
                quantity=_parsed(f"line_items[{i}].quantity", it.quantity, parse_money, errors),
                unit=_text(it.unit),
                unit_cost=_parsed(f"line_items[{i}].unit_cost", it.unit_cost, parse_money, errors),
                amount=amount,
            )
        )
    data["line_items"] = items

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
        "line_items": kept_items,
        "markups": kept_markups,
        "signatures": list(range(len(x.signatures))),
    }
    return Assembled(ChangeOrderDraft(**data), errors, defaulted, kept)


def _percent(raw: str) -> Decimal | None:
    # the model may copy "10%" or just "10"
    return parse_percent(raw) or parse_money(raw)


def scalar_fields(x: LlmExtraction) -> dict[str, Extracted]:
    return {name: item for name, item in x if isinstance(item, Extracted)}
