"""Stage 5 - domain rules: do the extracted values make sense together?

Each rule is a pure function of a ChangeOrderDraft that returns pass, fail or
not-applicable (when the fields it needs are missing). A failed rule never changes a
value; the scoring stage lowers confidence on the fields it names.
"""

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from co_extract.schema import REQUIRED_FIELDS, ChangeOrderDraft, Status

MONEY_TOLERANCE = Decimal("1.00")  # rounding slack on printed totals
EARLIEST_PLAUSIBLE = dt.date(1990, 1, 1)
MAX_SCHEDULE_DAYS = 3650

# "delete"/"omit" alone also appear in add-and-replace items, so only these count.
_CREDIT_WORDS = re.compile(r"\b(credit|deduct|deduction)\b", re.I)
_CO_NUMBER = re.compile(r"^(?:[A-Z]{1,4}\s*[-#]?\s*)?\d{1,4}[A-Z]?$", re.I)
_ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RuleResult:
    rule: str
    outcome: Outcome
    message: str
    fields: tuple[str, ...] = ()  # fields to penalise when the rule fails

    @property
    def failed(self) -> bool:
        return self.outcome is Outcome.FAIL


def _pass(rule: str, message: str, *fields: str) -> RuleResult:
    return RuleResult(rule, Outcome.PASS, message, fields)


def _fail(rule: str, message: str, *fields: str) -> RuleResult:
    return RuleResult(rule, Outcome.FAIL, message, fields)


def _na(rule: str, message: str) -> RuleResult:
    return RuleResult(rule, Outcome.NOT_APPLICABLE, message)


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


# --- rules ----------------------------------------------------------------------------


def required_fields_present(co: ChangeOrderDraft) -> RuleResult:
    missing = [name for name in REQUIRED_FIELDS if getattr(co, name) in (None, "")]
    if missing:
        return _fail("required_fields", f"missing: {', '.join(missing)}", *missing)
    return _pass("required_fields", "all required fields present")


def contract_math(co: ChangeOrderDraft) -> RuleResult:
    rule = "contract_math"
    fields = (
        "original_contract_sum",
        "previous_changes_total",
        "change_amount",
        "revised_contract_sum",
    )
    values = [getattr(co, f) for f in fields]
    if None in values:
        missing = [f for f, v in zip(fields, values, strict=True) if v is None]
        return _na(rule, f"needs {', '.join(missing)}")
    original, previous, change, revised = values
    expected = original + previous + change
    if abs(expected - revised) > MONEY_TOLERANCE:
        return _fail(
            rule,
            f"{_money(original)} + {_money(previous)} + {_money(change)} = {_money(expected)}, "
            f"but revised sum is {_money(revised)}",
            *fields,
        )
    return _pass(rule, f"original + previous + change = revised ({_money(revised)})", *fields)


def line_items_total(co: ChangeOrderDraft) -> RuleResult:
    """Line items (+ subtotal if printed) + markups add up to the change amount."""
    rule = "line_items_total"
    if not co.line_items:
        return _na(rule, "no line items")
    if co.change_amount is None:
        return _na(rule, "needs change_amount")
    items = sum((i.amount for i in co.line_items), Decimal(0))
    if co.subtotal is not None and abs(items - co.subtotal) > MONEY_TOLERANCE:
        return _fail(
            rule,
            f"line items sum to {_money(items)}, but subtotal is {_money(co.subtotal)}",
            "line_items",
            "subtotal",
        )
    markups = sum((m.amount for m in co.markups), Decimal(0))
    total = items + markups
    if abs(total - co.change_amount) > MONEY_TOLERANCE:
        involved = ["line_items", "change_amount"] + (["markups"] if co.markups else [])
        return _fail(
            rule,
            f"line items {_money(items)} + markups {_money(markups)} = {_money(total)}, "
            f"but change amount is {_money(co.change_amount)}",
            *involved,
        )
    return _pass(rule, f"items + markups = change amount ({_money(total)})", "line_items")


def line_item_math(co: ChangeOrderDraft) -> RuleResult:
    """quantity x unit cost = amount for every item that prints all three."""
    rule = "line_item_math"
    checked, bad = 0, []
    for i, item in enumerate(co.line_items):
        if item.quantity is None or item.unit_cost is None:
            continue
        checked += 1
        expected = abs(item.quantity * item.unit_cost)
        if abs(expected - abs(item.amount)) > MONEY_TOLERANCE:
            bad.append((i, expected, item.amount))
    if not checked:
        return _na(rule, "no item has both quantity and unit cost")
    if bad:
        detail = "; ".join(
            f"item {i + 1}: qty x unit cost = {_money(e)}, amount = {_money(a)}" for i, e, a in bad
        )
        return _fail(rule, detail, *(f"line_items[{i}]" for i, _, _ in bad))
    return _pass(rule, f"{checked} item(s) check out")


def markup_math(co: ChangeOrderDraft) -> RuleResult:
    """rate x base = amount, where the base is the subtotal or the subtotal plus earlier
    markups (bonds are often charged on the marked-up total)."""
    rule = "markup_math"
    with_rate = [(i, m) for i, m in enumerate(co.markups) if m.rate_percent is not None]
    if not with_rate:
        return _na(rule, "no markup with a stated rate")
    base = co.subtotal
    if base is None and co.line_items:
        base = sum((i.amount for i in co.line_items), Decimal(0))
    if base is None:
        return _na(rule, "needs subtotal or line items")
    bad, running = [], base
    for i, m in enumerate(co.markups):
        amount = m.amount
        if m.rate_percent is not None:
            candidates = (base * m.rate_percent / 100, running * m.rate_percent / 100)
            if all(abs(c - m.amount) > MONEY_TOLERANCE for c in candidates):
                bad.append((i, m))
                # carry the correct amount forward so one bad markup doesn't fail the next
                amount = candidates[1]
        running += amount
    if bad:
        detail = "; ".join(
            f"{m.description} {m.rate_percent}% of {_money(base)} != {_money(m.amount)}"
            for _, m in bad
        )
        return _fail(rule, detail, *(f"markups[{i}]" for i, _ in bad))
    return _pass(rule, f"{len(with_rate)} markup(s) check out")


def credit_signs(co: ChangeOrderDraft) -> RuleResult:
    """Items described as a credit/deduct must carry a negative amount."""
    rule = "credit_signs"
    credits = [
        (i, it) for i, it in enumerate(co.line_items) if _CREDIT_WORDS.search(it.description)
    ]
    if not credits:
        return _na(rule, "no credit or deduct items")
    wrong = [i for i, it in credits if it.amount > 0]
    if wrong:
        return _fail(
            rule,
            "credit item(s) with a positive amount: " + ", ".join(str(i + 1) for i in wrong),
            *(f"line_items[{i}]" for i in wrong),
        )
    return _pass(rule, f"{len(credits)} credit item(s) are negative")


def date_sanity(co: ChangeOrderDraft, today: dt.date | None = None) -> RuleResult:
    rule = "date_sanity"
    if co.date_issued is None:
        return _na(rule, "needs date_issued")
    today = today or dt.date.today()
    if co.date_issued > today:
        return _fail(rule, f"date issued {co.date_issued} is in the future", "date_issued")
    if co.date_issued < EARLIEST_PLAUSIBLE:
        return _fail(rule, f"date issued {co.date_issued} is implausibly early", "date_issued")
    early = [(i, s) for i, s in enumerate(co.signatures) if s.date and s.date < co.date_issued]
    if early:
        detail = ", ".join(f"{s.party} signed {s.date}" for _, s in early)
        return _fail(
            rule,
            f"signed before the issue date {co.date_issued}: {detail}",
            "date_issued",
            *(f"signatures[{i}]" for i, _ in early),
        )
    return _pass(rule, "dates are plausible and in order", "date_issued")


def co_number_format(co: ChangeOrderDraft) -> RuleResult:
    rule = "co_number_format"
    if not co.change_order_number:
        return _na(rule, "needs change_order_number")
    if not _CO_NUMBER.match(co.change_order_number.strip()):
        return _fail(
            rule,
            f"{co.change_order_number!r} does not look like a change order number",
            "change_order_number",
        )
    return _pass(rule, "looks like a change order number", "change_order_number")


def status_consistency(co: ChangeOrderDraft) -> RuleResult:
    rule = "status_consistency"
    if co.status is None:
        return _na(rule, "no status")
    signed = [s for s in co.signatures if s.signed]
    unsigned = [s for s in co.signatures if not s.signed]
    if co.status is Status.APPROVED:
        if not signed:
            return _fail(rule, "approved, but no signature found", "status", "signatures")
        if unsigned:
            missing = ", ".join(s.party for s in unsigned)
            return _fail(rule, f"approved, but unsigned: {missing}", "status", "signatures")
    if co.status in (Status.PENDING, Status.PROPOSED) and co.signatures and not unsigned:
        return _fail(rule, f"{co.status} although every listed party signed", "status")
    return _pass(rule, f"status {co.status} matches the signatures", "status")


def currency_code(co: ChangeOrderDraft) -> RuleResult:
    rule = "currency_code"
    if not co.currency:
        return _na(rule, "no currency")
    if not _ISO_CURRENCY.match(co.currency):
        return _fail(rule, f"{co.currency!r} is not a 3-letter ISO currency code", "currency")
    return _pass(rule, f"{co.currency} is a valid code", "currency")


def schedule_days_plausible(co: ChangeOrderDraft) -> RuleResult:
    rule = "schedule_days"
    if co.schedule_impact_days is None:
        return _na(rule, "no schedule impact")
    if abs(co.schedule_impact_days) > MAX_SCHEDULE_DAYS:
        return _fail(rule, f"{co.schedule_impact_days} days is implausible", "schedule_impact_days")
    return _pass(rule, f"{co.schedule_impact_days} day(s)", "schedule_impact_days")


RULES: tuple[Callable[[ChangeOrderDraft], RuleResult], ...] = (
    required_fields_present,
    contract_math,
    line_items_total,
    line_item_math,
    markup_math,
    credit_signs,
    date_sanity,
    co_number_format,
    status_consistency,
    currency_code,
    schedule_days_plausible,
)


def validate(co: ChangeOrderDraft, today: dt.date | None = None) -> list[RuleResult]:
    results = []
    for rule in RULES:
        results.append(rule(co, today) if rule is date_sanity else rule(co))
    return results


def failed_fields(results: list[RuleResult]) -> set[str]:
    """Top-level field names touched by any failed rule ("line_items[2]" -> "line_items")."""
    return {f.split("[")[0] for r in results if r.failed for f in r.fields}
