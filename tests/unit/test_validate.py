import datetime as dt
import json
from decimal import Decimal

import pytest

from co_extract import validate as v
from co_extract.schema import ChangeOrderDraft, LineItem, Markup, Status
from co_extract.validate import Outcome
from tests.factories import EXPECTED

TODAY = dt.date(2026, 10, 21)


def load(name: str) -> ChangeOrderDraft:
    return ChangeOrderDraft.model_validate(json.loads((EXPECTED / name).read_text()))


@pytest.fixture
def co() -> ChangeOrderDraft:
    """The real scanned CO-007: line items, markups, three signatures, owner unsigned."""
    return load("co_004_real_scan.json")


def change(co: ChangeOrderDraft, **updates) -> ChangeOrderDraft:
    return co.model_copy(update=updates)


# --- ground truth is internally consistent ----------


@pytest.mark.parametrize("path", sorted(EXPECTED.glob("*.json")), ids=lambda p: p.name)
def test_no_rule_fails_on_ground_truth(path):
    results = v.validate(load(path.name), today=TODAY)
    assert [r for r in results if r.failed] == []
    assert len(results) == len(v.RULES)


def test_every_rule_reports_a_message(co):
    for result in v.validate(co, today=TODAY):
        assert result.rule and result.message
        assert result.outcome in Outcome


# --- required fields ----------


def test_missing_required_field_fails(co):
    result = v.required_fields_present(change(co, change_amount=None, contractor=""))
    assert result.failed
    assert set(result.fields) == {"change_amount", "contractor"}


# --- contract math ----------


def test_contract_math_passes(co):
    assert v.contract_math(co).outcome is Outcome.PASS


def test_contract_math_catches_ocr_digit_error(co):
    # OCR read $2,533,183.56 as $2,533,188.56
    result = v.contract_math(change(co, revised_contract_sum=Decimal("2533188.56")))
    assert result.failed
    assert "revised_contract_sum" in result.fields
    assert "2,533,183.56" in result.message  # shows what the sum should be


def test_contract_math_tolerates_rounding(co):
    result = v.contract_math(change(co, revised_contract_sum=Decimal("2533184.00")))
    assert result.outcome is Outcome.PASS


def test_contract_math_not_applicable_when_sum_missing(co):
    result = v.contract_math(change(co, original_contract_sum=None))
    assert result.outcome is Outcome.NOT_APPLICABLE
    assert "original_contract_sum" in result.message


# --- line items total ----------


def test_line_items_plus_markups_equal_change_amount(co):
    assert v.line_items_total(co).outcome is Outcome.PASS


def test_missed_markups_fail_the_total(co):
    result = v.line_items_total(change(co, markups=[]))
    assert result.failed
    assert "9,640.00" in result.message and "10,763.06" in result.message


def test_line_item_lost_under_stamp_fails_subtotal(co):
    # the RECEIVED stamp hid line item 1 on the real scan
    result = v.line_items_total(change(co, line_items=co.line_items[1:]))
    assert result.failed
    assert set(result.fields) == {"line_items", "subtotal"}


def test_line_items_total_not_applicable_without_items(co):
    assert v.line_items_total(change(co, line_items=[])).outcome is Outcome.NOT_APPLICABLE


# --- per-item math ----------------------------------------------------------------------------


def test_line_item_math_passes_including_credit(co):
    assert v.line_item_math(co).outcome is Outcome.PASS  # 180 x 12.50 = (2,250.00)


def test_line_item_math_flags_the_bad_item(co):
    items = list(co.line_items)
    items[1] = items[1].model_copy(update={"amount": Decimal("1400.00")})  # 4 x 325 = 1,300
    result = v.line_item_math(change(co, line_items=items))
    assert result.failed
    assert result.fields == ("line_items[1]",)


def test_line_item_math_skips_lump_sums():
    co = ChangeOrderDraft(line_items=[LineItem(description="Relocate FD-3", amount=Decimal(3450))])
    assert v.line_item_math(co).outcome is Outcome.NOT_APPLICABLE


# --- markups ----------


def test_bond_on_marked_up_total_passes(co):
    # bond 1.5% is charged on 9,640.00 + 964.00 = 10,604.00 -> 159.06
    assert v.markup_math(co).outcome is Outcome.PASS


def test_wrong_markup_amount_fails(co):
    markups = [co.markups[0].model_copy(update={"amount": Decimal("1064.00")}), co.markups[1]]
    result = v.markup_math(change(co, markups=markups))
    assert result.failed
    assert result.fields == ("markups[0]",)


def test_markup_without_rate_not_applicable(co):
    markups = [Markup(description="Bond", amount=Decimal("159.06"))]
    assert v.markup_math(change(co, markups=markups)).outcome is Outcome.NOT_APPLICABLE


# --- signs ----------


def test_positive_credit_fails(co):
    items = list(co.line_items)
    items[3] = items[3].model_copy(update={"amount": Decimal("2250.00")})  # lost the parentheses
    result = v.credit_signs(change(co, line_items=items))
    assert result.failed
    assert result.fields == ("line_items[3]",)


def test_add_and_replace_item_is_not_a_credit():
    item = LineItem(description="Delete VCT and replace with LVT", amount=Decimal(900))
    assert v.credit_signs(ChangeOrderDraft(line_items=[item])).outcome is Outcome.NOT_APPLICABLE


# --- dates ----------


def test_future_issue_date_fails(co):
    assert v.date_sanity(change(co, date_issued=dt.date(2027, 1, 1)), today=TODAY).failed


def test_ancient_issue_date_fails(co):
    assert v.date_sanity(change(co, date_issued=dt.date(1926, 9, 18)), today=TODAY).failed


def test_signature_before_issue_date_fails(co):
    # fax header "SEP-19" misread as the issue date would put the contractor's 9/18 signing first
    result = v.date_sanity(change(co, date_issued=dt.date(2026, 9, 19)), today=TODAY)
    assert result.failed
    assert "signatures[1]" in result.fields


# --- CO number, status, currency, schedule ----------


@pytest.mark.parametrize("number", ["CO-007", "007", "PCO #7", "CO 12", "CO-012A"])
def test_co_number_accepted(co, number):
    assert v.co_number_format(change(co, change_order_number=number)).outcome is Outcome.PASS


@pytest.mark.parametrize("number", ["Riverside Medical", "CO-2026-09-18-001", "#"])
def test_co_number_rejected(co, number):
    assert v.co_number_format(change(co, change_order_number=number)).failed


def test_approved_with_unsigned_owner_fails(co):
    result = v.status_consistency(change(co, status=Status.APPROVED))
    assert result.failed
    assert "owner" in result.message


def test_pending_with_all_signed_fails(co):
    signed = [s.model_copy(update={"signed": True}) for s in co.signatures]
    assert v.status_consistency(change(co, signatures=signed)).failed


def test_approved_without_any_signature_fails():
    co = ChangeOrderDraft(status=Status.APPROVED, signatures=[])
    assert v.status_consistency(co).failed


def test_pending_with_owner_missing_passes(co):
    assert v.status_consistency(co).outcome is Outcome.PASS


@pytest.mark.parametrize(("code", "ok"), [("USD", True), ("$", False), ("usd", False)])
def test_currency_code(co, code, ok):
    assert v.currency_code(change(co, currency=code)).failed is not ok


def test_schedule_days(co):
    assert v.schedule_days_plausible(co).outcome is Outcome.PASS
    assert v.schedule_days_plausible(change(co, schedule_impact_days=60000)).failed


# --- helpers ----------


def test_failed_fields_collapses_indexes(co):
    broken = change(co, markups=[], revised_contract_sum=Decimal(1))
    fields = v.failed_fields(v.validate(broken, today=TODAY))
    assert {"line_items", "change_amount", "revised_contract_sum"} <= fields
    assert not any("[" in f for f in fields)


def test_rules_are_pure(co):
    before = co.model_dump()
    v.validate(co, today=TODAY)
    v.validate(co, today=TODAY)
    assert co.model_dump() == before


def test_empty_draft_only_fails_required_fields():
    results = v.validate(ChangeOrderDraft(), today=TODAY)
    assert [r.rule for r in results if r.failed] == ["required_fields"]
    assert sum(r.outcome is Outcome.NOT_APPLICABLE for r in results) == len(v.RULES) - 1
