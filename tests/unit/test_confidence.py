import datetime as dt
import json
from decimal import Decimal

import pytest

from co_extract import confidence as c
from co_extract.assemble import assemble
from co_extract.ingest import PageText
from co_extract.llm import LlmExtraction
from co_extract.schema import ChangeOrderRecord
from co_extract.validate import validate
from tests.factories import (
    EXPECTED,
    NULL,
    co002_extraction,
    co002_output,
    f,
    item,
    minimal_output,
    pages,
)

TODAY = dt.date(2026, 10, 21)
DIGITAL = pages("digital_text.json")
SCANNED = pages("scanned_text.json")  # same document, OCR'd at ~0.89 / 0.92 confidence


def run(extraction, source=DIGITAL):
    assembled = assemble(extraction)
    rules = validate(assembled.draft, today=TODAY)
    return assembled, rules, c.score_document(extraction, assembled, source, rules)


# --- assembly -----------------------------------------------------------------------------


def test_assembled_draft_matches_ground_truth_numbers():
    draft = assemble(co002_extraction()).draft
    truth = ChangeOrderRecord.model_validate(
        json.loads((EXPECTED / "co_002_digital.json").read_text())
    )
    for name in ("change_amount", "original_contract_sum", "revised_contract_sum",
                 "date_issued", "schedule_impact_days", "currency", "status"):  # fmt: skip
        assert getattr(draft, name) == getattr(truth, name), name
    assert [i.amount for i in draft.line_items] == [i.amount for i in truth.line_items]
    assert [s.date for s in draft.signatures] == [s.date for s in truth.signatures]


def test_unparseable_value_becomes_none_and_is_recorded():
    a = assemble(co002_extraction(date_issued=f("next Monday", "DATE: next Monday")))
    assert a.draft.date_issued is None
    assert a.parse_errors == {"date_issued": "next Monday"}


def no_currency_mark(**overrides):
    # amounts written without "$", and no currency field: nothing to derive from
    money = ("change_amount", "original_contract_sum", "revised_contract_sum")
    out = co002_output()
    plain = {
        k: f(out[k]["value"].replace("$", ""), out[k]["evidence"].replace("$", ""), out[k]["page"])
        for k in money
    }
    return co002_extraction(currency=NULL, **plain, **overrides)


def test_currency_derived_from_the_amounts_symbol():
    # s07: "Amount: $1,640.00" but the model left currency empty
    a = assemble(co002_extraction(currency=NULL))
    assert a.draft.currency == "USD" and a.defaulted == set()
    assert "currency" in a.derived and a.derived["currency"].evidence


def test_currency_defaults_to_usd_only_without_any_mark():
    a = assemble(no_currency_mark())
    assert a.draft.currency == "USD" and a.defaulted == {"currency"}


# --- signals ------------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert sum(c.WEIGHTS.values()) == pytest.approx(1.0)


def test_evidence_match_tolerates_whitespace_and_case():
    source = "CHANGE ORDER NO.:     CO-012\nDATE:   October 6, 2026"
    assert c.evidence_match("change order no.: CO-012", source) == 1.0
    assert c.evidence_match("CHANGE ORDER NO.: CO-099", source) < 1.0
    assert c.evidence_match("Total this Change Order: $99,999.99", source) < c.EVIDENCE_THRESHOLD
    assert c.evidence_match(None, source) == 0.0


@pytest.mark.parametrize(
    ("raw", "evidence", "expected"),
    [
        ("$10,763.06", "Total this Change Order: $10,763.06", 1.0),
        ("10763.06", "Total this Change Order: $10,763.06", 0.8),  # digits only
        ("$11,240.00", "Total this Change Order: $10,763.06", 0.0),
        ("$10,763.06", None, 0.0),
    ],
)
def test_value_in_evidence(raw, evidence, expected):
    assert c.value_in_evidence(raw, evidence) == expected


def test_source_quality_per_method():
    text = PageText(page=1, text="x", method="text", char_count=1)
    ocr = PageText(page=2, text="x", method="ocr", char_count=1, ocr_confidence=0.82)
    assert c.source_quality([text, ocr], 1) == 1.0
    assert c.source_quality([text, ocr], 2) == 0.82
    assert c.source_quality([text, ocr], None) == 0.82  # unknown page: worst page


def test_ambiguity_counts_distinct_competitors():
    assert c.ambiguity([], "$10,763.06") == 1.0
    assert c.ambiguity(["$11,240.00"], "$10,763.06") == 0.5
    assert c.ambiguity(["$10,763.06"], "$10,763.06") == 1.0  # repeating the value isn't a rival


# --- document scores ----------------------------------------------------------------------


def test_clean_digital_document_scores_high():
    _, rules, score = run(co002_extraction())
    assert not any(r.failed for r in rules)
    assert score.overall > 0.95
    assert not score.needs_review
    assert all(score.fields[n].confidence > 0.9 for n in ("change_amount", "date_issued"))
    assert len(score.items["line_items"]) == 4
    assert all(s.confidence > 0.9 for s in score.items["line_items"])


def test_same_document_from_ocr_scores_lower_but_passes():
    digital = run(co002_extraction())[2]
    scanned = run(co002_extraction(), SCANNED)[2]
    assert scanned.overall < digital.overall
    assert scanned.overall > 0.9
    assert scanned.fields["change_amount"].signals["source_quality"] == pytest.approx(0.8866)


def test_hallucinated_value_is_capped_and_flagged():
    fake = f("Ridgeline Construction Group", "CONTRACTOR: Ridgeline Construction Group")
    _, _, score = run(co002_extraction(contractor=fake))
    contractor = score.fields["contractor"]
    assert contractor.confidence <= c.CAP_EVIDENCE_NOT_FOUND
    assert "evidence not found in source" in contractor.caps
    assert score.needs_review


def test_value_missing_from_its_own_evidence_scores_lower():
    # real quote, but the value was swapped for the superseded proposal figure
    wrong = f("$11,240.00", "Total this Change Order: $10,763.06")
    _, _, score = run(co002_extraction(change_amount=wrong))
    assert score.fields["change_amount"].signals["value_in_evidence"] == 0.0
    assert score.fields["change_amount"].confidence <= c.CAP_RULE_FAILED  # math fails too
    assert score.needs_review


def test_competing_alternatives_lower_confidence():
    base = run(co002_extraction())[2].fields["change_amount"].confidence
    contested = f("$10,763.06", "Total this Change Order: $10,763.06", alternatives=["$11,240.00"])
    lowered = run(co002_extraction(change_amount=contested))[2].fields["change_amount"]
    assert lowered.signals["ambiguity"] == 0.5
    assert lowered.confidence == pytest.approx(base - 0.05, abs=0.001)


def test_failed_rule_caps_every_field_involved():
    bad = f(
        "$2,533,188.56", "The new Contract Sum including this Change Order will be $2,533,188.56"
    )
    _, rules, score = run(co002_extraction(revised_contract_sum=bad))
    assert any(r.rule == "contract_math" and r.failed for r in rules)
    for name in ("original_contract_sum", "previous_changes_total", "change_amount"):
        assert score.fields[name].confidence <= c.CAP_RULE_FAILED
        assert "validation rule failed" in score.fields[name].caps
    assert score.needs_review
    assert any("contract_math" in reason for reason in score.review_reasons)


def test_missing_required_field_scores_zero_and_needs_review():
    _, _, score = run(co002_extraction(change_order_number=NULL))
    assert score.fields["change_order_number"].confidence == 0.0
    assert score.needs_review
    assert score.overall < run(co002_extraction())[2].overall


def test_missing_optional_field_does_not_force_review():
    _, _, score = run(co002_extraction(owner=NULL))
    assert score.fields["owner"].confidence == 0.0
    assert not score.needs_review


def test_derived_currency_is_confident_and_labelled():
    _, _, score = run(co002_extraction(currency=NULL))
    currency = score.fields["currency"]
    assert 0.7 <= currency.confidence <= 0.9
    assert any(cap.startswith("derived: currency from '$'") for cap in currency.caps)
    assert not score.needs_review


def test_defaulted_currency_gets_half_confidence():
    _, _, score = run(no_currency_mark())
    assert score.fields["currency"].confidence == c.DEFAULTED_CONFIDENCE
    assert score.needs_review  # currency is required and 0.5 < 0.7


@pytest.mark.parametrize(
    "wording",
    ["No time extension is requested.", "Contract Time: Unchanged (0 days)",
     "There is no change to the contract time."],
)  # fmt: skip
def test_no_time_change_wording_gives_zero_days(wording):
    page = PageText(page=1, text=f"Scope as discussed.\n{wording}\n", method="text",
                    char_count=40)  # fmt: skip
    a = assemble(co002_extraction(schedule_impact_days=NULL), [page])
    assert a.draft.schedule_impact_days == 0
    assert a.derived["schedule_impact_days"].evidence == wording


def test_ten_days_is_not_read_as_zero():
    page = PageText(page=1, text="Time: add 10 days.", method="text", char_count=18)
    assert (
        assemble(co002_extraction(schedule_impact_days=NULL), [page]).draft.schedule_impact_days
        is None
    )


def test_unparseable_value_scores_low_and_is_reported():
    _, _, score = run(co002_extraction(date_issued=f("next Monday", "DATE: next Monday")))
    assert score.fields["date_issued"].signals["validity"] == 0.0
    assert any("could not parse date_issued" in r for r in score.review_reasons)


def test_mapped_enum_field_uses_evidence_match():
    status = run(co002_extraction())[2].fields["status"]
    assert status.signals["value_in_evidence"] == status.signals["evidence_match"]


def test_low_ocr_confidence_alone_can_trigger_review():
    murky = [p.model_copy(update={"ocr_confidence": 0.2}) for p in SCANNED]
    score = run(co002_extraction(), murky)[2]
    assert all(score.fields[n].signals["source_quality"] == 0.2 for n in ("change_amount",))
    assert score.overall < run(co002_extraction(), SCANNED)[2].overall


def test_scores_are_deterministic():
    a, b = run(co002_extraction())[2], run(co002_extraction())[2]
    assert a == b


def test_decimal_values_survive_assembly():
    draft = assemble(co002_extraction()).draft
    assert draft.line_items[2].amount == Decimal("-2250.00")


def test_unparseable_value_is_capped():
    # live run: OCR read "$2,533,183.56" as "$2,533. 183.56"; the parser refuses to guess,
    # so the typed value is empty and must not keep a high confidence
    garbled = f("$2,533. 183.56", "The new Contract Sum including this Change Order will be")
    _, _, score = run(co002_extraction(revised_contract_sum=garbled))
    field = score.fields["revised_contract_sum"]
    assert field.confidence <= c.CAP_UNPARSEABLE
    assert "value could not be parsed" in field.caps
    assert score.needs_review


def test_quote_stitched_across_columns_is_found():
    # s01 letterhead: "CHANGE ORDER" and "No. 007" sit in one column, other text between
    source = "RIDGELINE BUILDERS, INC.      CHANGE ORDER\nGeneral Contractors      No. 007"
    assert c.evidence_match("CHANGE ORDER No. 007", source) == 1.0


def test_stitching_does_not_rescue_an_invented_number():
    source = "RIDGELINE BUILDERS, INC.      CHANGE ORDER\nGeneral Contractors      No. 007"
    assert c.evidence_match("CHANGE ORDER No. 008", source) < c.EVIDENCE_THRESHOLD


def test_quote_built_from_scattered_words_is_not_found():
    # every word of the fabricated quote exists, but far apart: top of page vs signature block
    filler = " ".join(f"word{i}" for i in range(80))
    source = f"CONTRACTOR Ridgeline Builders Inc. {filler} revised contract sum $2,533,183.56"
    fabricated = "CONTRACTOR Ridgeline revised contract sum $2,533,183.56"
    assert c.stitched_match(fabricated, source) == 0.0
    assert c.evidence_match(fabricated, source) < c.EVIDENCE_THRESHOLD


def test_smallest_window():
    assert c._smallest_window(["a", "x", "b", "a", "b"], {"a", "b"}) == 2
    assert c._smallest_window(["a", "x"], {"a", "b"}) is None


# --- competing issue dates (evidence proves a date exists, not that it is the right one) --

EMAIL = """Date: Mon, 5 Oct 2026 09:14:22 -0400
Subject: RE: CO #9 - roof crickets
Approved as revised to $13,750.00.
Sent: Thursday, October 1, 2026 4:02 PM
Following our walk we revised CO #9 (dated 9/28/2026)."""


def test_email_thread_has_competing_issue_dates():
    rivals = c.competing_issue_dates(EMAIL, dt.date(2026, 10, 5), "CO #9", set())
    assert rivals == {dt.date(2026, 10, 1), dt.date(2026, 9, 28)}


def test_other_kinds_of_dates_are_not_rivals():
    text = """DATE: Sept. 18, 2026
Contract Date: 02/03/26
Contractor's original proposal PCO-012 dated 08/29/2026 was submitted at $11,240.00.
This revision supersedes CO-007 dated 06/02/2026, which is void.
By: Marcus T. Reyes  Date: 9/19/2026"""
    signed = {dt.date(2026, 9, 19)}
    assert c.competing_issue_dates(text, dt.date(2026, 9, 18), "007", signed) == set()


def test_competing_date_caps_confidence_and_flags():
    pages = [PageText(page=1, text=EMAIL, method="text", char_count=len(EMAIL))]
    out = minimal_output(
        change_order_number=f("CO #9", "Subject: RE: CO #9 - roof crickets"),
        date_issued=f("Mon, 5 Oct 2026", "Date: Mon, 5 Oct 2026 09:14:22 -0400"),
        change_amount=f("$13,750.00", "Approved as revised to $13,750.00."),
    )
    x = LlmExtraction.model_validate(out)
    a = assemble(x)
    score = c.score_document(x, a, pages, validate(a.draft, today=TODAY))
    date = score.fields["date_issued"]
    assert date.confidence <= c.CAP_AMBIGUOUS_DATE
    assert any("other candidate issue dates" in cap for cap in date.caps)
    assert score.needs_review


def test_unquoted_model_currency_is_replaced_by_the_amounts_symbol():
    # the model wrote "USD" with no quote; that is accepted (no retry) and code decides
    a = assemble(co002_extraction(currency=f("USD", None)))
    assert a.draft.currency == "USD" and "currency" in a.derived


def test_subtotal_under_another_label_is_found():
    # s04 prints the items' sum as "Net of items ($25,280.00)", not "Subtotal"
    text = "1 Credit A ($22,500.00)\n2 Add B $3,420.00\n3 Credit C ($6,200.00)\n" \
           "Net of items ($25,280.00)\nContractor fee @ 8% $273.60\n"  # fmt: skip
    page = PageText(page=1, text=text, method="text", char_count=len(text))
    out = co002_output(
        line_items=[item("Credit A", "1", "LS", "($22,500.00)", "($22,500.00)"),
                    item("Add B", "1", "LS", "$3,420.00", "$3,420.00"),
                    item("Credit C", "1", "LS", "($6,200.00)", "($6,200.00)")],
        markups=[{"description": "Contractor fee @ 8%", "rate_percent": "8%",
                  "amount": "$273.60", "evidence": "Contractor fee @ 8% $273.60", "page": 1}],
    )  # fmt: skip
    a = assemble(LlmExtraction.model_validate(out), [page])
    assert str(a.draft.subtotal) == "-25280.00"
    assert a.derived["subtotal"].evidence == "Net of items ($25,280.00)"


def test_no_subtotal_is_invented_without_markups():
    a = assemble(co002_extraction(), pages("digital_text.json"))
    assert a.draft.subtotal is None
