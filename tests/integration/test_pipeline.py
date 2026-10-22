import datetime as dt
import json

import pytest

from co_extract.llm import CachedProvider, OpenAIProvider
from co_extract.pipeline import build_provider, run_pipeline
from co_extract.result import ExtractionResult
from tests.factories import SAMPLES, ScriptedProvider, co002_output, f, make_settings

DIGITAL = SAMPLES / "co_002_digital.pdf"
SCANNED = SAMPLES / "co_003_scanned.pdf"
TODAY = dt.date(2026, 10, 21)
GOOD = json.dumps(co002_output())


def bad_total():
    out = co002_output()
    out["revised_contract_sum"] = f(
        "$2,533,188.56", "The new Contract Sum including this Change Order will be $2,533,188.56"
    )
    return json.dumps(out)


# --- pipeline -----------------------------------------------------------------------------


def test_digital_document_end_to_end(tmp_path):
    result = run_pipeline(DIGITAL, ScriptedProvider(GOOD), make_settings(tmp_path), today=TODAY)
    assert result.status == "ok"
    assert not result.needs_review
    assert result.overall_confidence > 0.95
    assert result.fields["change_amount"].value == "10763.06"
    assert result.fields["change_amount"].raw == "$10,763.06"
    assert result.fields["date_issued"].value == "2026-10-06"
    assert result.fields["schedule_impact_days"].value == 6
    assert [i.value["amount"] for i in result.line_items] == [
        "9120.00",
        "3450.00",
        "-2250.00",
        "443.06",
    ]
    assert [p.method for p in result.pages] == ["text", "text"]
    assert all(r.outcome != "fail" for r in result.validation)


def test_result_is_valid_json_and_round_trips(tmp_path):
    result = run_pipeline(DIGITAL, ScriptedProvider(GOOD), make_settings(tmp_path), today=TODAY)
    again = ExtractionResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert len(result.file_sha256) == 64


def test_cost_and_usage_are_reported(tmp_path):
    result = run_pipeline(DIGITAL, ScriptedProvider(GOOD), make_settings(tmp_path), today=TODAY)
    assert (result.run.input_tokens, result.run.output_tokens) == (4000, 2500)
    assert result.run.cost_usd == pytest.approx((4000 * 2 + 2500 * 10) / 1e6)
    assert result.run.model == "claude-sonnet-5-5"


def test_scanned_document_goes_through_ocr(tmp_path):
    result = run_pipeline(SCANNED, ScriptedProvider(GOOD), make_settings(tmp_path), today=TODAY)
    assert [p.method for p in result.pages] == ["ocr", "ocr"]
    assert result.overall_confidence < 1.0  # OCR source quality < digital


def test_inconsistent_numbers_are_flagged(tmp_path):
    result = run_pipeline(
        DIGITAL, ScriptedProvider(bad_total()), make_settings(tmp_path), today=TODAY
    )
    assert result.needs_review
    assert any(v.rule == "contract_math" and v.outcome == "fail" for v in result.validation)
    assert result.fields["revised_contract_sum"].confidence <= 0.6


def test_llm_failure_returns_error_result_instead_of_raising(tmp_path):
    result = run_pipeline(
        DIGITAL, ScriptedProvider("nope", "still nope"), make_settings(tmp_path), today=TODAY
    )
    assert result.status == "error"
    assert "ExtractionFailed" in result.error
    assert result.needs_review


def test_escalation_keeps_the_better_result(tmp_path):
    provider = ScriptedProvider(bad_total(), GOOD)  # cheap model errs, strong model is right
    result = run_pipeline(DIGITAL, provider, make_settings(tmp_path), escalate=True, today=TODAY)
    assert [r.model for r in provider.requests] == ["claude-sonnet-5-5", "claude-opus-5-5"]
    assert result.run.escalated and result.run.model == "claude-opus-5-5"
    assert not result.needs_review
    sonnet, opus = (4000 * 2 + 2500 * 10) / 1e6, (4000 * 4 + 2500 * 20) / 1e6
    assert result.run.cost_usd == pytest.approx(sonnet + opus)  # both calls are paid for


def test_no_escalation_when_document_is_clean(tmp_path):
    provider = ScriptedProvider(GOOD)
    result = run_pipeline(DIGITAL, provider, make_settings(tmp_path), escalate=True, today=TODAY)
    assert len(provider.requests) == 1 and not result.run.escalated


def test_second_run_is_served_from_the_cache(tmp_path):
    inner = ScriptedProvider(GOOD)
    cached = CachedProvider(inner, tmp_path / "cassettes")
    run_pipeline(DIGITAL, cached, make_settings(tmp_path), today=TODAY)
    again = run_pipeline(DIGITAL, cached, make_settings(tmp_path), today=TODAY)
    assert len(inner.requests) == 1 and again.run.cached


def test_openai_provider_selected_from_settings(tmp_path):
    provider = build_provider(make_settings(tmp_path, provider="openai", model="gpt-4o-mini"))
    assert isinstance(provider.inner, OpenAIProvider)


def test_unreadable_item_does_not_hide_the_others(tmp_path):
    out = co002_output()
    out["line_items"][0]["amount"] = None  # e.g. hidden under a stamp
    result = run_pipeline(
        DIGITAL, ScriptedProvider(json.dumps(out)), make_settings(tmp_path), today=TODAY
    )
    amounts = [i.value["amount"] for i in result.line_items]
    assert amounts == [None, "3450.00", "-2250.00", "443.06"]
    assert result.line_items[0].confidence == 0.0
    assert result.needs_review
