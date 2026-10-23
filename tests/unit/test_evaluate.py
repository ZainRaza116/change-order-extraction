import json

import pytest

from co_extract import evaluate as ev


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("007", "CO-007", True),
        ("PCO #7", "CO-007", True),
        ("CO-011, REVISION 1", "CO-011", True),
        ("Change Order #3", "CO-003", True),
        ("CO-07A", "CO-07", False),
        ("CO-012", "CO-021", False),
    ],
)
def test_co_number(a, b, same):
    assert ev.same_co_number(a, b) is same


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Ridgeline Builders, Inc.", "Ridgeline Builders Inc", True),
        ("Harmon Health Partners, LLC", "harmon health partners", True),
        ("Diane Calloway, AIA", "Diane Calloway", True),
        ("Ridgeline Construction Group", "Ridgeline Builders, Inc.", False),
    ],
)
def test_names(a, b, same):
    assert ev.same_name(a, b) is same


def test_project_number_is_strict():
    assert not ev.COMPARATORS["project_number"]("24 18", "24-118")
    assert ev.COMPARATORS["project_number"](" 24-118", "24-118")


def test_text_tolerates_wording_but_not_a_different_sentence():
    truth = "Owner-requested design revision."
    assert ev.same_text("Owner requested design revision", truth)
    assert not ev.same_text(
        "Per Owner request and Architect's Supplemental Instruction ASI-09, add structural support",
        truth,
    )


def test_money_to_the_cent():
    assert ev.same_money("10763.06", "10763.060")
    assert not ev.same_money("10763.06", "10763.16")
    assert not ev.same_money("abc", "1")


@pytest.mark.parametrize(
    ("predicted", "expected", "outcome"),
    [
        (None, None, "correct_null"),
        ("9640.00", None, "hallucinated"),
        (None, "9640.00", "missed"),
        ("9640.00", "9640.00", "correct"),
        ("9640.00", "9650.00", "wrong"),
    ],
)
def test_field_outcomes(predicted, expected, outcome):
    assert ev.check_field("d", "subtotal", predicted, expected, 0.9).outcome == outcome


def test_item_amounts_matched_as_multiset():
    result = {
        "line_items": [
            {"value": {"amount": "1300.00"}, "confidence": 0.9},
            {"value": {"amount": "1300.00"}, "confidence": 0.9},  # duplicate: only one counts
            {"value": {"amount": None}, "confidence": 0.0},
        ]
    }
    truth = {"line_items": [{"amount": "1300.00"}, {"amount": "8550.00"}]}
    check = ev.check_items("d", "line_items", result, truth)
    assert (check.expected, check.predicted, check.matched) == (2, 3, 1)
    assert [ok for _, ok in check.entries] == [True, False, False]


def write(folder, name, payload):
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(payload))


def result_doc(fields, needs_review=False, status="ok"):
    return {
        "status": status,
        "needs_review": needs_review,
        "fields": {k: {"value": v, "confidence": c} for k, (v, c) in fields.items()},
        "line_items": [],
        "markups": [],
        "run": {"model": "gpt-4o-mini", "cost_usd": 0.002, "llm_seconds": 10.0},
    }


def test_report_end_to_end(tmp_path):
    out, exp = tmp_path / "outputs", tmp_path / "expected"
    write(exp, "a", {"change_amount": "100.00", "project_number": "24-118", "subtotal": None})
    write(
        out,
        "a",
        result_doc(
            {
                "change_amount": ("100.00", 0.95),  # right, confident
                "project_number": ("24 18", 0.97),  # wrong, confident (silent error)
                "subtotal": ("100.00", 0.40),  # hallucinated, flagged
            }
        ),
    )
    report = ev.evaluate(out, exp)
    outcomes = {c.field: c.outcome for c in report.fields}
    assert outcomes["change_amount"] == "correct"
    assert outcomes["project_number"] == "wrong"
    assert outcomes["subtotal"] == "hallucinated"
    assert report.hallucinations()[0] == 1
    assert report.errors_flagged() == (1, 2)  # subtotal flagged, project_number not

    calib = {r["bucket"]: r for r in report.calibration()}
    assert calib["0.0-0.5"]["n"] == 1 and calib["0.0-0.5"]["accuracy"] == 0.0
    assert calib["0.8-1.0"]["n"] == 2 and calib["0.8-1.0"]["accuracy"] == 0.5

    md = ev.render_markdown(report)
    assert "| Hallucinated values" in md and "project_number" in md
    assert "$0.0020" in md and "10.0 s" in md
    json.dumps(report.to_dict(), default=str)  # serialisable


def test_documents_without_ground_truth_are_skipped(tmp_path):
    out, exp = tmp_path / "outputs", tmp_path / "expected"
    write(exp, "a", {"change_amount": "1"})
    write(out, "b", result_doc({"change_amount": ("1", 1.0)}))
    assert ev.evaluate(out, exp).documents == []


def test_free_text_is_kept_out_of_the_headline(tmp_path):
    out, exp = tmp_path / "outputs", tmp_path / "expected"
    write(exp, "a", {"change_amount": "1.00", "description": "Paraphrased scope summary"})
    got = {"change_amount": ("1.00", 0.95), "description": ("Verbatim scope from page", 0.95)}
    write(out, "a", result_doc(got))
    report = ev.evaluate(out, exp)
    assert report.errors_flagged() == (0, 0)  # the description mismatch is not an "error"
    assert [(d, f) for d, f, _ in report.free_text()] == [("a", "description")]
    assert "too few to judge" in ev.render_markdown(report)


def test_document_flags(tmp_path):
    out, exp = tmp_path / "outputs", tmp_path / "expected"
    for name, got, review in (
        ("bad", "2.00", True),
        ("missed", "2.00", False),
        ("ok", "1.00", False),
    ):
        write(exp, name, {"change_amount": "1.00"})
        write(out, name, result_doc({"change_amount": (got, 0.9)}, needs_review=review))
    # 2 documents with an error, 1 of them flagged; 1 clean document, passed
    assert ev.evaluate(out, exp).document_flags() == (1, 2, 1, 1)
