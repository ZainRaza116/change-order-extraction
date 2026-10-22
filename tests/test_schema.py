import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from co_extract.schema import ChangeOrderRecord

EXPECTED_DIR = Path(__file__).parent.parent / "expected"
GROUND_TRUTH = sorted(EXPECTED_DIR.glob("*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_ground_truth_exists():
    assert GROUND_TRUTH, "no ground-truth files in expected/"


@pytest.mark.parametrize("path", GROUND_TRUTH, ids=lambda p: p.name)
def test_ground_truth_validates(path):
    ChangeOrderRecord.model_validate(load(path))


@pytest.mark.parametrize("path", GROUND_TRUTH, ids=lambda p: p.name)
def test_ground_truth_math_is_consistent(path):
    co = ChangeOrderRecord.model_validate(load(path))
    if co.line_items:
        items = sum(i.amount for i in co.line_items)
        if co.subtotal is not None:
            assert items == co.subtotal
        assert items + sum(m.amount for m in co.markups) == co.change_amount
    sums = (co.original_contract_sum, co.previous_changes_total, co.revised_contract_sum)
    if None not in sums:
        original, previous, revised = sums
        assert original + previous + co.change_amount == revised


@pytest.fixture
def valid_record() -> dict:
    return load(GROUND_TRUTH[0])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("change_amount", "a lot"),
        ("date_issued", "next Monday"),
        ("schedule_impact_days", "six-ish"),
        ("status", "maybe"),
        ("change_order_number", 7),
    ],
)
def test_wrong_type_is_rejected(valid_record, field, bad_value):
    valid_record[field] = bad_value
    with pytest.raises(ValidationError):
        ChangeOrderRecord.model_validate(valid_record)


@pytest.mark.parametrize("field", ["change_order_number", "date_issued", "change_amount"])
def test_missing_required_field_is_rejected(valid_record, field):
    del valid_record[field]
    with pytest.raises(ValidationError):
        ChangeOrderRecord.model_validate(valid_record)


def test_unknown_field_is_rejected(valid_record):
    valid_record["chnage_amount"] = "100.00"
    with pytest.raises(ValidationError):
        ChangeOrderRecord.model_validate(valid_record)
