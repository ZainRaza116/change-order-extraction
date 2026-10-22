from datetime import date
from decimal import Decimal

import pytest

from co_extract.normalize import parse_date, parse_days, parse_money


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("($2,250.00)", Decimal("-2250.00")),
        ("$2,533,183.56", Decimal("2533183.56")),
        ("-$1,200", Decimal("-1200")),
        ("TBD", None),
        ("", None),
    ],
)
def test_parse_money(text, expected):
    assert parse_money(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sept. 18, 2026", date(2026, 9, 18)),
        ("02/03/26", date(2026, 2, 3)),  # US convention: MM/DD/YY
        ("9/19/2026", date(2026, 9, 19)),
        ("18 September 2026", date(2026, 9, 18)),
        ("2026-09-18", date(2026, 9, 18)),
    ],
)
def test_parse_date_to_iso(text, expected):
    result = parse_date(text)
    assert result == expected
    assert result.isoformat() == expected.isoformat()


@pytest.mark.parametrize("text", ["f/u Mon", "next week", "TBD", "13/45/2026"])
def test_parse_date_refuses_to_guess(text):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Six (6) calendar days", 6),
        ("3 working days", 3),
        ("approx. one week", 7),
        ("f/u Monday, no time impact", None),
    ],
)
def test_parse_days(text, expected):
    assert parse_days(text) == expected
