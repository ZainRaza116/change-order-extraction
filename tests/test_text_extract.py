import json
from decimal import Decimal
from pathlib import Path

import pytest

from co_extract.normalize import parse_money
from co_extract.text_extract import PageText, extract_text

ROOT = Path(__file__).parent.parent
DIGITAL = ROOT / "samples" / "co_002_digital.pdf"
FIXTURE = Path(__file__).parent / "fixtures" / "digital_text.json"


@pytest.fixture(scope="module")
def pages() -> list[PageText]:
    return extract_text(DIGITAL)


def test_every_page_is_digital_text(pages):
    assert len(pages) == 2
    assert all(p.method == "text" for p in pages)
    assert all(p.ocr_confidence is None for p in pages)
    assert all(p.char_count == len(p.text) for p in pages)


@pytest.mark.parametrize(
    "needle", ["CO", "10,763.06", "2,533,183.56", "Riverside Medical Office Building"]
)
def test_key_values_present(pages, needle):
    assert needle in pages[0].text


def test_line_item_table(pages):
    tables = [t for p in pages for t in p.tables]
    assert len(tables) == 1
    header, *rows = tables[0]
    assert header[-1] == "Amount"
    assert len(rows) == 4
    amounts = [parse_money(r[-1]) for r in rows]
    assert amounts == [
        Decimal("9120.00"),
        Decimal("3450.00"),
        Decimal("-2250.00"),
        Decimal("443.06"),
    ]
    assert sum(amounts) == Decimal("10763.06")


def test_txt_input_is_one_page():
    (page,) = extract_text(ROOT / "samples" / "co_001_email.txt")
    assert page.method == "text"
    assert "Six (6) calendar days" in page.text


def test_fixture_matches_current_output(pages):
    saved = [PageText.model_validate(p) for p in json.loads(FIXTURE.read_text())]
    assert saved == pages
