"""Deterministic parsers for money, dates, day counts, percentages and currencies.

Every parser returns None when it can't read the input rather than guessing.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

_MONEY_CHARS = re.compile(r"[^\d.\-]")


def parse_money(text: str) -> Decimal | None:
    """'($2,250.00)' -> Decimal('-2250.00'); parentheses or a minus sign mean negative."""
    s = text.strip()
    negative = (s.startswith("(") and s.endswith(")")) or "-" in s
    digits = _MONEY_CHARS.sub("", s).replace("-", "")
    if not re.fullmatch(r"\d+(\.\d+)?", digits):
        return None
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH_NAME = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b")
_MONTH_FIRST = re.compile(rf"\b{_MONTH_NAME}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH_NAME},?\s+(\d{{4}})\b", re.I)


def _make_date(year: int, month: int, day: int) -> date | None:
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    """Parse explicit dates only. Numeric dates are read US-style (MM/DD/YY).

    Relative or partial dates such as 'f/u Mon' or 'next week' return None.
    """
    if m := _ISO.search(text):
        return _make_date(int(m[1]), int(m[2]), int(m[3]))
    if m := _NUMERIC.search(text):
        return _make_date(int(m[3]), int(m[1]), int(m[2]))
    if m := _MONTH_FIRST.search(text):
        return _make_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))
    if m := _DAY_FIRST.search(text):
        return _make_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
    return None


def find_dates(text: str) -> list[tuple[int, date]]:
    """Every explicit date in `text`, as (start offset, date), in reading order."""
    found: dict[int, date] = {}
    patterns = (
        (_ISO, lambda m: (int(m[1]), int(m[2]), int(m[3]))),
        (_NUMERIC, lambda m: (int(m[3]), int(m[1]), int(m[2]))),
        (_MONTH_FIRST, lambda m: (int(m[3]), _MONTHS[m[1].lower()], int(m[2]))),
        (_DAY_FIRST, lambda m: (int(m[3]), _MONTHS[m[2].lower()], int(m[1]))),
    )
    for pattern, parts in patterns:
        for m in pattern.finditer(text):
            if (d := _make_date(*parts(m))) and not any(abs(m.start() - s) < 3 for s in found):
                found[m.start()] = d
    return sorted(found.items())


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
}  # fmt: skip
# "<count> [(<digits>)] [calendar|working] day(s)/week(s)"
_DAYS = re.compile(
    r"(\d+|[a-z]+)\s*(?:\(\s*(\d+)\s*\)\s*)?(?:calendar\s+|working\s+)?(day|week)s?\b", re.I
)


def parse_days(text: str) -> int | None:
    """'Six (6) calendar days' -> 6, 'one week' -> 7. Returns None if no count is stated."""
    for m in _DAYS.finditer(text):
        word, paren, unit = m[1].lower(), m[2], m[3].lower()
        if paren:
            count = int(paren)
        elif word.isdigit():
            count = int(word)
        elif word in _NUMBER_WORDS:
            count = _NUMBER_WORDS[word]
        else:
            continue  # e.g. "Monday" matched as "Mon" + "day"
        return count * 7 if unit == "week" else count
    return None


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def parse_percent(text: str) -> Decimal | None:
    """'Overhead & Profit @ 10%' -> Decimal('10'). Requires a % sign."""
    m = _PERCENT.search(text)
    return Decimal(m[1]) if m else None


_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "C$": "CAD", "A$": "AUD"}
_CURRENCY_CODE = re.compile(r"\b([A-Z]{3})\b")


def parse_currency(text: str) -> str | None:
    """'USD', 'usd', '$' or 'US Dollars' -> 'USD'. None if no currency can be read."""
    s = text.strip()
    if s in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[s]
    if re.search(r"\bdollars?\b", s, re.I) and re.search(r"\bU\.?S\.?\b", s, re.I):
        return "USD"
    m = _CURRENCY_CODE.search(s.upper())
    return m[1] if m else None
