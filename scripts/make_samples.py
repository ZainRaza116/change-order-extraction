"""Generate the synthetic sample PDFs in samples/. Run: uv run python scripts/make_samples.py"""

import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

LINE_ITEMS = [
    ("1", "Add 2-hr rated GWB at Stair 2", "640", "SF", "$14.25", "$9,120.00"),
    ("2", "Relocate floor drain FD-3", "1", "LS", "$3,450.00", "$3,450.00"),
    ("3", "Credit: delete rubber base, storage rooms", "1", "LS", "($2,250.00)", "($2,250.00)"),
    ("4", "Bond and insurance", "1", "LS", "$443.06", "$443.06"),
]
TABLE_HEADER = ("No.", "Description", "Qty", "Unit", "Unit Cost", "Amount")
TABLE_COLS = [50, 80, 300, 345, 385, 470, 562]


def _text(page, x, y, s, size=10, bold=False):
    page.insert_text((x, y), s, fontsize=size, fontname="hebo" if bold else "helv")


def _table(page, top, rows, row_h=20):
    for r, row in enumerate(rows):
        y0 = top + r * row_h
        for c, cell in enumerate(row):
            x0, x1 = TABLE_COLS[c], TABLE_COLS[c + 1]
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y0 + row_h), width=0.6)
            _text(page, x0 + 4, y0 + 14, cell, size=9, bold=r == 0)
    return top + len(rows) * row_h


def make_digital_g701(path: Path) -> None:
    doc = pymupdf.open()

    p = doc.new_page(width=612, height=792)
    _text(p, 50, 60, "CHANGE ORDER", size=18, bold=True)
    _text(p, 50, 76, "AIA Document G701 - style form", size=8)
    left = [
        ("PROJECT:", "Riverside Medical Office Building"),
        ("", "1450 River Road, Sacramento, CA"),
        ("OWNER:", "Riverside Health Partners LLC"),
        ("CONTRACTOR:", "Ridgeline Builders Inc."),
        ("ARCHITECT:", "Studio Lee Architects"),
    ]
    right = [
        ("CHANGE ORDER NO.:", "CO-012"),
        ("DATE:", "October 6, 2026"),
        ("PROJECT NO.:", "24-207"),
        ("CONTRACT DATE:", "02/03/26"),
    ]
    for i, (label, value) in enumerate(left):
        _text(p, 50, 110 + i * 15, label, size=9, bold=True)
        _text(p, 130, 110 + i * 15, value, size=9)
    for i, (label, value) in enumerate(right):
        _text(p, 360, 110 + i * 15, label, size=9, bold=True)
        _text(p, 470, 110 + i * 15, value, size=9)

    _text(p, 50, 205, "THE CONTRACT IS CHANGED AS FOLLOWS:", size=10, bold=True)
    p.insert_textbox(
        pymupdf.Rect(50, 212, 562, 250),
        "Upgrade fire separation at Stair 2 and relocate floor drain FD-3 per Owner request "
        "following the revised tenant layout (ASI-09). Reason: Owner request.",
        fontsize=9,
        fontname="helv",
    )
    bottom = _table(p, 255, [TABLE_HEADER, *LINE_ITEMS])
    _text(p, 385, bottom + 18, "Total this Change Order:", size=9, bold=True)
    _text(p, 495, bottom + 18, "$10,763.06", size=9, bold=True)

    sums = [
        ("The original Contract Sum was", "$2,450,000.00"),
        ("Net change by previously authorized Change Orders", "$72,420.50"),
        ("The Contract Sum prior to this Change Order was", "$2,522,420.50"),
        ("The Contract Sum will be increased by this Change Order in the amount of", "$10,763.06"),
        ("The new Contract Sum including this Change Order will be", "$2,533,183.56"),
    ]
    for i, (label, value) in enumerate(sums):
        y = bottom + 60 + i * 16
        _text(p, 50, y, label, size=9)
        _text(p, 470, y, value, size=9)
    _text(
        p, 50, bottom + 150, "The Contract Time will be increased by Six (6) calendar days.", size=9
    )
    _text(p, 50, 760, "Page 1 of 2", size=8)

    p = doc.new_page(width=612, height=792)
    _text(p, 50, 60, "CHANGE ORDER CO-012 - SIGNATURES", size=12, bold=True)
    signers = [
        ("ARCHITECT", "Studio Lee Architects", "Marcus Lee", "10/7/2026"),
        ("CONTRACTOR", "Ridgeline Builders Inc.", "Dana Whitfield", "10/6/2026"),
        ("OWNER", "Riverside Health Partners LLC", "Priya Nair", "10/9/2026"),
    ]
    for i, (role, firm, name, signed_on) in enumerate(signers):
        y = 110 + i * 90
        _text(p, 50, y, role, size=9, bold=True)
        _text(p, 50, y + 15, firm, size=9)
        _text(p, 50, y + 45, f"By: /s/ {name}", size=9)
        _text(p, 330, y + 45, f"Date: {signed_on}", size=9)
        p.draw_line((50, y + 50), (300, y + 50), width=0.5)
    _text(p, 50, 760, "Page 2 of 2", size=8)

    doc.set_metadata({"title": "Change Order CO-012", "producer": "co-extract make_samples"})
    doc.save(path, garbage=4, deflate=True)


def make_scanned(src: Path, path: Path, angle: float = 1.5, dpi: int = 150, seed: int = 7) -> None:
    """Image-only copy of `src` that looks like a fax/scan: tilted, blurred, noisy, specked."""
    rng = np.random.default_rng(seed)
    out = pymupdf.open()
    with pymupdf.open(src) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
            img = img.filter(ImageFilter.GaussianBlur(0.6))
            arr = np.asarray(img, dtype=np.float32)
            arr += rng.normal(0, 18, arr.shape)  # sensor noise
            specks = rng.random(arr.shape) < 0.0015  # toner specks
            arr[specks] = 0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
    out.set_metadata({"title": "Scanned change order", "producer": "co-extract make_samples"})
    out.save(path, garbage=4, deflate=True)


if __name__ == "__main__":
    digital = SAMPLES / "co_002_digital.pdf"
    make_digital_g701(digital)
    print(f"wrote {digital}")
    scanned = SAMPLES / "co_003_scanned.pdf"
    make_scanned(digital, scanned)
    print(f"wrote {scanned}")
