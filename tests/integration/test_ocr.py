from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

from co_extract.ingest import extract_text, is_scanned, ocr
from tests.factories import SAMPLES

DIGITAL = SAMPLES / "co_002_digital.pdf"
SCANNED = SAMPLES / "co_003_scanned.pdf"
MIN_OCR_CONFIDENCE = 0.6  # quality bar for the synthetic scan (measured ~0.89 / 0.92)

needs_tesseract = pytest.mark.skipif(
    not ocr.tesseract_available(), reason="tesseract binary not installed"
)


def render(path: Path, page: int = 0, dpi: int = 200, rotate: float = 0.0) -> np.ndarray:
    with pymupdf.open(path) as doc:
        pix = doc[page].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, fillcolor=255)
    return np.asarray(img)


# --- scanned-page detection ----------------------------------------------------


def test_low_char_count_means_scanned():
    assert is_scanned("")
    assert is_scanned("  Page 1 of 2  ")
    assert not is_scanned("CHANGE ORDER " * 10)


def test_scanned_pdf_has_no_text_layer():
    with pymupdf.open(SCANNED) as doc:
        assert all(is_scanned(page.get_text()) for page in doc)


# --- preprocessing ---------------------------------------------------------------


@pytest.mark.parametrize("angle", [-3.0, -1.5, 0.0, 1.5, 3.0])
def test_estimate_skew_recovers_rotation(angle):
    image = render(DIGITAL, rotate=angle)
    assert ocr.estimate_skew(image) == pytest.approx(angle, abs=0.4)


def test_preprocess_outputs_binary_deskewed_image():
    image = render(DIGITAL, rotate=2.0)
    binary, skew = ocr.preprocess(image)
    assert binary.shape == image.shape
    assert set(np.unique(binary)) <= {0, 255}
    assert skew == pytest.approx(2.0, abs=0.4)
    assert abs(ocr.estimate_skew(binary)) < 0.4  # straight after deskew


# --- Tesseract on the scanned sample -------------------------------------------


@pytest.fixture(scope="module")
def scanned_pages():
    return extract_text(SCANNED)


@needs_tesseract
def test_scanned_pages_go_through_ocr(scanned_pages):
    assert len(scanned_pages) == 2
    for page in scanned_pages:
        assert page.method == "ocr"
        assert page.ocr_confidence is not None and page.ocr_confidence > MIN_OCR_CONFIDENCE
        assert page.skew_degrees == pytest.approx(1.5, abs=0.4)
        assert page.tables == []


@needs_tesseract
@pytest.mark.parametrize(
    "needle", ["CO-012", "10,763.06", "2,533,183.56", "Riverside Medical Office Building"]
)
def test_scanned_text_has_key_values(scanned_pages, needle):
    assert needle in scanned_pages[0].text


@needs_tesseract
def test_word_confidences_are_0_to_1():
    binary, _ = ocr.preprocess(render(DIGITAL))
    text, mean, words = ocr.run_tesseract(binary)
    assert words > 50
    assert 0.0 <= mean <= 1.0
    assert "CHANGE ORDER" in text
