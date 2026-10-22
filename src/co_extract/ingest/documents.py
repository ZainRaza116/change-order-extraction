"""Page text from PyMuPDF, tables from pdfplumber, OCR for scanned pages.

A PDF page with fewer than MIN_TEXT_CHARS of extractable text is treated as a scan:
it is rendered to an image and sent through the OCR path in ocr.py.
"""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pdfplumber
import pymupdf
from pydantic import BaseModel

from co_extract.ingest.ocr import OCR_DPI, ocr_image

MIN_TEXT_CHARS = 50
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

Table = list[list[str | None]]


class PageText(BaseModel):
    page: int  # 1-based
    text: str
    method: Literal["text", "ocr", "vision"]
    char_count: int
    tables: list[Table] = []
    ocr_confidence: float | None = None  # mean Tesseract word confidence, 0-1
    skew_degrees: float | None = None


def is_scanned(text: str) -> bool:
    return len(text.strip()) < MIN_TEXT_CHARS


def _ocr_page(number: int, image: np.ndarray, vision_model: str | None) -> PageText:
    result = ocr_image(image, vision_model=vision_model)
    return PageText(
        page=number,
        text=result.text,
        method=result.method,
        char_count=len(result.text),
        ocr_confidence=round(result.mean_confidence, 4),
        skew_degrees=round(result.skew_degrees, 2),
    )


def _render_gray(page: pymupdf.Page) -> np.ndarray:
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def extract_pdf(path: Path, vision_model: str | None = None) -> list[PageText]:
    pages = []
    with pymupdf.open(path) as doc, pdfplumber.open(path) as plumber:
        for page, plumber_page in zip(doc, plumber.pages, strict=True):
            text = page.get_text("text", sort=True).strip()
            if is_scanned(text):
                pages.append(_ocr_page(page.number + 1, _render_gray(page), vision_model))
                continue
            pages.append(
                PageText(
                    page=page.number + 1,
                    text=text,
                    method="text",
                    char_count=len(text),
                    tables=plumber_page.extract_tables(),
                )
            )
    return pages


def extract_image(path: Path, vision_model: str | None = None) -> list[PageText]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return [_ocr_page(1, image, vision_model)]


def extract_txt(path: Path) -> list[PageText]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [PageText(page=1, text=text, method="text", char_count=len(text))]


def extract_text(path: Path, vision_model: str | None = None) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, vision_model)
    if suffix in IMAGE_SUFFIXES:
        return extract_image(path, vision_model)
    if suffix == ".txt":
        return extract_txt(path)
    raise ValueError(f"unsupported file type: {suffix}")
