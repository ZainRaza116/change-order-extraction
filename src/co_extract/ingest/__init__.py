"""Ingestion stage: turn a PDF, image or text file into per-page text.

Digital PDF pages use their text layer (PyMuPDF) and tables (pdfplumber); pages with
almost no text are treated as scans and go through OCR.
"""

from co_extract.ingest.documents import (
    IMAGE_SUFFIXES,
    MIN_TEXT_CHARS,
    PageText,
    extract_text,
    is_scanned,
)

SUPPORTED_SUFFIXES = {".pdf", ".txt", *IMAGE_SUFFIXES}

__all__ = [
    "IMAGE_SUFFIXES",
    "MIN_TEXT_CHARS",
    "SUPPORTED_SUFFIXES",
    "PageText",
    "extract_text",
    "is_scanned",
]
