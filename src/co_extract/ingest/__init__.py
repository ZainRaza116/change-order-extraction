"""Ingestion: turn a PDF or text file into per-page text.

Digital PDF pages use their text layer (PyMuPDF) and tables (pdfplumber); pages with
almost no text are treated as scans and go through OCR.
"""

from co_extract.ingest.documents import (
    MIN_TEXT_CHARS,
    SUPPORTED_SUFFIXES,
    PageText,
    extract_text,
    is_scanned,
)

__all__ = ["MIN_TEXT_CHARS", "SUPPORTED_SUFFIXES", "PageText", "extract_text", "is_scanned"]
