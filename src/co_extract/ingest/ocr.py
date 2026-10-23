"""OCR for scanned pages: OpenCV clean-up, then Tesseract with word confidences."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

OCR_DPI = 300
MAX_DESKEW_DEGREES = 10.0


@dataclass
class OcrResult:
    text: str
    mean_confidence: float  # 0-1, mean of Tesseract word confidences
    word_count: int
    skew_degrees: float


# --- preprocessing -------------------------------------------------------------


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    code = cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(image, code)


def estimate_skew(gray: np.ndarray) -> float:
    """Angle in degrees (counter-clockwise positive) that the text lines are tilted by.

    Smears text into horizontal bands, then takes the median angle of the band outlines.
    """
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bands = cv2.dilate(inverted, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    contours, _ = cv2.findContours(bands, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        if max(rect[1]) < 100 or min(rect[1]) < 5:
            continue  # specks and tiny marks say nothing about line direction
        # Angle of the rectangle's long edge; OpenCV's own angle field changes convention
        # across versions, so derive it from the corner points instead.
        p0, p1, p2 = cv2.boxPoints(rect)[:3]
        dx, dy = max((p1 - p0, p2 - p1), key=lambda v: np.hypot(*v))
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        if abs(angle) <= MAX_DESKEW_DEGREES:
            angles.append(-angle)  # image y-axis points down; flip to CCW-positive
    return float(np.median(angles)) if angles else 0.0


def remove_rules(binary: np.ndarray) -> np.ndarray:
    """White out long horizontal/vertical lines (table grids, signature lines).

    Tesseract reads ruled lines as '|', '_' or '-' junk with low confidence.
    """
    inverted = 255 - binary
    h, w = binary.shape
    horizontal = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 20), 1))
    )
    vertical = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 20)))
    )
    rules = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    cleaned = binary.copy()
    cleaned[rules > 0] = 255
    return cleaned


def rotate(gray: np.ndarray, degrees: float) -> np.ndarray:
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Grayscale -> denoise -> deskew -> Otsu threshold -> remove ruled lines.

    Returns (binary image, skew found in degrees).
    """
    gray = to_gray(image)
    # Median beats non-local-means here: scan noise is mostly specks, and on the scanned
    # sample it lifts mean word confidence from ~0.60 to ~0.89 (see WRITEUP).
    gray = cv2.medianBlur(gray, 3)
    skew = estimate_skew(gray)
    if abs(skew) > 0.1:
        gray = rotate(gray, -skew)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return remove_rules(binary), skew


# --- Tesseract -----------------------------------------------------------------


def tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        return False
    return True


def run_tesseract(binary: np.ndarray) -> tuple[str, float, int]:
    """Text with line breaks preserved, mean word confidence (0-1), and word count."""
    data = pytesseract.image_to_data(binary, config="--psm 4", output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not word.strip():
            continue  # -1 marks layout boxes, not words
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf / 100)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    mean = float(np.mean(confidences)) if confidences else 0.0
    return text, mean, len(confidences)


# --- entry point ---------------------------------------------------------------


def ocr_image(image: np.ndarray) -> OcrResult:
    """OCR one page image: clean it up, then read it with Tesseract."""
    binary, skew = preprocess(image)
    text, mean_conf, words = run_tesseract(binary)
    return OcrResult(text=text, mean_confidence=mean_conf, word_count=words, skew_degrees=skew)
