from __future__ import annotations
from dataclasses import dataclass
import re
import cv2
import numpy as np
import pytesseract

from candidates import find_character_boxes

PLATE_REGEXES: dict[str, str] = {
    "RO": r"^[A-Z]{1,2}\d{2,3}[A-Z]{2,3}$",
    # "US": r"^[A-Z0-9]{5,8}$",
    # "DE": r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$",
}
PLATE_REGION = "RO"

TESSERACT_CMD = ""
OCR_SCALE = 3.0
OCR_MIN_WIDTH = 160
OCR_BLUR_KERNEL = 3
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# PSM 10 = treat the image as a single character. PSM 7 = single text line.
PSM_PER_CHAR = 10
PSM_LINE = 7

CHAR_TARGET_HEIGHT = 48
CHAR_PAD = 8
MIN_SEGMENTED_CHARS = 4


@dataclass
class OCRResult:
    raw_text: str = ""
    text: str = ""
    valid: bool = False
    score: float = 0.0
    variant: str = ""
    image: np.ndarray | None = None


def current_plate_regex() -> str:
    return PLATE_REGEXES[PLATE_REGION]


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_text(text: str) -> bool:
    return re.fullmatch(current_plate_regex(), normalize_plate_text(text)) is not None


def ensure_gray(plate_image: np.ndarray) -> np.ndarray:
    if plate_image.ndim == 2:
        return plate_image
    return cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    scale = max(OCR_SCALE, OCR_MIN_WIDTH / max(width, 1))
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def ocr_score(text: str) -> float:
    normalized = normalize_plate_text(text)
    if not normalized:
        return 0.0

    score = min(len(normalized), 8) / 8.0
    if is_valid_plate_text(normalized):
        score += 2.0

    letter_count = sum(ch.isalpha() for ch in normalized)
    digit_count = sum(ch.isdigit() for ch in normalized)
    if 4 <= len(normalized) <= 8:
        score += 0.25
    if letter_count >= 3:
        score += 0.15
    if 2 <= digit_count <= 3:
        score += 0.15
    return score


def preprocess_char(char_img: np.ndarray) -> np.ndarray:
    height, width = char_img.shape
    scale = CHAR_TARGET_HEIGHT / max(height, 1)
    new_width = max(int(round(width * scale)), 1)
    resized = cv2.resize(char_img, (new_width, CHAR_TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tesseract expects dark text on light background.
    if float(np.mean(binary)) < 127:
        binary = 255 - binary
    return cv2.copyMakeBorder(
        binary,
        CHAR_PAD, CHAR_PAD, CHAR_PAD, CHAR_PAD,
        cv2.BORDER_CONSTANT, value=255,
    )


def ocr_character(char_img: np.ndarray, config: str) -> str:
    raw = pytesseract.image_to_string(char_img, config=config)
    cleaned = normalize_plate_text(raw)
    return cleaned[:1] if cleaned else ""


def recognize_plate_per_char(plate_image: np.ndarray) -> OCRResult | None:
    gray = ensure_gray(plate_image)
    resized = resize_for_ocr(gray)
    boxes = find_character_boxes(resized)
    if len(boxes) < MIN_SEGMENTED_CHARS:
        return None

    config = (
        f"--oem 3 --psm {PSM_PER_CHAR} "
        f"-c tessedit_char_whitelist={OCR_WHITELIST}"
    )

    vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    chars: list[str] = []
    for x, y, w, h in boxes:
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(resized.shape[1], x + w + pad)
        y1 = min(resized.shape[0], y + h + pad)
        char_img = resized[y0:y1, x0:x1]
        if char_img.size == 0:
            continue
        preprocessed = preprocess_char(char_img)
        chars.append(ocr_character(preprocessed, config))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)

    raw = "".join(chars)
    normalized = normalize_plate_text(raw)
    return OCRResult(
        raw_text=raw,
        text=normalized,
        valid=is_valid_plate_text(normalized),
        score=ocr_score(normalized),
        variant="per_char",
        image=vis,
    )


def recognize_plate_full(plate_image: np.ndarray) -> OCRResult:
    gray = ensure_gray(plate_image)
    resized = resize_for_ocr(gray)
    blur_kernel = max(1, OCR_BLUR_KERNEL)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(resized, (blur_kernel, blur_kernel), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = (
        f"--oem 3 --psm {PSM_LINE} "
        f"-c tessedit_char_whitelist={OCR_WHITELIST}"
    )
    raw = pytesseract.image_to_string(otsu, config=config)
    normalized = normalize_plate_text(raw)
    return OCRResult(
        raw_text=raw.strip(),
        text=normalized,
        valid=is_valid_plate_text(normalized),
        score=ocr_score(normalized),
        variant="full_otsu",
        image=otsu,
    )


def recognize_plate(plate_image: np.ndarray | None) -> OCRResult:
    if plate_image is None or plate_image.size == 0:
        return OCRResult()

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    best = OCRResult()
    try:
        per_char = recognize_plate_per_char(plate_image)
        if per_char is not None:
            best = per_char

        if not best.valid:
            fallback = recognize_plate_full(plate_image)
            if fallback.valid or fallback.score > best.score:
                best = fallback
    except Exception:
        return best

    return best
