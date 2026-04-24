from __future__ import annotations
from dataclasses import dataclass
import re
import cv2
import numpy as np
import pytesseract

PLATE_REGEX = r"^[A-Z]{1,2}\d{2,3}[A-Z]{2,3}$"
TESSERACT_CMD = ""
TESSERACT_PSM = 7
OCR_SCALE = 3.0
OCR_MIN_WIDTH = 160
OCR_BLUR_KERNEL = 3
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class OCRResult:
    raw_text: str = ""
    text: str = ""
    valid: bool = False
    score: float = 0.0
    variant: str = ""
    image: np.ndarray | None = None


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_text(text: str) -> bool:
    return re.fullmatch(PLATE_REGEX, normalize_plate_text(text)) is not None

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


def prepare_plate_variants(plate_image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = ensure_gray(plate_image)
    resized = resize_for_ocr(gray)

    blur_kernel = max(1, OCR_BLUR_KERNEL)
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(resized, (blur_kernel, blur_kernel), 0)
    equalized = cv2.equalizeHist(blurred)

    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    return [
        ("gray", resized),
        ("equalized", equalized),
        ("otsu", otsu),
        ("adaptive", adaptive),
    ]


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


def recognize_plate(plate_image: np.ndarray | None) -> OCRResult:
    if plate_image is None or plate_image.size == 0:
        return OCRResult()

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    tesseract_config = (
        f"--oem 3 --psm {TESSERACT_PSM} "
        f"-c tessedit_char_whitelist={OCR_WHITELIST}"
    )

    best = OCRResult()
    try:
        variants = prepare_plate_variants(plate_image)
        for name, variant in variants:
            raw = pytesseract.image_to_string(variant, config=tesseract_config)
            normalized = normalize_plate_text(raw)
            score = ocr_score(normalized)
            current = OCRResult(
                raw_text=raw.strip(),
                text=normalized,
                valid=is_valid_plate_text(normalized),
                score=score,
                variant=name,
                image=variant,
            )
            if current.valid and not best.valid:
                best = current
            elif current.valid == best.valid and current.score > best.score:
                best = current
    except Exception:
        return OCRResult()

    return best
