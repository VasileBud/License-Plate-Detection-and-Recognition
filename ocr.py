from __future__ import annotations
from dataclasses import dataclass
import re
import cv2
import numpy as np
import pytesseract


@dataclass
class OCRResult:
    text: str = ""
    valid: bool = False
    image: np.ndarray | None = None


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_text(text: str) -> bool:
    return re.fullmatch(r"^[A-Z]{1,2}\d{2,3}[A-Z]{2,3}$", normalize_plate_text(text)) is not None


def ensure_gray(plate_image: np.ndarray) -> np.ndarray:
    return plate_image if plate_image.ndim == 2 else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    scale = max(3.0, 160 / max(w, 1))
    return cv2.resize(gray, (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
                      interpolation=cv2.INTER_CUBIC)


def recognize_plate(plate_image: np.ndarray | None) -> OCRResult:
    if plate_image is None or plate_image.size == 0:
        return OCRResult()
    try:
        gray = ensure_gray(plate_image)
        resized = resize_for_ocr(gray)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw = pytesseract.image_to_string(binary, config=config)
    except Exception:
        return OCRResult()
    text = normalize_plate_text(raw)
    return OCRResult(text=text, valid=is_valid_plate_text(text), image=binary)


def _quality(result: OCRResult) -> int:
    return (10 if result.valid else 0) + len(result.text)


def recognize_best_plate(plate_images: list[np.ndarray | None]) -> tuple[OCRResult, int]:
    best = OCRResult()
    best_idx = -1
    for i, plate in enumerate(plate_images):
        if plate is None or plate.size == 0:
            continue
        result = recognize_plate(plate)
        if best_idx < 0 or _quality(result) > _quality(best):
            best = result
            best_idx = i
    return best, best_idx
