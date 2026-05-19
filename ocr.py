from __future__ import annotations
from dataclasses import dataclass
import re
import cv2
import numpy as np
import pytesseract

from candidates import find_character_boxes

PLATE_REGEXES: dict[str, str] = {
    "RO": r"^[A-Z]{1,2}\d{2,3}[A-Z]{2,3}$",
}
PLATE_REGION = "RO"

TESSERACT_CMD = ""
OCR_SCALE = 3.0
OCR_MIN_WIDTH = 160
OCR_BLUR_KERNEL = 3
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

PSM_PER_CHAR = 10
PSM_LINE = 7

CHAR_TARGET_HEIGHT = 48
CHAR_PAD = 8
MIN_SEGMENTED_CHARS = 4

CONFIDENCE_LOW = 60.0

@dataclass
class OCRResult:
    raw_text: str = ""
    text: str = ""
    valid: bool = False
    score: float = 0.0
    variant: str = ""
    image: np.ndarray | None = None
    confidence: float = 0.0

def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_text(text: str) -> bool:
    return re.fullmatch(PLATE_REGEXES[PLATE_REGION], normalize_plate_text(text)) is not None


def ensure_gray(plate_image: np.ndarray) -> np.ndarray:
    return plate_image if plate_image.ndim == 2 else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    scale = max(OCR_SCALE, OCR_MIN_WIDTH / max(w, 1))
    return cv2.resize(gray, (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
                      interpolation=cv2.INTER_CUBIC)


def ocr_score(text: str) -> float:
    s = normalize_plate_text(text)
    if not s:
        return 0.0
    score = min(len(s), 8) / 8.0
    if is_valid_plate_text(s):
        score += 2.0
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    if 4 <= len(s) <= 8:
        score += 0.25
    if letters >= 3:
        score += 0.15
    if 2 <= digits <= 3:
        score += 0.15
    return score


def ocr_quality(result: OCRResult) -> float:
    return (10.0 if result.valid else 0.0) + result.score + result.confidence / 100.0


def preprocess_char(char_img: np.ndarray) -> np.ndarray:
    h, w = char_img.shape
    new_w = max(int(round(w * CHAR_TARGET_HEIGHT / max(h, 1))), 1)
    resized = cv2.resize(char_img, (new_w, CHAR_TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(binary.mean()) < 127:
        binary = 255 - binary
    return cv2.copyMakeBorder(binary, CHAR_PAD, CHAR_PAD, CHAR_PAD, CHAR_PAD,
                              cv2.BORDER_CONSTANT, value=255)


def ocr_character_with_conf(char_img: np.ndarray, config: str) -> tuple[str, float]:
    data = pytesseract.image_to_data(char_img, config=config, output_type=pytesseract.Output.DICT)
    best_ch = ""
    best_conf = -1.0
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 0:
            continue
        normalized = normalize_plate_text(text)
        if not normalized:
            continue
        if conf > best_conf:
            best_ch = normalized[0]
            best_conf = conf
    return best_ch, best_conf


def _build_result(raw: str, variant: str, image, confidence: float = 0.0) -> OCRResult:
    normalized = normalize_plate_text(raw)
    return OCRResult(
        raw_text=raw.strip() if isinstance(raw, str) else raw,
        text=normalized,
        valid=is_valid_plate_text(normalized),
        score=ocr_score(normalized),
        variant=variant,
        image=image,
        confidence=confidence,
    )


def recognize_plate_per_char(plate_image: np.ndarray) -> OCRResult | None:
    resized = resize_for_ocr(ensure_gray(plate_image))
    boxes = find_character_boxes(resized)
    if len(boxes) < MIN_SEGMENTED_CHARS:
        return None

    config = f"--oem 3 --psm {PSM_PER_CHAR} -c tessedit_char_whitelist={OCR_WHITELIST}"
    vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    chars: list[str] = []
    confs: list[float] = []
    pad = 2
    for x, y, w, h in boxes:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(resized.shape[1], x + w + pad), min(resized.shape[0], y + h + pad)
        char_img = resized[y0:y1, x0:x1]
        if char_img.size == 0:
            continue
        ch, conf = ocr_character_with_conf(preprocess_char(char_img), config)
        chars.append(ch)
        # color the bbox green for high confidence, orange for low
        color = (0, 255, 0) if conf >= CONFIDENCE_LOW else (0, 165, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
        if conf >= 0:
            confs.append(conf)

    avg_conf = float(np.mean(confs)) if confs else 0.0
    return _build_result("".join(chars), "per_char", vis, avg_conf)


def recognize_plate_full(plate_image: np.ndarray) -> OCRResult:
    resized = resize_for_ocr(ensure_gray(plate_image))
    blurred = cv2.GaussianBlur(resized, (OCR_BLUR_KERNEL, OCR_BLUR_KERNEL), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f"--oem 3 --psm {PSM_LINE} -c tessedit_char_whitelist={OCR_WHITELIST}"
    data = pytesseract.image_to_data(otsu, config=config, output_type=pytesseract.Output.DICT)

    parts: list[str] = []
    confs: list[float] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 0:
            continue
        parts.append(text)
        confs.append(conf)

    raw = " ".join(parts)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return _build_result(raw, "full_otsu", otsu, avg_conf)


def recognize_plate(plate_image: np.ndarray | None) -> OCRResult:
    if plate_image is None or plate_image.size == 0:
        return OCRResult()
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    best = OCRResult()
    try:
        per_char = recognize_plate_per_char(plate_image)
        if per_char is not None and ocr_quality(per_char) > ocr_quality(best):
            best = per_char
        full = recognize_plate_full(plate_image)
        if ocr_quality(full) > ocr_quality(best):
            best = full
    except Exception:
        return best
    return best


def recognize_best_plate(plate_images: list[np.ndarray | None]) -> tuple[OCRResult, int]:
    best = OCRResult()
    best_idx = -1
    for i, plate in enumerate(plate_images):
        if plate is None or plate.size == 0:
            continue
        result = recognize_plate(plate)
        if best_idx < 0 or ocr_quality(result) > ocr_quality(best):
            best = result
            best_idx = i
    return best, best_idx
