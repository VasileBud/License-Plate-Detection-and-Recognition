import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from main import TOP_K_OCR, crop_plate, detect_plate_candidates
from ocr import OCRResult, recognize_best_plate
from viz import draw_result_overlay

INPUT_DIR = Path("dataset/images")
RESULTS_DIR = Path("dataset/results")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def list_images(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        print(f"Input folder not found: {directory}")
        sys.exit(1)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def reset_results_dir(result_dir: Path) -> None:
    dataset_dir = Path("dataset").resolve()
    if dataset_dir not in result_dir.resolve().parents:
        print(f"Invalid results folder: {result_dir}")
        sys.exit(1)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def process_image(path: Path) -> tuple[np.ndarray, OCRResult]:
    image = load_image(path)
    candidates, _, _, _, _, _ = detect_plate_candidates(image)

    plate_images = [crop_plate(image, c["box"]) for c in candidates[:TOP_K_OCR]]
    ocr_result, ocr_idx = recognize_best_plate(plate_images)
    if ocr_idx >= 0:
        best_box = candidates[ocr_idx]["box"]
    elif candidates:
        best_box = candidates[0]["box"]
    else:
        best_box = None

    return draw_result_overlay(image, best_box, ocr_result), ocr_result


def main() -> None:
    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        sys.exit(0)

    reset_results_dir(RESULTS_DIR)

    total = 0
    valid = 0
    for image_path in image_paths:
        try:
            result_image, ocr_result = process_image(image_path)
        except ValueError as exc:
            print(exc)
            continue

        cv2.imwrite(str(RESULTS_DIR / image_path.name), result_image)
        total += 1
        if ocr_result.valid:
            valid += 1
        tag = "VALID" if ocr_result.valid else "----"
        print(f"[{tag}] {ocr_result.text!r:12s} conf={ocr_result.confidence:5.1f}  {image_path.name}")

    if total:
        print(f"Done. Valid plates: {valid}/{total} ({100.0 * valid / total:.1f}%)")
    else:
        print("Done")


if __name__ == "__main__":
    main()
