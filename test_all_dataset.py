import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from candidates import compute_white_mask, contour_length_threshold, filter_plate_candidates, order_4_points
from contours import find_contours
from edges import canny
from morphology import closing
from ocr import OCRResult, recognize_plate
from viz import draw_result_overlay

INPUT_DIR = Path("dataset/images")
RESULTS_DIR = Path("dataset/results")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
CLOSING_KERNEL_RATIO = 0.004
MIN_PLATE_CROP_WIDTH_RATIO = 0.035
MIN_PLATE_CROP_HEIGHT_RATIO = 0.012


def list_images(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        print(f"Input folder not found: {directory}")
        sys.exit(1)

    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def reset_results_dir(result_dir: Path) -> None:
    dataset_dir = Path("dataset").resolve()
    resolved_result_dir = result_dir.resolve()
    if dataset_dir not in resolved_result_dir.parents:
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


def compute_closing_kernel(image_shape: tuple[int, ...]) -> int:
    min_dim = min(image_shape[0], image_shape[1])
    kernel = max(3, int(round(min_dim * CLOSING_KERNEL_RATIO)))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def plate_crop_thresholds(image_shape: tuple[int, ...]) -> tuple[float, float]:
    image_height, image_width = image_shape[:2]
    min_width = max(30.0, image_width * MIN_PLATE_CROP_WIDTH_RATIO)
    min_height = max(10.0, image_height * MIN_PLATE_CROP_HEIGHT_RATIO)
    return min_width, min_height


def crop_plate(image: np.ndarray, box: list[tuple[float, float]]) -> np.ndarray | None:
    source = order_4_points(box)
    width = int(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3])))
    height = int(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1])))
    min_crop_width, min_crop_height = plate_crop_thresholds(image.shape)
    if width < min_crop_width or height < min_crop_height:
        return None

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def process_image(path: Path) -> np.ndarray:
    image = load_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_mask = compute_white_mask(image)
    edges = canny(gray)
    closing_kernel = compute_closing_kernel(image.shape)
    min_contour_len = contour_length_threshold(image.shape)
    closed = closing(edges, np.ones((closing_kernel, closing_kernel), dtype=np.uint8))
    contours = find_contours(closed, min_contour_len)
    candidates = filter_plate_candidates(contours, image.shape, edges, gray, white_mask)

    best_candidate = candidates[0] if candidates else None
    best_box = best_candidate["box"] if best_candidate is not None else None
    plate_image = crop_plate(image, best_box) if best_box is not None else None
    ocr_result = recognize_plate(plate_image) if plate_image is not None else OCRResult()

    return draw_result_overlay(image, best_box, ocr_result)


def main() -> None:
    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        sys.exit(0)

    reset_results_dir(RESULTS_DIR)

    for image_path in image_paths:
        try:
            result_image = process_image(image_path)
        except ValueError as exc:
            print(exc)
            continue

        output_path = RESULTS_DIR / image_path.name
        cv2.imwrite(str(output_path), result_image)
        print(f"Saved: {output_path}")

    print("Done")

if __name__ == "__main__":
    main()
