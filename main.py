import sys
from tkinter import Tk, filedialog
import cv2
import numpy as np

from candidates import contour_length_threshold, filter_plate_candidates, order_4_points
from contours import find_contours
from edges import canny
from morphology import closing
from ocr import OCRResult, recognize_plate
from viz import close_debug_views, show_debug_views

TOP_K = 5
SHOW_DEBUG_WINDOWS = True
WAIT_KEY_MS = 0
CLOSING_KERNEL_RATIO = 0.004
MIN_PLATE_CROP_WIDTH_RATIO = 0.035
MIN_PLATE_CROP_HEIGHT_RATIO = 0.012


def select_image() -> str | None:
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir="dataset",
        title="Choose an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )
    root.destroy()
    return path or None


def load_image(path: str) -> cv2.typing.MatLike:
    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        sys.exit(1)
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


def print_summary(
    contours: list[list[tuple[float, float]]],
    candidates: list[dict],
    best_candidate: dict | None,
    plate_image: np.ndarray | None,
    ocr_result: OCRResult,
) -> None:
    print("LICENSE PLATE DETECTION")
    print(f"Contours: {len(contours)}")
    print(f"Candidates: {len(candidates)}")

    for i, candidate in enumerate(candidates[:TOP_K]):
        print(
            f"#{i + 1}: SCORE={candidate['score']:.2f}, "
            f"AR={candidate['aspect_ratio']:.2f}, "
            f"dens={candidate['density']:.2f}, "
            f"contrast={candidate['contrast']:.0f}, "
            f"bright={candidate['brightness']:.0f}, "
            f"chars={candidate['n_chars']}, "
            f"size={candidate['size'][0]:.0f}x{candidate['size'][1]:.0f}"
        )

    if best_candidate is None:
        print("No candidate found!")
        return

    best = best_candidate
    print(
        f"Best: SCORE={best['score']:.2f}, "
        f"AR={best['aspect_ratio']:.2f}, "
        f"contrast={best['contrast']:.0f}, "
        f"bright={best['brightness']:.0f}"
    )

    if plate_image is None:
        print("A plate-like region was detected, but cropping failed or the ROI is too small.")
        return

    if ocr_result.text:
        validity = "valid" if ocr_result.valid else "invalid"
        print(
            f"OCR: {ocr_result.text} "
            f"({validity}, variant={ocr_result.variant}, score={ocr_result.score:.2f})"
        )
    else:
        print("OCR: no result.")


def main() -> None:
    path = select_image()
    if not path:
        print("No image.")
        sys.exit(0)

    image = load_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    closing_kernel = compute_closing_kernel(image.shape)
    min_contour_len = contour_length_threshold(image.shape)
    element = np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
    closed = closing(edges, element)
    contours = find_contours(closed, min_contour_len)
    candidates = filter_plate_candidates(contours, image.shape, edges, gray)

    best_candidate = candidates[0] if candidates else None
    best_box_original = None
    plate_image = None
    ocr_result = OCRResult()

    if best_candidate is not None:
        best_box_original = best_candidate["box"]
        plate_image = crop_plate(image, best_box_original)
        ocr_result = recognize_plate(plate_image)

    print_summary(contours, candidates, best_candidate, plate_image, ocr_result)

    if SHOW_DEBUG_WINDOWS:
        show_debug_views(
            image=image,
            edges=edges,
            closed=closed,
            contours=contours,
            candidates=candidates,
            best_box_original=best_box_original,
            plate_image=plate_image,
            ocr_result=ocr_result,
            closing_kernel=closing_kernel,
            show_top_k=TOP_K,
        )
        close_debug_views(WAIT_KEY_MS)


if __name__ == "__main__":
    main()