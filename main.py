import sys
from tkinter import Tk, filedialog
import cv2
import numpy as np

from candidates import contour_length_threshold, filter_plate_candidates, nms_candidates, order_4_points
from contours import find_contours
from edges import canny
from morphology import closing
from ocr import OCRResult, recognize_best_plate
from viz import close_debug_views, show_debug_views

TOP_K = 5
TOP_K_OCR = 3
SHOW_DEBUG_WINDOWS = True
WAIT_KEY_MS = 0


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


def compute_closing_kernel(image_shape) -> int:
    kernel = max(3, int(round(min(image_shape[:2]) * 0.004)))
    return kernel + 1 if kernel % 2 == 0 else kernel


def detect_plate_candidates(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    kh = compute_closing_kernel(image.shape)
    element = np.ones((kh, kh), dtype=np.uint8)
    closed = closing(edges, element)
    contours = find_contours(closed, contour_length_threshold(image.shape))
    candidates = filter_plate_candidates(image, contours, edges, gray)
    candidates = nms_candidates(candidates, 0.4)
    return candidates, edges, closed, contours, element


def crop_plate(image: np.ndarray, box) -> np.ndarray | None:
    src = order_4_points(box)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    img_h, img_w = image.shape[:2]
    if w < max(30.0, img_w * 0.035) or h < max(10.0, img_h * 0.012):
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h))


def print_summary(contours, candidates, best, plate_image, ocr_result: OCRResult, ocr_idx: int) -> None:
    print("LICENSE PLATE DETECTION")
    print(f"Contours: {len(contours)}")
    print(f"Candidates: {len(candidates)}")
    for i, c in enumerate(candidates[:TOP_K]):
        marker = " <- OCR" if i == ocr_idx else ""
        print(f"#{i + 1}: SCORE={c['score']:.2f}, AR={c['aspect_ratio']:.2f}, "
              f"dens={c['density']:.2f}, contrast={c['contrast']:.0f}, "
              f"bright={c['brightness']:.0f}, chars={c['n_chars']}, "
              f"size={c['size'][0]:.0f}x{c['size'][1]:.0f}{marker}")

    if best is None:
        print("No candidate found!")
        return
    print(f"Best (for OCR, #{ocr_idx + 1}): SCORE={best['score']:.2f}, AR={best['aspect_ratio']:.2f}, "
          f"contrast={best['contrast']:.0f}, bright={best['brightness']:.0f}")

    if plate_image is None:
        print("A plate-like region was detected, but cropping failed or the ROI is too small.")
        return
    if ocr_result.text:
        print(f"OCR: {ocr_result.text}")
    else:
        print("OCR: no result.")


def main() -> None:
    path = select_image()
    if not path:
        print("No image.")
        sys.exit(0)

    image = load_image(path)
    candidates, edges, closed, contours, element = detect_plate_candidates(image)

    plate_images = [crop_plate(image, c["box"]) for c in candidates[:TOP_K_OCR]]
    ocr_result, ocr_idx = recognize_best_plate(plate_images)

    if ocr_idx >= 0:
        best = candidates[ocr_idx]
        best_box = best["box"]
        plate_image = plate_images[ocr_idx]
    elif candidates:
        ocr_idx = 0
        best = candidates[0]
        best_box = best["box"]
        plate_image = plate_images[0] if plate_images else None
    else:
        best = None
        best_box = None
        plate_image = None

    print_summary(contours, candidates, best, plate_image, ocr_result, ocr_idx)

    if SHOW_DEBUG_WINDOWS:
        show_debug_views(image=image, edges=edges, closed=closed,
                         contours=contours, candidates=candidates,
                         best_box_original=best_box, plate_image=plate_image,
                         ocr_result=ocr_result, closing_kernel_shape=element.shape,
                         show_top_k=TOP_K)
        close_debug_views(WAIT_KEY_MS)


if __name__ == "__main__":
    main()
