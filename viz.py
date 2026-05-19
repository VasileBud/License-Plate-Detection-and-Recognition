import cv2
import numpy as np
from ocr import OCRResult


def draw_contours_overlay(image, contours):
    overlay = image.copy()
    for contour in contours:
        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)
    return overlay


def draw_candidates_overlay(image, candidates, show_top_k):
    overlay = image.copy()
    for i, c in enumerate(candidates[:show_top_k]):
        pts = np.array(c["box"], dtype=np.int32)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.polylines(overlay, [pts], True, color, 2)
        cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(overlay, f"#{i + 1} s={c['score']:.1f}", (cx - 35, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return overlay


def draw_result_overlay(image, best_box, ocr_result: OCRResult):
    overlay = image.copy()
    if best_box is None:
        return overlay
    pts = np.array(best_box, dtype=np.int32)
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
    if ocr_result.text:
        text = ocr_result.text if ocr_result.valid else f"{ocr_result.text} ?"
        x, y = pts[0]
        cv2.putText(overlay, text, (int(x), max(int(y) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return overlay


def show_debug_views(image, working, edges, closed, contours, candidates, best_box_original,
                     plate_image, ocr_result, closing_kernel_shape, show_top_k):
    kh, kw = closing_kernel_shape
    views = {
        "1. Canny": edges,
        f"2. Closing {kh}x{kw}": closed,
        "3. Contours": draw_contours_overlay(working, contours),
        "4. Candidates": draw_candidates_overlay(image, candidates, show_top_k),
        "7. Result": draw_result_overlay(image, best_box_original, ocr_result),
    }
    if plate_image is not None:
        views["5. Cropped plate"] = plate_image
    if ocr_result.image is not None:
        views["6. OCR preprocess"] = ocr_result.image
    for name, view in views.items():
        cv2.imshow(name, view)


def close_debug_views(wait_key_ms: int = 0) -> None:
    cv2.waitKey(wait_key_ms)
    cv2.destroyAllWindows()
