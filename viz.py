import cv2
import numpy as np
from ocr import OCRResult


def draw_contours_overlay(image: np.ndarray, contours: list[list[tuple[float, float]]]) -> np.ndarray:
    overlay = image.copy()
    for contour in contours:
        points = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], True, (0, 255, 0), 1)
    return overlay


def draw_candidates_overlay(image: np.ndarray, candidates: list[dict], show_top_k: int) -> np.ndarray:
    overlay = image.copy()
    for i, candidate in enumerate(candidates[:show_top_k]):
        box_points = np.array(candidate["box"], dtype=np.int32)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.polylines(overlay, [box_points], True, color, 2)
        center_x, center_y = int(candidate["center"][0]), int(candidate["center"][1])
        label = f"#{i + 1} s={candidate['score']:.1f}"
        cv2.putText(
            overlay,
            label,
            (center_x - 35, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return overlay


def draw_result_overlay(image: np.ndarray, best_box_original: list[tuple[float, float]] | None, ocr_result: OCRResult) -> np.ndarray:
    overlay = image.copy()
    if best_box_original is not None:
        box_points = np.array(best_box_original, dtype=np.int32)
        cv2.polylines(overlay, [box_points], True, (0, 255, 0), 3)

        if ocr_result.text:
            text = ocr_result.text
            if not ocr_result.valid:
                text = f"{text} ?"
            x, y = box_points[0]
            cv2.putText(
                overlay,
                text,
                (int(x), max(int(y) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
    return overlay


def build_debug_views(image, edges, closed, contours, candidates, best_box_original, plate_image, ocr_result, closing_kernel, show_top_k) -> dict[str, np.ndarray]:

    views = {
        "1. Canny": edges,
        f"2. Closing {closing_kernel}x{closing_kernel}": closed,
        "3. Contours": draw_contours_overlay(image, contours),
        "4. Candidates": draw_candidates_overlay(image, candidates, show_top_k),
        "7. Result": draw_result_overlay(image, best_box_original, ocr_result),
    }
    if plate_image is not None:
        views["5. Cropped plate"] = plate_image
    if ocr_result.image is not None:
        views["6. OCR preprocess"] = ocr_result.image
    return views


def show_debug_views(
        image: np.ndarray,
        edges: np.ndarray,
        closed: np.ndarray,
        contours: list[list[tuple[float, float]]],
        candidates: list[dict],
        best_box_original: list[tuple[float, float]] | None,
        plate_image: np.ndarray | None,
        ocr_result: OCRResult,
        closing_kernel: int,
        show_top_k: int,
) -> None:
    for name, view in build_debug_views(
            image=image,
            edges=edges,
            closed=closed,
            contours=contours,
            candidates=candidates,
            best_box_original=best_box_original,
            plate_image=plate_image,
            ocr_result=ocr_result,
            closing_kernel=closing_kernel,
            show_top_k=show_top_k,
    ).items():
        cv2.imshow(name, view)


def close_debug_views(wait_key_ms: int = 0) -> None:
    cv2.waitKey(wait_key_ms)
    cv2.destroyAllWindows()
