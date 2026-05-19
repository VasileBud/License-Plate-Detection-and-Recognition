from collections import deque
import cv2
import numpy as np

MIN_CONTOUR_LENGTH_RATIO = 0.015
MIN_HEIGHT_RATIO = 0.012
MIN_WIDTH_RATIO = 0.035
MAX_HEIGHT_RATIO = 0.20
MIN_ASPECT_RATIO = 2.5
MAX_ASPECT_RATIO = 5.5
MIN_AREA_RATIO = 0.001
MAX_AREA_RATIO = 0.06
REQUIRE_HORIZONTAL_SPAN = True

TARGET_ASPECT_RATIO = 4.7
TARGET_AREA_RATIO = 0.008
AREA_RATIO_TOLERANCE = 0.05

MIN_DENSITY = 0.03
DENSITY_PLATEAU = 0.35
DENSITY_FALLOFF = 0.30

CONTRAST_NORMALIZATION = 60.0
MIN_BRIGHTNESS = 80.0
BRIGHTNESS_RANGE = 120.0

WHITE_HSV_LOW = (0, 0, 140)
WHITE_HSV_HIGH = (180, 80, 255)
MIN_WHITE_RATIO = 0.30
TARGET_WHITE_RATIO = 0.60

POSITION_CENTER_Y = 0.72
POSITION_SIGMA = 0.18

WEIGHT_ASPECT_RATIO = 2.5
WEIGHT_DENSITY = 2.0
WEIGHT_CONTRAST = 3.0
WEIGHT_BRIGHTNESS = 2.0
WEIGHT_POSITION = 1.0
WEIGHT_AREA = 0.5
WEIGHT_TEXT = 2.5
WEIGHT_WHITE = 2.0

MIN_CHAR_HEIGHT_RATIO = 0.30
MAX_CHAR_HEIGHT_RATIO = 0.95
IDEAL_MIN_CHARS = 6
IDEAL_MAX_CHARS = 8
MIN_ABSOLUTE_CHARS = 3
MAX_ABSOLUTE_CHARS = 14


def contour_length_threshold(img_shape) -> int:
    min_dim = min(img_shape[0], img_shape[1])
    return max(15, int(round(min_dim * MIN_CONTOUR_LENGTH_RATIO)))


def plate_size_thresholds(img_shape):
    h, w = img_shape[:2]
    min_h = max(10.0, h * MIN_HEIGHT_RATIO)
    min_w = max(30.0, w * MIN_WIDTH_RATIO)
    max_h = max(min_h, h * MAX_HEIGHT_RATIO)
    return min_h, min_w, max_h


def order_4_points(points):
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(sums)],   # top-left
        pts[np.argmin(diffs)],  # top-right
        pts[np.argmax(sums)],   # bottom-right
        pts[np.argmax(diffs)],  # bottom-left
    ], dtype=np.float32)


def warp_roi(src, box):
    pts = order_4_points(box)
    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    if w < 10 or h < 5:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(src, cv2.getPerspectiveTransform(pts, dst), (w, h))


def has_horizontal_span(box) -> bool:
    pts = np.asarray(box, dtype=np.float64)
    sx = pts[:, 0].max() - pts[:, 0].min()
    sy = pts[:, 1].max() - pts[:, 1].min()
    return sx >= sy


def compute_white_mask(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(WHITE_HSV_LOW, np.uint8), np.array(WHITE_HSV_HIGH, np.uint8))


def label_components_bfs(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    di = [-1, -1, -1, 0, 0, 1, 1, 1]
    dj = [-1, 0, 1, -1, 1, -1, 0, 1]
    components = []
    label = 0

    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                label += 1
                labels[i, j] = label
                Q = deque()
                Q.append((i, j))
                min_i = i
                max_i = i
                min_j = j
                max_j = j
                count = 0
                while Q:
                    q = Q.popleft()
                    qi = q[0]
                    qj = q[1]
                    count += 1
                    if qi < min_i: min_i = qi
                    if qi > max_i: max_i = qi
                    if qj < min_j: min_j = qj
                    if qj > max_j: max_j = qj
                    for k in range(8):
                        ni = qi + di[k]
                        nj = qj + dj[k]
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if binary_image[ni, nj] == 255 and labels[ni, nj] == 0:
                                labels[ni, nj] = label
                                Q.append((ni, nj))
                components.append((min_i, min_j, max_i, max_j, count))
    return components


def _char_boxes_from_binary(binary, roi_height, min_area=10):
    min_h = MIN_CHAR_HEIGHT_RATIO * roi_height
    max_h = MAX_CHAR_HEIGHT_RATIO * roi_height
    boxes = []
    for min_i, min_j, max_i, max_j, area in label_components_bfs(binary):
        h = max_i - min_i + 1
        w = max_j - min_j + 1
        if min_h <= h <= max_h and h >= w and area >= min_area:
            boxes.append((min_j, min_i, w, h))
    return boxes


def count_characters_roi(gray_roi) -> int:
    h, w = gray_roi.shape
    if h < 8 or w < 20:
        return 0
    binary = np.where(gray_roi < gray_roi.mean(), 255, 0).astype(np.uint8)
    return len(_char_boxes_from_binary(binary, h))


def find_character_boxes(gray_roi, min_area: int = 10):
    h, w = gray_roi.shape
    if h < 8 or w < 20:
        return []
    blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    boxes = _char_boxes_from_binary(binary, h, min_area)
    boxes.sort(key=lambda b: b[0])
    return boxes


def score_text(n_chars: int) -> float:
    if n_chars < MIN_ABSOLUTE_CHARS or n_chars > MAX_ABSOLUTE_CHARS:
        return 0.0
    if IDEAL_MIN_CHARS <= n_chars <= IDEAL_MAX_CHARS:
        return 1.0
    if n_chars < IDEAL_MIN_CHARS:
        span = IDEAL_MIN_CHARS - MIN_ABSOLUTE_CHARS
        return (n_chars - MIN_ABSOLUTE_CHARS) / span if span > 0 else 0.5
    span = MAX_ABSOLUTE_CHARS - IDEAL_MAX_CHARS
    return (MAX_ABSOLUTE_CHARS - n_chars) / span if span > 0 else 0.5


def min_area_rect_pca(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        return ((x0 + w / 2, y0 + h / 2), (w, h), 0.0, corners)

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / len(pts)
    _, eigvecs = np.linalg.eigh(cov)
    angle = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    aligned = centered @ rot.T
    min_xy = aligned.min(axis=0)
    max_xy = aligned.max(axis=0)
    w = float(max(max_xy[0] - min_xy[0], 1))
    h = float(max(max_xy[1] - min_xy[1], 1))

    rot_back = rot.T
    local = np.array([[min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
                      [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]]])
    corners = local @ rot_back.T + centroid
    center = (min_xy + max_xy) / 2 @ rot_back.T + centroid

    return (
        (float(center[0]), float(center[1])),
        (w, h),
        float(np.degrees(angle)),
        [(float(x), float(y)) for x, y in corners],
    )


def _score(aspect, density, contrast, brightness, area_ratio, n_chars, white_ratio, center_y, img_h):
    ar = max(0.0, 1.0 - abs(aspect - TARGET_ASPECT_RATIO) / TARGET_ASPECT_RATIO)
    if density < MIN_DENSITY:
        dens = 0.0
    elif density <= DENSITY_PLATEAU:
        dens = 1.0
    else:
        dens = max(0.0, 1.0 - (density - DENSITY_PLATEAU) / DENSITY_FALLOFF)
    contr = min(1.0, contrast / CONTRAST_NORMALIZATION)
    bright = min(1.0, max(0.0, (brightness - MIN_BRIGHTNESS) / BRIGHTNESS_RANGE))
    pos = float(np.exp(-0.5 * ((center_y / img_h - POSITION_CENTER_Y) / POSITION_SIGMA) ** 2))
    area = max(0.0, 1.0 - abs(area_ratio - TARGET_AREA_RATIO) / AREA_RATIO_TOLERANCE)
    text = score_text(n_chars)
    white = min(1.0, white_ratio / TARGET_WHITE_RATIO) if TARGET_WHITE_RATIO > 0 else 0.0
    return (ar * WEIGHT_ASPECT_RATIO + dens * WEIGHT_DENSITY + contr * WEIGHT_CONTRAST
            + bright * WEIGHT_BRIGHTNESS + pos * WEIGHT_POSITION + area * WEIGHT_AREA
            + text * WEIGHT_TEXT + white * WEIGHT_WHITE)


def filter_plate_candidates(img, contours, edges, gray, debug=False):
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    min_contour_len = contour_length_threshold(img.shape)
    min_height, min_width, max_height = plate_size_thresholds(img.shape)
    white_mask = compute_white_mask(img)
    candidates = []

    for contour in contours:
        if len(contour) < min_contour_len:
            continue
        rect = min_area_rect_pca(contour)
        if rect is None:
            continue
        center, (w, h), angle, box = rect
        if REQUIRE_HORIZONTAL_SPAN and not has_horizontal_span(box):
            continue
        if h > w:
            w, h = h, w
        if h < min_height or w < min_width or h > max_height:
            continue

        aspect = w / h
        area_ratio = (w * h) / img_area
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue

        gray_roi = warp_roi(gray, box)
        edges_roi = warp_roi(edges, box)
        white_roi = warp_roi(white_mask, box)
        if gray_roi is None or edges_roi is None or white_roi is None or white_roi.size == 0:
            continue

        white_ratio = float(np.count_nonzero(white_roi)) / white_roi.size
        if white_ratio < MIN_WHITE_RATIO:
            continue

        density = np.count_nonzero(edges_roi) / edges_roi.size if edges_roi.size else 0.0
        contrast = float(gray_roi.std())
        brightness = float(gray_roi.mean())
        n_chars = count_characters_roi(gray_roi)
        if n_chars < MIN_ABSOLUTE_CHARS or n_chars > MAX_ABSOLUTE_CHARS:
            continue

        score = _score(aspect, density, contrast, brightness, area_ratio, n_chars, white_ratio, center[1], img_h)
        candidates.append({
            "contour": contour, "center": center, "size": (w, h), "angle": angle, "box": box,
            "area": w * h, "aspect_ratio": aspect, "area_ratio": area_ratio,
            "density": density, "contrast": contrast, "brightness": brightness,
            "n_chars": n_chars, "white_ratio": white_ratio, "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    if debug:
        print(f"[filter] accepted={len(candidates)}")
    return candidates


def _aabb_from_box(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_candidates(candidates, iou_threshold=0.4):
    if not candidates:
        return []
    aabbs = [_aabb_from_box(c["box"]) for c in candidates]
    suppressed = [False] * len(candidates)
    keep = []
    for i in range(len(candidates)):
        if suppressed[i]:
            continue
        keep.append(candidates[i])
        for j in range(i + 1, len(candidates)):
            if not suppressed[j] and _iou(aabbs[i], aabbs[j]) > iou_threshold:
                suppressed[j] = True
    return keep
