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

WEIGHT_ASPECT_RATIO = 2.5
WEIGHT_DENSITY = 2.0
WEIGHT_CONTRAST = 3.0
WEIGHT_BRIGHTNESS = 2.0
WEIGHT_POSITION = 1.0
WEIGHT_AREA = 0.5
WEIGHT_TEXT = 2.5

MIN_CHAR_HEIGHT_RATIO = 0.30
MAX_CHAR_HEIGHT_RATIO = 0.95
IDEAL_MIN_CHARS = 6
IDEAL_MAX_CHARS = 8
MIN_ABSOLUTE_CHARS = 3
MAX_ABSOLUTE_CHARS = 14


def contour_length_threshold(img_shape) -> int:
    img_height, img_width = img_shape[:2]
    min_dim = min(img_height, img_width)
    return max(15, int(round(min_dim * MIN_CONTOUR_LENGTH_RATIO)))


def plate_size_thresholds(img_shape) -> tuple[float, float, float]:
    img_height, img_width = img_shape[:2]
    min_height = max(10.0, img_height * MIN_HEIGHT_RATIO)
    min_width = max(30.0, img_width * MIN_WIDTH_RATIO)
    max_height = max(min_height, img_height * MAX_HEIGHT_RATIO)
    return min_height, min_width, max_height


def order_4_points(points):
    points_array = np.asarray(points, dtype=np.float32)
    sums = points_array.sum(axis=1)
    diffs = np.diff(points_array, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points_array[np.argmin(sums)]  # top-left: minimum x+y
    ordered[1] = points_array[np.argmin(diffs)]  # top-right: minimum y-x
    ordered[2] = points_array[np.argmax(sums)]  # bottom-right: maximum x+y
    ordered[3] = points_array[np.argmax(diffs)]  # bottom-left: maximum y-x
    return ordered


def warp_roi(src, box):
    points = order_4_points(box)
    width = int(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    if width < 10 or height < 5:
        return None
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(src, transform, (width, height))


def edge_density_roi(edges_roi):
    total = edges_roi.size
    return np.count_nonzero(edges_roi) / total if total > 0 else 0.0


def contrast_roi(gray_roi):
    return float(np.std(gray_roi.astype(np.float64)))


def brightness_roi(gray_roi):
    return float(np.mean(gray_roi.astype(np.float64)))


def has_horizontal_span(box) -> bool:
    points = np.asarray(box, dtype=np.float64)
    span_x = float(points[:, 0].max() - points[:, 0].min())
    span_y = float(points[:, 1].max() - points[:, 1].min())
    return span_x >= span_y


def label_components_bfs(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    stats: dict[int, tuple[int, int, int]] = {}
    current_label = 0

    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                current_label += 1
                labels[i, j] = current_label
                queue = deque([(i, j)])
                min_i = max_i = i
                min_j = max_j = j
                count = 0
                while queue:
                    current_i, current_j = queue.popleft()
                    count += 1
                    if current_i < min_i:
                        min_i = current_i
                    if current_i > max_i:
                        max_i = current_i
                    if current_j < min_j:
                        min_j = current_j
                    if current_j > max_j:
                        max_j = current_j
                    for delta_i, delta_j in neighbors:
                        next_i, next_j = current_i + delta_i, current_j + delta_j
                        if 0 <= next_i < rows and 0 <= next_j < cols \
                                and binary_image[next_i, next_j] == 255 and labels[next_i, next_j] == 0:
                            labels[next_i, next_j] = current_label
                            queue.append((next_i, next_j))
                stats[current_label] = (max_i - min_i + 1, max_j - min_j + 1, count)
    return stats


def count_characters_roi(gray_roi) -> int:
    roi_height, roi_width = gray_roi.shape
    if roi_height < 8 or roi_width < 20:
        return 0

    mean = gray_roi.mean()
    binary_image = np.where(gray_roi < mean, 255, 0).astype(np.uint8)

    stats = label_components_bfs(binary_image)

    min_height = MIN_CHAR_HEIGHT_RATIO * roi_height
    max_height = MAX_CHAR_HEIGHT_RATIO * roi_height
    count = 0
    for height, width, area in stats.values():
        if min_height <= height <= max_height and height >= width and area >= 10:
            count += 1
    return count


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
    n = len(points)
    if n < 3:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x0, y0 = min(xs), min(ys)
        width, height = max(xs) - x0, max(ys) - y0
        return (
            (x0 + width / 2, y0 + height / 2),
            (max(width, 1), max(height, 1)),
            0.0,
            [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)],
        )

    points_array = np.asarray(points, dtype=np.float64)
    centroid = points_array.mean(axis=0)
    centered = points_array - centroid

    cxx = np.mean(centered[:, 0] ** 2)
    cyy = np.mean(centered[:, 1] ** 2)
    cxy = np.mean(centered[:, 0] * centered[:, 1])
    covariance = np.array([[cxx, cxy], [cxy, cyy]])

    _, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, -1]
    angle_rad = np.arctan2(major_axis[1], major_axis[0])

    cosine, sine = np.cos(-angle_rad), np.sin(-angle_rad)
    rotation_to_axis = np.array([[cosine, -sine], [sine, cosine]])
    aligned = centered @ rotation_to_axis.T

    min_x, min_y = aligned.min(axis=0)
    max_x, max_y = aligned.max(axis=0)
    width, height = max_x - min_x, max_y - min_y

    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rotation_back = np.array([[cosine, -sine], [sine, cosine]])
    corners_local = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    corners = corners_local @ rotation_back.T + centroid
    center_local = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    center = center_local @ rotation_back.T + centroid

    angle_deg = float(np.degrees(angle_rad))
    return (
        (float(center[0]), float(center[1])),
        (float(max(width, 1)), float(max(height, 1))),
        angle_deg,
        [(float(x), float(y)) for x, y in corners],
    )


def filter_plate_candidates(contours, img_shape, edges, gray):
    img_height, img_width = img_shape[:2]
    img_area = img_height * img_width
    min_contour_len = contour_length_threshold(img_shape)
    min_height, min_width, max_height = plate_size_thresholds(img_shape)
    candidates = []

    for contour in contours:
        if len(contour) < min_contour_len:
            continue

        rect = min_area_rect_pca(contour)
        if rect is None:
            continue

        center, (width, height), angle, box = rect
        if REQUIRE_HORIZONTAL_SPAN and not has_horizontal_span(box):
            continue
        if height > width:
            width, height = height, width
        if height < min_height or width < min_width or height > max_height:
            continue

        aspect_ratio = width / height
        area = width * height
        area_ratio = area / img_area

        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
                and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue

        gray_roi = warp_roi(gray, box)
        edges_roi = warp_roi(edges, box)
        if gray_roi is None or edges_roi is None:
            continue

        density = edge_density_roi(edges_roi)
        contrast = contrast_roi(gray_roi)
        brightness = brightness_roi(gray_roi)
        n_chars = count_characters_roi(gray_roi)

        # Reject regions that clearly do not look like text.
        if n_chars < MIN_ABSOLUTE_CHARS or n_chars > MAX_ABSOLUTE_CHARS:
            continue

        ar_score = max(0, 1.0 - abs(aspect_ratio - TARGET_ASPECT_RATIO) / TARGET_ASPECT_RATIO)

        if density < MIN_DENSITY:
            dens_score = 0.0
        elif density <= DENSITY_PLATEAU:
            dens_score = 1.0
        else:
            dens_score = max(0, 1.0 - (density - DENSITY_PLATEAU) / DENSITY_FALLOFF)

        contrast_score = min(1.0, contrast / CONTRAST_NORMALIZATION)
        bright_score = min(1.0, max(0, (brightness - MIN_BRIGHTNESS) / BRIGHTNESS_RANGE))

        _, center_y = center
        pos_score = center_y / img_height

        area_score = max(0, 1.0 - abs(area_ratio - TARGET_AREA_RATIO) / AREA_RATIO_TOLERANCE)
        text_score = score_text(n_chars)

        score = (ar_score * WEIGHT_ASPECT_RATIO
                 + dens_score * WEIGHT_DENSITY
                 + contrast_score * WEIGHT_CONTRAST
                 + bright_score * WEIGHT_BRIGHTNESS
                 + pos_score * WEIGHT_POSITION
                 + area_score * WEIGHT_AREA
                 + text_score * WEIGHT_TEXT)

        candidates.append({
            "contour": contour,
            "center": center,
            "size": (width, height),
            "angle": angle,
            "box": box,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "density": density,
            "contrast": contrast,
            "brightness": brightness,
            "n_chars": n_chars,
            "score": score,
        })

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates
