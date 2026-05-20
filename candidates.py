from collections import deque
import cv2
import numpy as np

# calculeaza cate puncte trebuie sa aiba un contur ca sa fie luat in considerare
# intr-o imagine de 1000 pixeli, conturul e minim 15 puncte
def contour_length_threshold(img_shape) -> int:
    min_dim = min(img_shape[0], img_shape[1])
    return max(10, int(round(min_dim * 0.010)))

#ca functia anterioara, pnntru inaltimea si latimea minima / maxima a unei placute
def plate_size_thresholds(img_shape):
    h, w = img_shape[:2]
    min_h = max(8.0, h * 0.008)
    min_w = max(24.0, w * 0.025)
    max_h = max(min_h, h * 0.25)
    return min_h, min_w, max_h

# sorteaza colturile: top-left, top_right, bottom_right, bottom_left
# opencv asteapta colturile intr-o ordine fixata
def order_4_points(points):
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(sums)],
        pts[np.argmin(diffs)],
        pts[np.argmax(sums)],
        pts[np.argmax(diffs)],
    ], dtype=np.float32)

#calculeaza latimea ca distanta intre colturile de sus
#calculeaza inaltimea ca distanta intre colturile din stanga / dreapta
#warpPerspective indreapta regiunea intr-un dreptunghi w x h
def warp_roi(src, box):
    pts = order_4_points(box)
    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    if w < 10 or h < 5:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(src, cv2.getPerspectiveTransform(pts, dst), (w, h))

#verifica daca dreptunghiul e mai lat decat inalt
def has_horizontal_span(box) -> bool:
    pts = np.asarray(box, dtype=np.float64)
    sx = pts[:, 0].max() - pts[:, 0].min()
    sy = pts[:, 1].max() - pts[:, 1].min()
    return sx >= sy

#converteste din BGR in HSV si creeaza o masca binara
#unde pixelii albi (saturatie scazuta, valoare mare) devin 255
#ceilalti 0
# pt ca placutele de romania au mult alb, deci alb => semn bun
def compute_white_mask(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    low = np.array((0, 0, 120), np.uint8)
    high = np.array((180, 100, 255), np.uint8)
    return cv2.inRange(hsv, low, high)


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
                min_i = max_i = i
                min_j = max_j = j
                count = 0
                while Q:
                    qi, qj = Q.popleft()
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

# folosit la scoring
# binarizeaza ROI (pixelii sub medie devin albi) => text
# numara cate componente trec filtrul de caractere
def count_characters_roi(gray_roi) -> int:
    h, w = gray_roi.shape
    if h < 8 or w < 20:
        return 0
    binary = np.where(gray_roi < gray_roi.mean(), 255, 0).astype(np.uint8)
    min_h = 0.30 * h
    max_h = 0.95 * h
    count = 0
    for min_i, min_j, max_i, max_j, area in label_components_bfs(binary):
        height = max_i - min_i + 1
        width = max_j - min_j + 1
        if min_h <= height <= max_h and height >= width and area >= 10:
            count += 1
    return count

# scor in functie de cate caractere sunt detectate
def score_text(n_chars: int) -> float:
    if n_chars < 3 or n_chars > 14:
        return 0.0
    if 6 <= n_chars <= 8:
        return 1.0
    if n_chars < 6:
        return (n_chars - 3) / 3
    return (14 - n_chars) / 6


def min_area_rect_pca(points):
    pts = np.asarray(points, dtype=np.float64)
    # < 3 puncte => nu putem face PCA
    # deci returnam un dreptunghi aliniat simplu
    if len(pts) < 3:
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        return ((x0 + w / 2, y0 + h / 2), (w, h), 0.0, corners)

    # centroid = media pe fiecare axa
    centroid = pts.mean(axis=0)
    # mutam originea in centroid
    centered = pts - centroid

    # matricea de covarianta = cat de raspandite sunt punctele si in ce directie
    cov = (centered.T @ centered) / len(pts)

    # vectorii proprii
    # ne dau unghiul de inclinare al norului de punct
    _, eigvecs = np.linalg.eigh(cov)
    angle = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))

    # rotim incat axa principala sa devina orizontala
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    aligned = centered @ rot.T

    # pe punctele rotite, dreptunghiul = min/max pe fiecare axa
    min_xy = aligned.min(axis=0)
    max_xy = aligned.max(axis=0)
    w = float(max(max_xy[0] - min_xy[0], 1))
    h = float(max(max_xy[1] - min_xy[1], 1))

    # rotire inapoi si translatie la pozitia reala
    rot_back = rot.T

    # cele 4 colturi ale dreptunghiului
    local = np.array([[min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
                      [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]]])

    # rotim inapoi la unghiul original si translatam la centroid
    corners = local @ rot_back.T + centroid
    center = (min_xy + max_xy) / 2 @ rot_back.T + centroid

    # returnam centru, dimensiuni, unghi, colturi
    return (
        (float(center[0]), float(center[1])),
        (w, h),
        float(np.degrees(angle)),
        [(float(x), float(y)) for x, y in corners],
    )


def _score(aspect, density, contrast, brightness, n_chars, white_ratio):
    ar = max(0.0, 1.0 - abs(aspect - 4.7) / 4.7)
    if density < 0.02:
        dens = 0.0
    elif density <= 0.35:
        dens = 1.0
    else:
        dens = max(0.0, 1.0 - (density - 0.35) / 0.30)
    contr = min(1.0, contrast / 60.0)
    bright = min(1.0, max(0.0, (brightness - 80.0) / 120.0))
    text = score_text(n_chars)
    white = min(1.0, white_ratio / 0.60)
    return ar * 2.5 + dens * 2.0 + contr * 3.0 + bright * 2.0 + text * 2.5 + white * 2.0

    # pipeline
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
        if not has_horizontal_span(box):
            continue
        if h > w:
            w, h = h, w
        if h < min_height or w < min_width or h > max_height:
            continue

        aspect = w / h
        area_ratio = (w * h) / img_area
        if not (2.0 <= aspect <= 6.5 and 0.0005 <= area_ratio <= 0.10):
            continue

        gray_roi = warp_roi(gray, box)
        edges_roi = warp_roi(edges, box)
        white_roi = warp_roi(white_mask, box)
        if gray_roi is None or edges_roi is None or white_roi is None or white_roi.size == 0:
            continue

        white_ratio = float(np.count_nonzero(white_roi)) / white_roi.size
        if white_ratio < 0.20:
            continue

        density = np.count_nonzero(edges_roi) / edges_roi.size if edges_roi.size else 0.0
        contrast = float(gray_roi.std())
        brightness = float(gray_roi.mean())
        n_chars = count_characters_roi(gray_roi)
        if n_chars < 3 or n_chars > 14:
            continue

        score = _score(aspect, density, contrast, brightness, n_chars, white_ratio)
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

# cand mai multe contururi detecteaza aceeasi placuta
# ajung cu candidati suprapusi
# nms_candidates parcurge candidatii de la cel mai bun scor in jos si
# suprima candidatii ulteriori care se suprapun cu IoU > 0.4
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
