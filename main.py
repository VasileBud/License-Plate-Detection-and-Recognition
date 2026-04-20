"""
Detectie placuta de inmatriculare.
  1. Canny
  2. Closing minimal (3x3) — doar inchide micro-goluri
  3. Detectia de conturur
  4. minAreaRect (AABB + Brute Force Rotation)
  5. Filtrare geometrica + scoring cu
     - densitate muchii (pe edges)
     - contrast grayscale (deviatie standard pe imaginea originala)
     - aspect ratio, pozitie, arie
  6. warpPerspective pt decupare
"""

import sys
from collections import deque

import cv2
from tkinter import Tk, filedialog

import numpy as np

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

def convolution(IS, H):
    IS = IS.astype(np.float64)
    H = H.astype(np.float64)
    w = H.shape[0]
    k = w // 2
    rows, cols = IS.shape
    ID = np.zeros((rows, cols), dtype=np.float64)
    for u in range(w):
        for v in range(w):
            ID[k:rows - k, k:cols - k] += (H[u, v] * IS[k + u - k:rows - k + u - k, k + v - k:cols - k + v - k])
    return ID


def gaussian_kernel(w, sigma):
    x0 = w // 2
    G = np.zeros((w, w), dtype=np.float64)
    for y in range(w):
        for x in range(w):
            G[y, x] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-((x - x0) ** 2 + (y - x0) ** 2) / (2.0 * sigma ** 2))
    G /= G.sum()
    return G


def gaussian_filter(img, w=5):
    return convolution(img, gaussian_kernel(w, w / 6.0))


def sobel(img):
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    fx = convolution(img, Hx)
    fy = convolution(img, Hy)
    modul = np.sqrt(fx ** 2 + fy ** 2)
    dir = np.rad2deg(np.arctan2(fy, fx))
    dir[dir < 0] += 360
    return modul, dir


def directie(angle):
    a = angle % 360
    if (0 <= a < 22.5) or (157.5 <= a < 202.5) or (337.5 <= a < 360):
        return 0
    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5):
        return 1
    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5):
        return 2
    else:
        return 3


def keep_local_max(modul, dir):
    rows, cols = modul.shape
    result = np.zeros_like(modul)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            d = directie(dir[i, j])
            if d == 0:
                v1, v2 = modul[i, j - 1], modul[i, j + 1]
            elif d == 1:
                v1, v2 = modul[i - 1, j + 1], modul[i + 1, j - 1]
            elif d == 2:
                v1, v2 = modul[i - 1, j], modul[i + 1, j]
            else:
                v1, v2 = modul[i - 1, j - 1], modul[i + 1, j + 1]
            if modul[i, j] > v1 and modul[i, j] > v2:
                result[i, j] = modul[i, j]
    return result


def adaptive_binarization(modul_nms, p=0.1):
    nm = np.clip(modul_nms / (4.0 * np.sqrt(2.0)), 0, 255).astype(np.int32)
    hist = np.zeros(256, dtype=np.int64)
    rows, cols = nm.shape
    for i in range(rows):
        for j in range(cols):
            hist[nm[i, j]] += 1
    nr = int((1.0 - p) * (rows * cols - hist[0]))
    s = 0
    thr = 1
    for i in range(1, 256):
        s += hist[i]
        if s >= nr:
            thr = i
            break
    return thr, nm


def histeresis(mn, thr, k=0.4):
    rows, cols = mn.shape
    lo = k * thr
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if mn[i, j] > thr:
                result[i, j] = 255
            elif mn[i, j] > lo:
                result[i, j] = 128
    q = deque()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if result[i, j] == 255:
                q.append((i, j))
    while q:
        ci, cj = q.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < rows and 0 <= nj < cols and result[ni, nj] == 128:
                    result[ni, nj] = 255
                    q.append((ni, nj))
    result[result == 128] = 0
    return result


def canny(image, gauss_w=5, p=0.1, k=0.4):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    filtered = gaussian_filter(gray, gauss_w)
    modul, dir = sobel(filtered)
    nms = keep_local_max(modul, dir)
    thr, mn = adaptive_binarization(nms, p)
    return histeresis(mn, thr, k)

def dilatare(img, element):
    rows, cols = img.shape
    kh, kw = element.shape
    ph, pw = kh // 2, kw // 2
    result = np.zeros_like(img)
    for i in range(ph, rows - ph):
        for j in range(pw, cols - pw):
            region = img[i - ph:i + ph + 1, j - pw:j + pw + 1]
            if np.any(region[element == 1] == 255):
                result[i, j] = 255
    return result


def eroziune(img, element):
    rows, cols = img.shape
    kh, kw = element.shape
    ph, pw = kh // 2, kw // 2
    result = np.zeros_like(img)
    for i in range(ph, rows - ph):
        for j in range(pw, cols - pw):
            region = img[i - ph:i + ph + 1, j - pw:j + pw + 1]
            if np.all(region[element == 1] == 255):
                result[i, j] = 255
    return result


def closing(img, element):
    return eroziune(dilatare(img, element), element)


def find_contours_lab6(img_bin):
    rows, cols = img_bin.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
    padded[1:rows + 1, 1:cols + 1] = img_bin
    visited = np.zeros_like(padded, dtype=bool)
    dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    contours = []

    def trace(sr, sc):
        P0 = (sr, sc)
        pts = [P0]
        visited[sr, sc] = True
        d = 7
        ss = (d + 6) % 8
        P1 = None
        for step in range(8):
            dd = (ss + step) % 8
            nr, nc = sr + dirs[dd][0], sc + dirs[dd][1]
            if padded[nr, nc] == 255:
                P1 = (nr, nc);
                d = dd;
                break
        if P1 is None: return pts
        pts.append(P1)
        visited[P1[0], P1[1]] = True
        prev, curr = P0, P1
        for _ in range(2 * (rows + 2) * (cols + 2)):
            ss = (d + 7) % 8 if d % 2 == 0 else (d + 6) % 8
            nxt = None
            for step in range(8):
                dd = (ss + step) % 8
                nr, nc = curr[0] + dirs[dd][0], curr[1] + dirs[dd][1]
                if padded[nr, nc] == 255:
                    nxt = (nr, nc);
                    d = dd;
                    break
            if nxt is None: break
            if nxt == P1 and curr == P0: break
            pts.append(nxt)
            visited[nxt[0], nxt[1]] = True
            prev, curr = curr, nxt
        return pts

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if padded[r, c] == 255 and not visited[r, c]:
                if padded[r, c - 1] == 0:
                    contour = trace(r, c)
                    if len(contour) >= 15:
                        contours.append([(cc - 1, cr - 1) for cr, cc in contour])
    return contours

def roteste_puncte(points, angle_deg, cx, cy):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return [(c * (x - cx) - s * (y - cy) + cx, s * (x - cx) + c * (y - cy) + cy)
            for x, y in points]


def min_area_rect_bf(points):
    if len(points) < 3:
        xs = [p[0] for p in points];
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        return (x0 + w / 2, y0 + h / 2), (max(w, 1), max(h, 1)), 0.0, \
            [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    best_angle, best_area, best_bbox = 0, float('inf'), None

    for angle in range(0, 91, 2):
        rot = roteste_puncte(points, angle, cx, cy)
        xs = [p[0] for p in rot];
        ys = [p[1] for p in rot]
        xn, xx, yn, yx = min(xs), max(xs), min(ys), max(ys)
        area = (xx - xn) * (yx - yn)
        if area < best_area:
            best_area = area;
            best_angle = angle
            best_bbox = (xn, yn, xx, yx)

    xn, yn, xx, yx = best_bbox
    w, h = xx - xn, yx - yn
    corners = roteste_puncte([(xn, yn), (xx, yn), (xx, yx), (xn, yx)], -best_angle, cx, cy)
    center = roteste_puncte([((xn + xx) / 2, (yn + yx) / 2)], -best_angle, cx, cy)[0]
    return center, (max(w, 1), max(h, 1)), float(best_angle), corners


def get_roi_bbox(box, img_shape):
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    y0 = max(0, min(ys))
    y1 = min(img_shape[0] - 1, max(ys))
    x0 = max(0, min(xs))
    x1 = min(img_shape[1] - 1, max(xs))
    return y0, y1, x0, x1


def densitate_muchii(edges, box):
    y0, y1, x0, x1 = get_roi_bbox(box, edges.shape)
    if y1 <= y0 or x1 <= x0: return 0.0
    roi = edges[y0:y1 + 1, x0:x1 + 1]
    total = roi.shape[0] * roi.shape[1]
    return np.count_nonzero(roi) / total if total > 0 else 0.0


def contrast_grayscale(gray, box):
    y0, y1, x0, x1 = get_roi_bbox(box, gray.shape)
    if y1 <= y0 or x1 <= x0: return 0.0
    roi = gray[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    return np.std(roi)


def luminozitate_medie(gray, box):
    y0, y1, x0, x1 = get_roi_bbox(box, gray.shape)
    if y1 <= y0 or x1 <= x0: return 0.0
    roi = gray[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    return np.mean(roi)


def filtreaza_placute(contours, img_shape, edges, gray):
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    candidati = []

    for contour in contours:
        if len(contour) < 15:
            continue

        rect = min_area_rect_bf(contour)
        if rect is None: continue

        center, (w, h), angle, box = rect
        if h > w: w, h = h, w
        if h < 10 or w < 30 or h > 200: continue

        aspect_ratio = w / h
        area = w * h
        area_ratio = area / img_area

        if not (2.5 <= aspect_ratio <= 5.5 and 0.001 <= area_ratio <= 0.06):
            continue

        density = densitate_muchii(edges, box)
        contrast = contrast_grayscale(gray, box)
        brightness = luminozitate_medie(gray, box)

        ar_score = max(0, 1.0 - abs(aspect_ratio - 4.7) / 4.7)

        if density < 0.03:
            dens_score = 0.0
        elif density <= 0.35:
            dens_score = 1.0
        else:
            dens_score = max(0, 1.0 - (density - 0.35) / 0.30)

        contrast_score = min(1.0, contrast / 60.0)

        bright_score = min(1.0, max(0, (brightness - 80) / 120.0))

        _, cy_pos = center
        pos_score = cy_pos / img_h

        area_score = max(0, 1.0 - abs(area_ratio - 0.008) / 0.05)

        score = (ar_score * 2.5 +
                 dens_score * 2.0 +
                 contrast_score * 3.0 +  # contrastul e cel mai influent
                 bright_score * 2.0 +  # luminozitatea separa placuta de grila
                 pos_score * 1.0 +
                 area_score * 0.5)

        candidati.append({
            "contour": contour,
            "center": center,
            "size": (w, h),
            "angle": angle,
            "box": box,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "density": density,
            "contrast": contrast,
            "brightness": brightness,
            "score": score,
        })

    candidati.sort(key=lambda c: c["score"], reverse=True)
    return candidati

def ordoneaza_4_puncte(pts):
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).flatten()
    o = np.zeros((4, 2), dtype=np.float32)
    o[0] = arr[np.argmin(s)]
    o[1] = arr[np.argmin(d)]
    o[2] = arr[np.argmax(s)]
    o[3] = arr[np.argmax(d)]
    return o


def decupeaza_placuta(image, box):
    src = ordoneaza_4_puncte(box)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 30 or h < 10: return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def detect_plate(image):
    print("DETECTIE PLACUTA DE INMATRICULARE")

    max_w = 800
    scale = 1.0
    if image.shape[1] > max_w:
        scale = max_w / image.shape[1]
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()

    print(f"{image.shape[1]}x{image.shape[0]} -> {work.shape[1]}x{work.shape[0]}")
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    print("Canny...")
    edges = canny(gray, gauss_w=5, p=0.1, k=0.4)
    cv2.imshow("1. Canny", edges)

    print("Closing 3x3")
    elem = np.ones((3, 3), dtype=np.uint8)
    closed = closing(edges, elem)
    cv2.imshow("2. Closing 3x3", closed)

    print("Detecting contours")
    contours = find_contours_lab6(closed)
    print(f"Contururi: {len(contours)}")

    img_c = work.copy()
    for c in contours:
        pts = np.array(c, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_c, [pts], True, (0, 255, 0), 1)
    cv2.imshow("3. Contururi", img_c)

    print("Filtrare + scoring")
    candidati = filtreaza_placute(contours, work.shape, edges, gray)
    print(f"Candidati: {len(candidati)}")

    for i, c in enumerate(candidati[:5]):
        print(f"#{i + 1}: SCOR={c['score']:.2f}, AR={c['aspect_ratio']:.2f}, "
              f"dens={c['density']:.2f}, contrast={c['contrast']:.0f}, "
              f"bright={c['brightness']:.0f}, size={c['size'][0]:.0f}x{c['size'][1]:.0f}")

    img_cand = work.copy()
    for i, c in enumerate(candidati[:5]):
        box_np = np.array(c["box"], dtype=np.int32)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.polylines(img_cand, [box_np], True, color, 2)
        cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(img_cand, f"#{i + 1} s={c['score']:.1f}",
                    (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imshow("4. Candidati", img_cand)

    if not candidati:
        print("Niciun candidat!")
        cv2.waitKey(0);
        cv2.destroyAllWindows()
        return

    best = candidati[0]
    print(f"Best: SCOR={best['score']:.2f}, AR={best['aspect_ratio']:.2f}, "
          f"contrast={best['contrast']:.0f}, bright={best['brightness']:.0f}")

    box_orig = [(x / scale, y / scale) for x, y in best["box"]]
    plate = decupeaza_placuta(image, box_orig)
    if plate is not None:
        cv2.imshow("5. Placuta decupata", plate)
        img_final = image.copy()
        box_np = np.array(box_orig, dtype=np.int32)
        cv2.polylines(img_final, [box_np], True, (0, 255, 0), 3)
        cv2.imshow("6. Rezultat", img_final)
    else:
        print("Prea mica.")

    cv2.waitKey(0);
    cv2.destroyAllWindows()


def main():
    path = select_image()
    if not path:
        print("No image.");
        sys.exit(0)
    detect_plate(load_image(path))


if __name__ == "__main__":
    main()