import sys
from collections import deque

import cv2
from tkinter import Tk, filedialog

import numpy as np


def select_image():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir="dataset",
        title="Choose an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )
    root.destroy()
    return path or None


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        sys.exit(1)
    return image

def convolution(IS: np.ndarray, H: np.ndarray):
    IS = IS.astype(np.float64)
    H = H.astype(np.float64)
    w = H.shape[0]
    k = w // 2
    rows, cols = IS.shape
    ID = np.zeros((rows, cols), dtype=np.float64)
    for u in range(w):
        for v in range(w):
            ID[k:rows-k, k:cols-k] += (H[u, v] * IS[k + u - k:rows - k + u - k, k + v - k:cols - k + v - k])
    return ID


def gaussian_kernel(w: int, sigma: float):
    x0 = w // 2
    y0 = w // 2
    G = np.zeros((w, w), dtype=np.float64)
    for y in range(w):
        for x in range(w):
            G[y, x] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-((x-x0) ** 2 + (y-y0) ** 2) / (2.0 * sigma ** 2))
    G /= G.sum()
    return G


def gaussian_filter(img: np.ndarray, w: int = 5):
    sigma = w / 6.0
    G = gaussian_kernel(w, sigma)
    return convolution(img, G)


def sobel(img: np.ndarray):
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    fx = convolution(img, Hx)
    fy = convolution(img, Hy)
    modul = np.sqrt(fx ** 2 + fy ** 2)
    dir = np.rad2deg(np.arctan2(fy, fx))
    dir[dir < 0] += 360
    dir[dir > 360] -= 360
    return modul, dir

def directie(angle: float):
    a = angle % 360
    if (0 <= a < 22.5) or (157.5 <= a < 202.5) or (337.5 <= a < 360):
        return 0
    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5):
        return 1
    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5):
        return 2
    else:
        return 3

def keep_local_max(modul: np.ndarray, dir: np.ndarray):
    rows, cols = modul.shape
    result = np.zeros_like(modul)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            d = directie(dir[i, j])
            if d == 0:
                v1 = modul[i, j-1]; v2 = modul[i, j+1]
            elif d == 1:
                v1 = modul[i-1, j+1]; v2 = modul[i+1, j-1]
            elif d == 2:
                v1 = modul[i-1, j]; v2 = modul[i+1, j]
            else:
                v1 = modul[i-1, j-1]; v2 = modul[i+1, j+1]
            if modul[i, j] > v1 and modul[i, j] > v2:
                result[i, j] = modul[i, j]
    return result

def adaptive_binarization(modul_nms: np.ndarray, p: float = 0.1):
    normalized_module = np.clip(modul_nms / (4.0*np.sqrt(2.0)), 0, 255).astype(np.int32)
    histogram = np.zeros(256, dtype=np.int64)
    rows, cols = normalized_module.shape
    for i in range(rows):
        for j in range(cols):
            histogram[normalized_module[i, j]] += 1
    nr_non_muchie = int((1.0-p) * (rows * cols - histogram[0]))
    sum = 0
    adaptive_threshold = 1
    for i in range(1, 256):
        sum += histogram[i]
        if sum >= nr_non_muchie:
            adaptive_threshold = i
            break
    return adaptive_threshold, normalized_module

def histeresis(modul_norm: np.ndarray, adaptive_threshold: int, k: float = 0.4):
    rows, cols = modul_norm.shape
    high_threshold = adaptive_threshold
    low_threshold = k * high_threshold
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            value = modul_norm[i, j]
            if value > high_threshold:
                result[i, j] = 255
            elif value > low_threshold:
                result[i, j] = 128
    q = deque()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i, j] == 255:
                q.append((i, j))
    while q:
        ci, cj = q.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if result[ni, nj] == 128:
                        result[ni, nj] = 255
                        q.append((ni, nj))
    result[result == 128] = 0
    return result

def canny(image: cv2.typing.MatLike, gauss_w: int = 5, p: float = 0.1, k: float = 0.4):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    filtered = gaussian_filter(gray, gauss_w)
    module, dir = sobel(filtered)
    nms_module = keep_local_max(module, dir)
    adaptive_threshold, module_norm = adaptive_binarization(nms_module, p)
    edges = histeresis(module_norm, adaptive_threshold, k)
    return edges

def dilation(img: np.ndarray, element: np.ndarray):
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


def erosion(img: np.ndarray, element: np.ndarray):
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


def closing(img: np.ndarray, element: np.ndarray):
    return erosion(dilation(img, element), element)


def detect_contours(img_bin: np.ndarray):
    rows, cols = img_bin.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
    padded[1:rows+1, 1:cols+1] = img_bin
    visited = np.zeros_like(padded, dtype=bool)

    dirs = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]

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
                P1 = (nr, nc)
                d = dd
                break
        if P1 is None:
            return pts
        pts.append(P1)
        visited[P1[0], P1[1]] = True
        prev, curr = P0, P1
        for _ in range(2 * (rows+2) * (cols+2)):
            if d % 2 == 0:
                ss = (d + 7) % 8
            else:
                ss = (d + 6) % 8
            nxt = None
            for step in range(8):
                dd = (ss + step) % 8
                nr, nc = curr[0] + dirs[dd][0], curr[1] + dirs[dd][1]
                if padded[nr, nc] == 255:
                    nxt = (nr, nc)
                    d = dd
                    break
            if nxt is None:
                break
            if nxt == P1 and curr == P0:
                break
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
                        contours.append([(cc-1, cr-1) for cr, cc in contour])

    return contours

def aabb(points: list):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min

def rotate_points(points: list, angle_deg: float, cx: float, cy: float):
    # x' = cos(a) * (x - cx) - sin(a) * (y - cy) + cx
    # y' = sin(a) * (x - cx) + cos(a) * (y - cy) + cy
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        rx = cos_a * dx - sin_a * dy + cx
        ry = sin_a * dx + cos_a * dy + cy
        rotated.append((rx, ry))
    return rotated


def min_area_rectangle(points: list):
    """
    Gaseste dreptunghiul minim de arie care incadreaza un set de puncte.
      Calculeaza centrul punctelor (centroid)
      Pentru fiecare unghi de la 0 la 90 grade (pas de 2 grade):
         Roteste toate punctele cu acel unghi in jurul centroidului
         Calculeaza AABB (x_min, x_max, y_min, y_max) pe punctele rotite
         Calculeaza aria = width * height
      Retine unghiul care a dat aria cea mai mica
      Calculeaza cele 4 colturi ale AABB la unghiul optim
      Roteste colturile inapoi (cu -unghi) ca sa obtina pozitia reala
      (center, (width, height), angle_deg, box_4_points)
    """
    if len(points) < 3:
        x_min, y_min, w, h = aabb(points)
        cx = x_min + w / 2
        cy = y_min + h / 2
        box = [(x_min, y_min), (x_min + w, y_min),
               (x_min + w, y_min + h), (x_min, y_min + h)]
        return (cx, cy), (w, h), 0.0, box

    # centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    best_angle = 0
    best_area = float('inf')
    best_bbox = None

    # incearca fiecare unghi de la 0 la 90 grade, pas de 2
    for angle in range(0, 91, 2):
        rotated = rotate_points(points, angle, cx, cy)

        # AABB pe punctele rotite
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        w = x_max - x_min
        h = y_max - y_min
        area = w * h

        if area < best_area:
            best_area = area
            best_angle = angle
            best_bbox = (x_min, y_min, x_max, y_max)

    # acum avem unghiul optim — calculam colturile AABB in spatiul rotit
    x_min, y_min, x_max, y_max = best_bbox
    w = x_max - x_min
    h = y_max - y_min

    # cele 4 colturi ale AABB (in spatiul rotit)
    corners_rotated = [
        (x_min, y_min),  # top-left
        (x_max, y_min),  # top-right
        (x_max, y_max),  # bottom-right
        (x_min, y_max),  # bottom-left
    ]

    # roteste colturile inapoi cu -best_angle ca sa obtinem pozitia reala
    corners_real = rotate_points(corners_rotated, -best_angle, cx, cy)

    # centrul dreptunghiului
    center_rotated = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    center_real = rotate_points([center_rotated], -best_angle, cx, cy)[0]

    return center_real, (w, h), float(best_angle), corners_real


def filtreaza_placute(contours: list, img_shape: tuple):
    # pentru fiecare contur: calculeaza minAreaRect (AABB + brute force rotation)
    # apoi filtreaza dupa aspect ratio, arie, soliditate
    # sorteaza dupa scor
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    candidati = []

    for contour in contours:
        if len(contour) < 15:
            continue

        rect = min_area_rectangle(contour)
        if rect is None:
            continue

        center, (w, h), angle, box = rect

        if h > w:
            w, h = h, w

        if h < 15 or w < 40 or h > 150:
            continue

        aspect_ratio = w / h
        area = w * h
        area_ratio = area / img_area

        if not (2.5 <= aspect_ratio <= 5.5 and 0.002 <= area_ratio <= 0.05):
            continue

        # soliditate = aria conturului (Shoelace) / aria dreptunghiului
        n = len(contour)
        contour_area = 0.0
        for ci in range(n):
            cj = (ci + 1) % n
            contour_area += contour[ci][0] * contour[cj][1]
            contour_area -= contour[cj][0] * contour[ci][1]
        contour_area = abs(contour_area) / 2.0
        soliditate = contour_area / max(area, 1)

        if soliditate < 0.3:
            continue

        ar_score = 1.0 - abs(aspect_ratio - 4.7) / 4.7
        solid_score = soliditate
        _, cy_pos = center
        pos_score = cy_pos / img_shape[0]
        ar_area_score = 1.0 - abs(area_ratio - 0.01) / 0.05

        score = (ar_score * 3.0 +
                 solid_score * 2.0 +
                 pos_score * 1.5 +
                 ar_area_score * 1.0)

        candidati.append({
            "contour": contour,
            "center": center,
            "size": (w, h),
            "angle": angle,
            "box": box,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "soliditate": soliditate,
            "score": score,
        })

    candidati.sort(key=lambda c: c["score"], reverse=True)
    return candidati

def sort_4_points(pts):
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).flatten()
    o = np.zeros((4, 2), dtype=np.float32)
    o[0] = arr[np.argmin(s)]
    o[1] = arr[np.argmin(d)]
    o[2] = arr[np.argmax(s)]
    o[3] = arr[np.argmax(d)]
    return o


def decupeaza_placuta(image: np.ndarray, box):
    src = sort_4_points(box)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 30 or h < 10:
        return None
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))



def detect_plate(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Canny")
    edges = canny(gray, gauss_w=5, p=0.1, k=0.4)
    cv2.imshow("Canny", edges)

    print("Dilation + closing")
    elem_dilat = np.ones((1, 9), dtype=np.uint8)
    dilated = dilation(edges, elem_dilat)
    elem_close = np.ones((3, 11), dtype=np.uint8)
    morphed = closing(dilated, elem_close)
    cv2.imshow("Dilation + closing", morphed)

    print("Detecting contours")
    contours = detect_contours(morphed)
    print(f"Contururi: {len(contours)}")

    img_c = image.copy()
    for contour in contours:
        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_c, [pts], True, (0, 255, 0), 1)
    cv2.imshow("Contururi", img_c)

    print("Filtrare geometrica")
    candidati = filtreaza_placute(contours, image.shape)
    print(f"Candidati: {len(candidati)}")

    for i, c in enumerate(candidati[:5]):
        print(f"#{i+1}: SCOR={c['score']:.2f}, AR={c['aspect_ratio']:.2f}, "
              f"solid={c['soliditate']:.2f}, arie={c['area_ratio']:.4f}, "
              f"size={c['size'][0]:.0f}x{c['size'][1]:.0f}, unghi={c['angle']:.0f}°")

    img_cand = image.copy()
    for i, c in enumerate(candidati[:5]):
        box_np = np.array(c["box"], dtype=np.int32)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.polylines(img_cand, [box_np], True, color, 2)
        cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.putText(img_cand, f"#{i+1} AR={c['aspect_ratio']:.1f}",
                    (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow("Candidati", img_cand)

    if not candidati:
        print("Niciun candidat gasit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    best = candidati[0]
    print(f"Best: SCOR={best['score']:.2f}, AR={best['aspect_ratio']:.2f}, "
          f"unghi={best['angle']:.0f}°")

    plate = decupeaza_placuta(image, best["box"])
    if plate is not None:
        cv2.imshow("Placuta decupata", plate)

        img_final = image.copy()
        box_np = np.array(best["box"], dtype=np.int32)
        cv2.polylines(img_final, [box_np], True, (0, 255, 0), 2)
        cv2.imshow("Rezultat", img_final)
    else:
        print("Placuta de inmatriculare n-a putut fi detectata")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = select_image()
    if image_path is None:
        print("No image selected.")
        sys.exit(0)
    image = load_image(image_path)
    detect_plate(image)


if __name__ == "__main__":
    main()