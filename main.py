import sys
from collections import deque

import cv2
from tkinter import Tk, filedialog

import numpy as np


def select_image() -> str | None:
    """Open a file dialog and return the selected image path, or None."""
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
    """Load an image from disk or exit with an error."""
    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        sys.exit(1)
    return image


def show_results(image: cv2.typing.MatLike, gray: cv2.typing.MatLike, edges: cv2.typing.MatLike) -> None:
    """Display the original, grayscale, and edge-detected images."""
    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolution(IS: np.ndarray, H: np.ndarray) -> np.ndarray:
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

def gaussian_kernel(w: int, sigma: float) -> np.ndarray:
    x0 = w // 2
    y0 = w // 2
    G = np.zeros((w, w), dtype=np.float64)

    for y in range(w):
        for x in range(w):
            G[y, x] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-((x-x0) ** 2 + (y-y0) ** 2) / (2.0 * sigma ** 2))

    G /= G.sum()

    return G

def gaussian_filter(img: np.ndarray, w: int = 5) -> np.ndarray:
    sigma = w / 6.0
    G = gaussian_kernel(w, sigma)
    return convolution(img, G)

def sobel(img: np.ndarray):
    Hx = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64)
    Hy = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]], dtype=np.float64)

    fx = convolution(img, Hx)
    fy = convolution(img, Hy)

    modul = np.sqrt(fx ** 2 + fy ** 2)

    dir = np.rad2deg(np.arctan2(fy, fx))
    dir[dir < 0] += 360
    dir[dir > 360] -= 360

    return modul, dir

def directie(angle: float) -> int:
    a = angle % 360
    if (0 <= a < 22.5) or (157.5 <= a < 202.5) or (337.5 <= a < 360):
        return 0
    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5):
        return 1
    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5):
        return 2
    else:
        return 3

def keep_local_max(modul: np.ndarray, dir: np.ndarray) -> np.ndarray:
    rows, cols = modul.shape
    result = np.zeros_like(modul)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            d = directie(dir[i, j])
            if d == 0:
                v1 = modul[i, j-1]
                v2 = modul[i, j+1]
            elif d == 1:
                v1 = modul[i-1, j+1]
                v2 = modul[i+1, j-1]
            elif d == 2:
                v1 = modul[i-1, j]
                v2 = modul[i+1, j]
            else:
                v1 = modul[i-1, j-1]
                v2 = modul[i+1, j+1]
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

def histeresis(modul_norm: np.ndarray, adaptive_threshold: int, k: float = 0.4) -> np.ndarray:
    strong_edge = 255
    weak_edge = 128
    non_edge = 0

    rows, cols = modul_norm.shape
    high_threshold = adaptive_threshold
    low_threshold = k * high_threshold

    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            value = modul_norm[i, j]
            if value > high_threshold:
                result[i, j] = strong_edge
            elif value > low_threshold:
                result[i, j] = weak_edge

    q = deque()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i, j] == strong_edge:
                q.append((i, j))

    while q:
        ci, cj = q.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if result[ni, nj] == weak_edge:
                        result[ni, nj] = strong_edge
                        q.append((ni, nj))
    result[result == weak_edge] = non_edge

    return result

def canny(image: cv2.typing.MatLike, gauss_w: int = 5, p: float = 0.1, k: float = 0.4) -> np.ndarray:
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

def main():
    image_path = select_image()
    if image_path is None:
        print("No image selected.")
        sys.exit(0)

    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(image, gauss_w=5, p=0.1, k=0.4)
    show_results(image, gray, edges)


if __name__ == "__main__":
    main()
