from collections import deque
import cv2
import numpy as np


def convolution(input_signal, kernel):
    input_signal = input_signal.astype(np.float64)
    kernel = kernel.astype(np.float64)
    width = kernel.shape[0]
    k = width // 2
    rows, cols = input_signal.shape
    output = np.zeros((rows, cols), dtype=np.float64)
    for i in range(k, rows - k):
        for j in range(k, cols - k):
            s = 0.0
            for u in range(width):
                for v in range(width):
                    s += kernel[u, v] * input_signal[i + u - k, j + v - k]
            output[i, j] = s
    return output


def gaussian_kernel(width, sigma):
    # G(x,y) = (1 / (2*pi*sigma^2)) * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    center = width // 2
    kernel = np.zeros((width, width), dtype=np.float64)
    for y in range(width):
        for x in range(width):
            kernel[y, x] = (1.0 / (2.0 * np.pi * sigma * sigma)) * np.exp(
                -((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma * sigma)
            )
    # normalizam ca suma ponderilor sa fie 1
    # altfel convolutia ar afecta luminiozitatea
    total = 0.0
    for y in range(width):
        for x in range(width):
            total += kernel[y, x]
    for y in range(width):
        for x in range(width):
            kernel[y, x] /= total
    return kernel


def gaussian_filter(img, width=5):
    return convolution(img, gaussian_kernel(width, width / 6.0))


def sobel(img):
    # detectia variatiilor orizontale si verticale
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    grad_x = convolution(img, kernel_x)
    grad_y = convolution(img, kernel_y)
    rows, cols = img.shape
    magnitude = np.zeros((rows, cols), dtype=np.float64)
    direction = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            magnitude[i, j] = np.sqrt(grad_x[i, j] ** 2 + grad_y[i, j] ** 2)
            angle = np.rad2deg(np.arctan2(grad_y[i, j], grad_x[i, j]))
            if angle < 0:
                angle += 360
            direction[i, j] = angle
    return magnitude, direction


def direction_bin(angle):
    angle = angle % 360
    if (0 <= angle < 22.5) or (157.5 <= angle < 202.5) or (337.5 <= angle < 360):
        return 0
    if (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
        return 1
    if (67.5 <= angle < 112.5) or (247.5 <= angle < 292.5):
        return 2
    return 3

#subtiem muchiile la un singur pixel grosime
#verificand cei 2 vecini de-a lungul directiei
def keep_local_max(magnitude, direction):
    rows, cols = magnitude.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            d = direction_bin(direction[i, j])
            if d == 0:
                v1 = magnitude[i, j - 1]
                v2 = magnitude[i, j + 1]
            elif d == 1:
                v1 = magnitude[i - 1, j + 1]
                v2 = magnitude[i + 1, j - 1]
            elif d == 2:
                v1 = magnitude[i - 1, j]
                v2 = magnitude[i + 1, j]
            else:
                v1 = magnitude[i - 1, j - 1]
                v2 = magnitude[i + 1, j + 1]
            if magnitude[i, j] > v1 and magnitude[i, j] > v2:
                result[i, j] = magnitude[i, j]
    return result

#calculul pragului folosind histograma magnitudinilor
#pastram 10% din pixelii non-zero ca muchii sigure
def adaptive_binarization(nms_magnitude, p=0.1):
    rows, cols = nms_magnitude.shape
    normalized = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            v = nms_magnitude[i, j] / (4.0 * np.sqrt(2.0))
            if v < 0:
                v = 0
            if v > 255:
                v = 255
            normalized[i, j] = int(v)

    hist = np.zeros(256, dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            hist[normalized[i, j]] += 1

    no_non_edge = int((1.0 - p) * (rows * cols - hist[0]))

    running_sum = 0
    threshold = 1
    for i in range(1, 256):
        running_sum += hist[i]
        if running_sum >= no_non_edge:
            threshold = i
            break
    return threshold, normalized

#pixel > high => muchie sigura
#pixel < low => eliminati
#pixeli intre low si high sunt candidati
#candidatii conectati la muchii sigure sunt promovati
def hysteresis(magnitude_map, high_threshold, k=0.4):
    rows, cols = magnitude_map.shape
    low_threshold = k * high_threshold

    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if magnitude_map[i, j] > high_threshold:
                result[i, j] = 255
            elif magnitude_map[i, j] > low_threshold:
                result[i, j] = 128

    queue = deque()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if result[i, j] == 255:
                queue.append((i, j))
    while queue:
        ci, cj = queue.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < rows and 0 <= nj < cols and result[ni, nj] == 128:
                    result[ni, nj] = 255
                    queue.append((ni, nj))

    for i in range(rows):
        for j in range(cols):
            if result[i, j] == 128:
                result[i, j] = 0
    return result


def canny(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    filtered = gaussian_filter(gray, 5)
    magnitude, direction = sobel(filtered)
    nms = keep_local_max(magnitude, direction)
    threshold, normalized = adaptive_binarization(nms, 0.1)
    return hysteresis(normalized, threshold, 0.4)
