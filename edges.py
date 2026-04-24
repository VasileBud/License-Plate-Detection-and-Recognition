from collections import deque
import cv2
import numpy as np


def convolution(input_signal, kernel):
    input_signal = input_signal.astype(np.float64)
    kernel = kernel.astype(np.float64)
    width = kernel.shape[0]
    half = width // 2
    rows, cols = input_signal.shape
    output = np.zeros((rows, cols), dtype=np.float64)
    for u in range(width):
        for v in range(width):
            output[half:rows - half, half:cols - half] += (
                kernel[u, v]
                * input_signal[half + u - half:rows - half + u - half, half + v - half:cols - half + v - half]
            )
    return output


def gaussian_kernel(width, sigma):
    center = width // 2
    kernel = np.zeros((width, width), dtype=np.float64)
    for y in range(width):
        for x in range(width):
            kernel[y, x] = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(
                -((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma ** 2)
            )
    kernel /= kernel.sum()
    return kernel


def gaussian_filter(img, width=5):
    return convolution(img, gaussian_kernel(width, width / 6.0))


def sobel(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    grad_x = convolution(img, kernel_x)
    grad_y = convolution(img, kernel_y)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.rad2deg(np.arctan2(grad_y, grad_x))
    direction[direction < 0] += 360
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


def keep_local_max(magnitude, direction):
    rows, cols = magnitude.shape
    result = np.zeros_like(magnitude)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction_code = direction_bin(direction[i, j])
            if direction_code == 0:
                v1, v2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction_code == 1:
                v1, v2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif direction_code == 2:
                v1, v2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                v1, v2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] > v1 and magnitude[i, j] > v2:
                result[i, j] = magnitude[i, j]
    return result


def adaptive_binarization(nms_magnitude, p=0.1):
    normalized_magnitude = np.clip(nms_magnitude / (4.0 * np.sqrt(2.0)), 0, 255).astype(np.int32)
    hist = np.zeros(256, dtype=np.int64)
    rows, cols = normalized_magnitude.shape
    for i in range(rows):
        for j in range(cols):
            hist[normalized_magnitude[i, j]] += 1
    target_count = int((1.0 - p) * (rows * cols - hist[0]))
    running_sum = 0
    threshold = 1
    for i in range(1, 256):
        running_sum += hist[i]
        if running_sum >= target_count:
            threshold = i
            break
    return threshold, normalized_magnitude


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
        current_i, current_j = queue.popleft()
        for delta_i in [-1, 0, 1]:
            for delta_j in [-1, 0, 1]:
                if delta_i == 0 and delta_j == 0:
                    continue
                next_i, next_j = current_i + delta_i, current_j + delta_j
                if 0 <= next_i < rows and 0 <= next_j < cols and result[next_i, next_j] == 128:
                    result[next_i, next_j] = 255
                    queue.append((next_i, next_j))
    result[result == 128] = 0
    return result


def canny(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    filtered = gaussian_filter(gray, 5)
    magnitude, direction = sobel(filtered)
    nms = keep_local_max(magnitude, direction)
    threshold, normalized_magnitude = adaptive_binarization(nms, 0.1)
    return hysteresis(normalized_magnitude, threshold, 0.4)
