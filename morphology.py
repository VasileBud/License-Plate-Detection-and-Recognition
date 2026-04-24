import numpy as np


def dilate(img, element):
    rows, cols = img.shape
    kernel_h, kernel_w = element.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    result = np.zeros_like(img)
    for i in range(pad_h, rows - pad_h):
        for j in range(pad_w, cols - pad_w):
            region = img[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            if np.any(region[element == 1] == 255):
                result[i, j] = 255
    return result


def erode(img, element):
    rows, cols = img.shape
    kernel_h, kernel_w = element.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    result = np.zeros_like(img)
    for i in range(pad_h, rows - pad_h):
        for j in range(pad_w, cols - pad_w):
            region = img[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            if np.all(region[element == 1] == 255):
                result[i, j] = 255
    return result


def closing(img, element):
    return erode(dilate(img, element), element)
