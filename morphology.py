import numpy as np


def dilate(img, element):
    rows, cols = img.shape
    kh, kw = element.shape
    ph, pw = kh // 2, kw // 2
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(ph, rows - ph):
        for j in range(pw, cols - pw):
            if img[i, j] == 255:
                for u in range(kh):
                    for v in range(kw):
                        if element[u, v] == 1:
                            result[i - ph + u, j - pw + v] = 255
    return result


def erode(img, element):
    rows, cols = img.shape
    kh, kw = element.shape
    ph, pw = kh // 2, kw // 2
    result = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(ph, rows - ph):
        for j in range(pw, cols - pw):
            all_object = True
            for u in range(kh):
                for v in range(kw):
                    if element[u, v] == 1 and img[i - ph + u, j - pw + v] != 255:
                        all_object = False
                        break
                if not all_object:
                    break
            if all_object:
                result[i, j] = 255
    return result


def closing(img, element):
    return erode(dilate(img, element), element)
