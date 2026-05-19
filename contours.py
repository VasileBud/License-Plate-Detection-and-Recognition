from collections import deque
import numpy as np

DIRS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]


def find_contours(img_bin, min_len=15):
    rows, cols = img_bin.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            padded[i + 1, j + 1] = img_bin[i, j]
    visited = np.zeros((rows + 2, cols + 2), dtype=bool)
    contours = []

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if padded[row, col] == 255 and not visited[row, col] and padded[row, col - 1] == 0:
                contour = trace_border(padded, visited, row, col)
                if len(contour) >= min_len:

                    image_contour = []
                    for cr, cc in contour:
                        image_contour.append((cc - 1, cr - 1))
                    contours.append(image_contour)
    return contours


def trace_border(padded, visited, sr, sc):
    P0 = (sr, sc)
    points = [P0]
    visited[sr, sc] = True
    direction = 7

    search_start = (direction + 6) % 8
    P1 = None
    for step in range(8):
        d = (search_start + step) % 8
        nr = sr + DIRS[d][0]
        nc = sc + DIRS[d][1]
        if padded[nr, nc] == 255:
            P1 = (nr, nc)
            direction = d
            break
    if P1 is None:
        return points
    points.append(P1)
    visited[P1[0], P1[1]] = True

    current = P1
    max_iterations = 2 * padded.shape[0] * padded.shape[1]
    for _ in range(max_iterations):
        if direction % 2 == 0:
            search_start = (direction + 7) % 8
        else:
            search_start = (direction + 6) % 8
        next_point = None
        for step in range(8):
            d = (search_start + step) % 8
            nr = current[0] + DIRS[d][0]
            nc = current[1] + DIRS[d][1]
            if padded[nr, nc] == 255:
                next_point = (nr, nc)
                direction = d
                break
        if next_point is None:
            break

        if next_point == P1 and current == P0:
            break
        points.append(next_point)
        visited[next_point[0], next_point[1]] = True
        current = next_point
    return points
