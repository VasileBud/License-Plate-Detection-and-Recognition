import numpy as np


def find_contours(img_bin, min_len: int = 15):
    rows, cols = img_bin.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
    padded[1:rows + 1, 1:cols + 1] = img_bin
    visited = np.zeros_like(padded, dtype=bool)
    dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    contours = []

    def trace(sr, sc):
        start = (sr, sc)
        points = [start]
        visited[sr, sc] = True
        direction_index = 7
        search_start = (direction_index + 6) % 8
        second = None
        for step in range(8):
            current_dir = (search_start + step) % 8
            next_row, next_col = sr + dirs[current_dir][0], sc + dirs[current_dir][1]
            if padded[next_row, next_col] == 255:
                second = (next_row, next_col)
                direction_index = current_dir
                break
        if second is None:
            return points
        points.append(second)
        visited[second[0], second[1]] = True
        previous, current = start, second
        for _ in range(2 * (rows + 2) * (cols + 2)):
            search_start = (direction_index + 7) % 8 if direction_index % 2 == 0 else (direction_index + 6) % 8
            next_point = None
            for step in range(8):
                current_dir = (search_start + step) % 8
                next_row, next_col = current[0] + dirs[current_dir][0], current[1] + dirs[current_dir][1]
                if padded[next_row, next_col] == 255:
                    next_point = (next_row, next_col)
                    direction_index = current_dir
                    break
            if next_point is None:
                break
            if next_point == second and current == start:
                break
            points.append(next_point)
            visited[next_point[0], next_point[1]] = True
            previous, current = current, next_point
        return points

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if padded[row, col] == 255 and not visited[row, col]:
                if padded[row, col - 1] == 0:
                    contour = trace(row, col)
                    if len(contour) >= min_len:
                        contours.append([(current_col - 1, current_row - 1) for current_row, current_col in contour])
    return contours

