import sys

import cv2
from tkinter import Tk, filedialog


def select_image() -> str | None:
    """Open a file dialog and return the selected image path, or None."""
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        initialdir="images",
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


def process_image(image: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """Convert to grayscale and detect edges. Returns (gray, edges)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    return gray, edges


def show_results(image: cv2.typing.MatLike, gray: cv2.typing.MatLike, edges: cv2.typing.MatLike) -> None:
    """Display the original, grayscale, and edge-detected images."""
    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    image_path = select_image()
    if image_path is None:
        print("No image selected.")
        sys.exit(0)

    image = load_image(image_path)
    gray, edges = process_image(image)
    show_results(image, gray, edges)


if __name__ == "__main__":
    main()
