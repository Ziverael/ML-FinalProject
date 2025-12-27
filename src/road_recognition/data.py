from pathlib import Path
from src.constans import IMG_EXT, IMG_SHAPE, COLOR_INT_RANGE, PROJECT_PATHS
import cv2
import numpy as np
from numpy.typing import NDArray

type ImageMatrix = NDArray[np.uint8]
type ImageNormedMatrix = NDArray[np.float64]

def list_data_dir(subdir: str) -> list[Path]:
    return [p for p in (PROJECT_PATHS.data / subdir).iterdir()]


def load_image_and_label(filename: str) -> tuple[ImageMatrix, ImageMatrix]:
    if not filename.endswith(IMG_EXT):
        msg = f"Ivalid file extension {filename}. Expected {IMG_EXT}."
        raise ValueError(msg)
    img_path = PROJECT_PATHS.data / f"image/image{filename}"
    if not img_path.is_file():
        msg = f"{img_path} does not exist."
        raise ValueError(msg)
    lab_path = PROJECT_PATHS.data / f"label/{filename}"
    if not lab_path.is_file():
        msg = f"{lab_path} does not exist."
        raise ValueError(msg)
    return cv2.imread(img_path), cv2.imread(lab_path)

def normalize_image(img: ImageMatrix) -> ImageNormedMatrix: 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SHAPE)
    return img / COLOR_INT_RANGE

def normalize_label(img: ImageMatrix) -> ImageNormedMatrix: 
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SHAPE)
    return (img > 0).astype(np.float64)


def show_image(img: ImageMatrix) -> None:
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example_file = list_data_dir("label")[0].name
    img, lab = load_image_and_label(example_file)
    show_image(normalize_image(img))
