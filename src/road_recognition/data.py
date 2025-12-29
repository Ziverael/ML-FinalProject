from pathlib import Path
from src.constans import IMG_EXT, IMG_SHAPE, COLOR_INT_RANGE, PROJECT_PATHS
import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

type ImageMatrix = NDArray[np.uint8]
type ImageNormedMatrix = NDArray[np.float32]


def list_data_dir(subdir: str) -> list[Path]:
    return [p for p in (PROJECT_PATHS.data / subdir).iterdir()]


def load_image_and_label(filename: str) -> tuple[ImageMatrix, ImageMatrix]:
    if not filename.endswith(IMG_EXT):
        msg = f"Ivalid file extension {filename}. Expected {IMG_EXT}."
        raise ValueError(msg)
    img_path = PROJECT_PATHS.data / f"image/{filename}"
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
    img = img.astype(np.float32)
    return (img / COLOR_INT_RANGE)


def normalize_label(img: ImageMatrix) -> ImageNormedMatrix:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SHAPE)
    img = (img > 0).astype(np.float32)
    return img[..., np.newaxis]


def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return list(map(lambda x: np.stack(x), (X_train, X_val, y_train, y_val)))


def save_image(img: ImageMatrix | ImageNormedMatrix) -> None:
    plt.imshow(img)
    plt.savefig(PROJECT_PATHS.tmp / "sample.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example_file = list_data_dir("label")[0].name
    img, lab = load_image_and_label(example_file)
    save_image(normalize_image(img))
