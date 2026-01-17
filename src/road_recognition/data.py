from pathlib import Path
from constans import IMG_EXT, IMG_SHAPE, COLOR_INT_RANGE, PROJECT_PATHS
import cv2
import numpy as np
from typing import Literal, Self
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, model_validator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

type ImageMatrix = NDArray[np.uint8]
type ImageNormedMatrix = NDArray[np.float32]

DataSource = Literal["DeepGlobe", "KaggleSatelite"]

class DataConfig(BaseModel):
    source: DataSource
    label_ext: str
    image_ext: str

    def _check_extension(self, attr: str) -> None:
        if all(not attr.endswith(ext) for ext in IMG_EXT):
            msg = f"Ivalid extension. Expected {IMG_EXT}."
            raise ValueError(msg)

    @model_validator(mode='after')
    def check_label_extension(self) -> Self:
        self._check_extension(self.label_ext)
        return self
    
    @model_validator(mode='after')
    def check_image_extension(self) -> Self:
        self._check_extension(self.image_ext)
        return self
    

CONFIG_DEEP_GLOBE = DataConfig(
    source="DeepGlobe",
    label_ext="png",
    image_ext="jpg"
)

CONFIG_KAGGLE_SATELITE = DataConfig(
    source="KaggleSatelite",
    label_ext="jpg",
    image_ext="jpg"
)

def list_data_dir(subdir: str) -> list[Path]:
    return [p for p in (PROJECT_PATHS.data / subdir).iterdir()]


def load_image_and_label(config: DataConfig, filename: str) -> tuple[ImageMatrix, ImageMatrix]:
    filename, _ext = filename.split(".")
    img_path = PROJECT_PATHS.data / f"{config.source}/image/{filename}.{config.image_ext}"
    if not img_path.is_file():
        msg = f"{img_path} does not exist."
        raise ValueError(msg)
    lab_path = PROJECT_PATHS.data / f"{config.source}/label/{filename}.{config.label_ext}"
    if not lab_path.is_file():
        msg = f"{lab_path} does not exist."
        raise ValueError(msg)
    return load_img(img_path, target_size=IMG_SHAPE, color_mode="rgb"), load_img(lab_path, target_size=IMG_SHAPE, color_mode="grayscale")


def normalize_image(img: ImageMatrix) -> ImageNormedMatrix:
    img_array = img_to_array(img)
    return img_array / COLOR_INT_RANGE


def normalize_label(img: ImageMatrix) -> ImageNormedMatrix:
    return img_to_array(img, dtype=np.bool_)
    # return img[..., np.newaxis]


def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return list(map(lambda x: np.stack(x), (X_train, X_val, y_train, y_val)))


def save_image(img: ImageMatrix | ImageNormedMatrix, dest: Path) -> None:
    plt.imshow(img)
    plt.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    data = CONFIG_DEEP_GLOBE
    example_file = list_data_dir(f"{data.source}/label")[0].name
    img, lab = load_image_and_label(data, example_file)
    save_image(normalize_image(img), PROJECT_PATHS.tmp / "sample.png")

    try:
        invalid_config = DataConfig(source="KaggleSatelite", label_ext="png", image_ext="jpeg")
        msg = "Not captured validation error"
        raise RuntimeError(msg)
    except ValueError:
        print("captured validation error.")
    try:
        invalid_config = DataConfig(source="KaggleSatelite", label_ext="svg", image_ext="jpg")
        msg = "Not captured validation error"
        raise RuntimeError(msg)
    except ValueError:
        print("captured validation error.")