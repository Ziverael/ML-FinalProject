from src.road_recognition.data import (
    list_data_dir,
    load_image_and_label,
    normalize_image,
    normalize_label,
    split_data,
)
from pathlib import Path


def _check_input_data(image_files: list[Path], label_files: list[Path]):
    diff = set([p.stem for p in image_files]).symmetric_difference(
        set(f"image{p.stem}" for p in label_files)
    )
    if diff != set():
        msg = f"Missing files {diff}"
        raise ValueError(msg)


def extract():
    image_files = list_data_dir("image")
    label_files = list_data_dir("label")
    _check_input_data(image_files, label_files)
    return [load_image_and_label(f"{p.stem}.bmp") for p in label_files]


def transform(image_label_pairs: list):
    normed_images = []
    normed_labels = []
    for img, lab in image_label_pairs:
        normed_images.append(normalize_image(img))
        normed_labels.append(normalize_label(lab))
    return split_data(normed_images, normed_labels)


if __name__ == "__main__":
    image_label_pairs = extract()
    image_label_normed_splited = transform(image_label_pairs)