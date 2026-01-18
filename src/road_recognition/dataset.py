from road_recognition.data import (
    list_data_dir,
    load_image_and_label,
    normalize_image,
    normalize_label,
    ImageNormedMatrix,
    DataConfig,
)
import random
from functools import partialmethod, partial
from pathlib import Path
from sklearn.model_selection import train_test_split
from constans import BATCH_SIZE
from pydantic import BaseModel, ConfigDict
from numpy.typing import NDArray
from typing import Generator
import numpy as np

class SLRawData(BaseModel):
    """Supervised Learning dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: list[str]
    y: list[str]


class Dataset:
    def __init__(self, config: DataConfig, batch_size: int = BATCH_SIZE, shuffle: bool = True, size: int | None = None) -> None:
        self._data_config = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_files: list[Path] = list_data_dir(f"{config.source}/image")
        self.label_files: list[Path] = list_data_dir(f"{config.source}/label")
        if size is not None:
            self.image_files = self.image_files[:size]
            self.label_files = self.label_files[:size]
        self._check_input_data()

        self.filenames: list[str] = [p.stem + p.suffix for p in self.image_files]
        self.train_data: SLRawData | None = None
        self.val_data: SLRawData | None = None
        self.test_data: SLRawData | None = None

        self._load_image_and_label = partial(load_image_and_label, config=config)

    
    def _check_input_data(self) -> None:
        diff = set([p.stem for p in self.image_files]).symmetric_difference(
            set(p.stem for p in self.label_files)
        )
        if diff != set():
            msg = f"Missing files {diff}"
            raise ValueError(msg)
            
    
    def split_dataset(self, validation_size: float = 0.2, test_size: float = 0.2, seed: int | None = None):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.filenames, self.filenames, test_size=validation_size + test_size, random_state=seed
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            val_images, val_labels, test_size=(test_size)/(test_size+validation_size), random_state=seed
        )
        self.train_data = SLRawData(x=train_images, y=train_labels)
        self.val_data = SLRawData(x=val_images, y=val_labels)
        self.test_data = SLRawData(x=test_images, y=test_labels)
    

    def _generate_dataset(self, filenames: list[str]) -> Generator:
        while True: # To avoid StopIteration
            if self.shuffle:
                random.shuffle(filenames)
            for i in range(0, len(filenames), self.batch_size):
                # We assume that names are the same so we take ad hoc the list name
                batch_files = filenames[i:i+self.batch_size]
                extracted = list(map(self._load_image_and_label, batch_files))
                X_normed, y_normed = transform(extracted)
                yield X_normed, y_normed
    
    def generate_train_dataset(self) -> Generator:
        return self._generate_dataset(self.train_data.x)

    def generate_validation_dataset(self) -> Generator:
        return self._generate_dataset(self.val_data.x)

    def generate_validation_dataset(self) -> Generator:
        return self._generate_dataset(self.test_data.x)
    
    def get_random_sample(self, size: int | None = None) -> tuple[list[ImageNormedMatrix], list[ImageNormedMatrix]]:
        size: int = len(self.test_data.x) if size is None else size
        images = self.test_data.x.copy()
        random.shuffle(images)
        images = images[:size]
        extracted = list(map(self._load_image_and_label, images))
        X_normed, y_normed = transform(extracted)
        return X_normed, y_normed


    
    

def transform(image_label_pairs: list) -> tuple[list[ImageNormedMatrix], list[ImageNormedMatrix]]:
    n = len(image_label_pairs)
    sample_img = normalize_image(image_label_pairs[0][0])
    sample_lab = normalize_label(image_label_pairs[0][1])
    images = np.empty((n, *sample_img.shape), dtype=sample_img.dtype)
    labels = np.empty((n, *sample_lab.shape), dtype=sample_lab.dtype)
    for i, (img, lab) in enumerate(image_label_pairs):
        images[i] = normalize_image(img)
        labels[i] = normalize_label(lab)
    return images, labels




if __name__ == "__main__":
    import sys
    from road_recognition.data import CONFIG_DEEP_GLOBE
    ds = Dataset(CONFIG_DEEP_GLOBE)
    ds.split_dataset()
    ds_gen = ds.generate_validation_dataset()
    n = len(ds.val_data.x) + len(ds.train_data.x) + len(ds.test_data.x)
    print(f"Train: {len(ds.train_data.x) / n:.2f}; Test: {len(ds.test_data.x) / n:.2f}; Val: {len(ds.val_data.x) / n:.2f}")
    print(type(ds_gen))
    iter_max = 10
    for i in range(iter_max):
        batch = ds_gen.__next__()
        print(type(batch[0]))
        print(batch[0].shape)
        print(sys.getsizeof(batch))