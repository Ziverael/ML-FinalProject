from src.road_recognition.data import (
    list_data_dir,
    load_image_and_label,
    normalize_image,
    normalize_label,
    split_data,
    ImageNormedMatrix,
)
import random
from functools import partialmethod
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.constans import IMG_EXT, BATCH_SIZE
from pydantic import BaseModel, ConfigDict
from numpy.typing import NDArray
from typing import Generator

class SLRawData(BaseModel):
    """Supervised Learning dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: list[str]
    y: list[str]


class Dataset:
    def __init__(self, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_files: list[Path] = list_data_dir("image")
        self.label_files: list[Path] = list_data_dir("label")
        self._check_input_data()

        self.filenames: list[str] = [p.stem + p.suffix for p in self.image_files]
        self.train_data: SLRawData | None = None
        self.val_data: SLRawData | None = None

    
    def _check_input_data(self) -> None:
        diff = set([p.stem for p in self.image_files]).symmetric_difference(
            set(p.stem for p in self.label_files)
        )
        if diff != set():
            msg = f"Missing files {diff}"
            raise ValueError(msg)
            
    
    def split_dataset(self, validation_size: float = 0.2, seed: int | None = None):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.filenames, self.filenames, test_size=validation_size, random_state=seed
        )
        self.train_data = SLRawData(x=train_images, y=train_labels)
        self.val_data = SLRawData(x=val_images, y=val_labels)
    

    def _generate_dataset(self, filenames: list[str]) -> Generator:
        while True: # To avoid StopIteration
            if self.shuffle:
                random.shuffle(filenames)
            for i in range(0, len(filenames), self.batch_size):
                # We assume that names are the same so we take ad hoc the list name
                batch_files = filenames[i:i+self.batch_size]
                extracted = list(map(load_image_and_label, batch_files))
                X_normed, y_normed = transform(extracted)
                yield X_normed, y_normed
    
    def generate_train_dataset(self) -> Generator:
        return self._generate_dataset(self.train_data.x)

    def generate_validation_dataset(self) -> Generator:
        return self._generate_dataset(self.val_data.x)
    
    

def transform(image_label_pairs: list) -> tuple[list[ImageNormedMatrix], list[ImageNormedMatrix]]:
    normed_images = []
    normed_labels = []
    for img, lab in image_label_pairs:
        normed_images.append(normalize_image(img))
        normed_labels.append(normalize_label(lab))
    return normed_images, normed_labels




if __name__ == "__main__":
    import sys

    ds = Dataset()
    ds.split_dataset()
    ds_gen = ds.generate_validation_dataset()
    print(type(ds_gen))
    iter_max = 10
    for i in range(iter_max):
        batch = ds_gen.__next__()
        print(sys.getsizeof(batch))