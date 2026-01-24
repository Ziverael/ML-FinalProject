import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer
from tensorflow.keras.models import load_model

from constans import BATCH_SIZE, PROJECT_PATHS
from road_recognition.data import DataConfig, DataSource
from road_recognition.dataset import Dataset
from road_recognition.models.metrics import dice_coef, dice_coef_loss, iou_coef


app = typer.Typer(
    name="ModelEvaluation",
    help="Evaluate choosen model on the given dataset.",
    epilog="This is a part of the WUST Machine Learning project 2026.",
)


def get_dataset(source: DataSource) -> DataConfig:
    if source == "DeepGlobe":
        from road_recognition.data import CONFIG_DEEP_GLOBE

        return CONFIG_DEEP_GLOBE
    from road_recognition.data import CONFIG_KAGGLE_SATELITE

    return CONFIG_KAGGLE_SATELITE


@app.command()
def main(
    dataset: DataSource,
    model_path: str,
    random_seed: int | None = None,
):
    ds = Dataset(get_dataset(dataset), batch_size=BATCH_SIZE, size=10_000)
    ds.split_dataset()
    images_to_show = 10
    test_sample, test_labels = ds.get_random_sample(images_to_show)
    model = load_model(
        PROJECT_PATHS.models / model_path,
        custom_objects={
            "iou_coef": iou_coef,
            "dice_coef": dice_coef,
            "dice_coef_loss": dice_coef_loss,
        },
    )

    random.seed(random_seed)

    predictions = model.predict(test_sample)
    predictions = (predictions > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(10, 4, figsize=(10, 3 * 10))
    for i in range(len(test_sample)):
        image = (test_sample[i] * 255).astype(np.uint8)
        mask = predictions[i]

        overlay = image.copy()
        mask = np.repeat(
            mask, 3, axis=2
        )  # matching the size of the channel of the mask and the image to perform an overlay
        inverted_mask = 1 - mask
        yellow_mask = np.array([255, 255, 255]) * mask

        result = image * inverted_mask + yellow_mask
        alpha = 0.2
        predicted_overlay = cv2.addWeighted(
            overlay, alpha, result.astype(overlay.dtype), 1 - alpha, 0
        )

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 2].imshow(yellow_mask)
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(predicted_overlay)
        axes[i, 3].set_title("Predicted Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    fig.savefig(
        PROJECT_PATHS.tmp / "prediction.png", dpi=200, bbox_inches="tight"
    )
