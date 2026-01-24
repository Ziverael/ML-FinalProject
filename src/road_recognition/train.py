from typing import TYPE_CHECKING

import tensorflow as tf
import typer
from keras.callbacks import ModelCheckpoint, TensorBoard

from constans import BATCH_SIZE, PROJECT_PATHS
from road_recognition.data import DataConfig, DataSource
from road_recognition.dataset import Dataset
from road_recognition.models.utils import get_model


if TYPE_CHECKING:
    from keras.models import Model


app = typer.Typer(
    name="ModelTraining",
    help="Train choosen model on given dataset.",
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
    model_name: str,
    model_target_dir: str,
    data_size: int = 5_000,
):
    ds = Dataset(get_dataset(dataset), batch_size=BATCH_SIZE, size=data_size)
    ds.split_dataset()
    train_gen = ds.generate_train_dataset()
    val_gen = ds.generate_validation_dataset()
    test_images, test_labels = ds.get_random_sample()
    epochs = 20
    steps_per_epoch = len(ds.train_data.x) // BATCH_SIZE
    validation_steps = len(ds.val_data.x) // BATCH_SIZE
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10
    )

    model_path = PROJECT_PATHS.models / model_target_dir
    model_path.mkdir()
    metrics_directory = PROJECT_PATHS.logs / model_target_dir
    callbacks = [
        ModelCheckpoint(
            model_path / "checkopint.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="max",
        ),
        TensorBoard(log_dir=metrics_directory),
        early_stopping,
    ]

    model: Model = get_model(model_name)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    model.save(model_path / "model.keras")
    results = model.evaluate(test_images, test_labels, return_dict=True)
    writer = tf.summary.create_file_writer(str(metrics_directory / "eval_test"))
    with writer.as_default():
        for name, value in results.items():
            tf.summary.scalar(name, value, step=0)


if __name__ == "__main__":
    app()
