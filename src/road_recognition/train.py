from road_recognition.dataset import Dataset
import tensorflow as tf
from road_recognition.data import DataSource, DataConfig
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from constans import BATCH_SIZE, PROJECT_PATHS
import typer
from typing import Literal


app = typer.Typer(
        name="ModelTraining",
        help="Train choosen model on given dataset.",
        epilog="This is a part of the WUST Machine Learning project 2026.",
    )

ModelType = Literal["Unet-v1", "Unet-v2"] 

def get_model(model: ModelType) -> Model:
    if model == "Unet-v1":
        from road_recognition.model import unet
        return unet()
    else:
        from road_recognition.model import unet2
        return unet2()

def get_dataset(source: DataSource) -> DataConfig:
    if source == "DeepGlobe":
        from road_recognition.data import CONFIG_DEEP_GLOBE
        return CONFIG_DEEP_GLOBE
    else:
        from road_recognition.data import CONFIG_KAGGLE_SATELITE
        return CONFIG_KAGGLE_SATELITE

@app.command()
def main(
    dataset: DataSource,
    model_type: ModelType,
    model_target_dir: str,
):
    ds = Dataset(get_dataset(dataset), batch_size=BATCH_SIZE, size=2000)
    ds.split_dataset()
    train_gen = ds.generate_train_dataset()
    val_gen = ds.generate_validation_dataset()
    test_images, test_labels = ds.get_random_sample()

    model_path = PROJECT_PATHS.models / model_target_dir
    model_path.mkdir()

    model: Model = get_model(model_type)

    epochs = 20
    steps_per_epoch = len(ds.train_data.x) // BATCH_SIZE
    validation_steps = len(ds.val_data.x) // BATCH_SIZE
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}epochs.h5', monitor='val_loss', save_best_only=True)

    callbacks = [
        ModelCheckpoint(
            model_path / "unet_checkopint.h5",
            monitor="val_dice_coef",
            save_best_only=True,
            mode="max"
        ),
        TensorBoard(log_dir=PROJECT_PATHS.logs),
        early_stopping,
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    model.save(model_path / "unet_final.keras")
    results = model.evaluate(test_images, test_labels, return_dict=True)
    writer = tf.summary.create_file_writer(PROJECT_PATHS.logs / "test")
    with writer.as_default():
        for name, value in results.items():
            tf.summary.scalar(name, value, step=0)

if __name__ == "__main__":
    app()
