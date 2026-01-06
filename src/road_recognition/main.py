from src.road_recognition.model import unet
from src.road_recognition.dataset import Dataset
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from src.constans import BATCH_SIZE

if __name__ == "__main__":
    ds = Dataset(batch_size=BATCH_SIZE, size=2000)
    ds.split_dataset()
    train_gen = ds.generate_train_dataset()
    val_gen = ds.generate_validation_dataset()

    model = unet()

    epochs = 20
    steps_per_epoch = len(ds.train_data.x) // BATCH_SIZE
    validation_steps = len(ds.val_data.x) // BATCH_SIZE
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}epochs.h5', monitor='val_loss', save_best_only=True)

    callbacks = [
        ModelCheckpoint(
            "unet_roads.h5",
            monitor="val_dice_coef",
            save_best_only=True,
            mode="max"
        ),
        TensorBoard(log_dir="logs"),
        early_stopping,
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    model.save("my_model.keras")