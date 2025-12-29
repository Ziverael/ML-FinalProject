from src.road_recognition.model import unet
from src.road_recognition.pipeline import extract, transform
from src.road_recognition.model import unet, dice_coef
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard

if __name__ == "__main__":
    image_label_pairs = extract()
    image_label_normed_splited = transform(image_label_pairs)

    model = unet()

    epochs = 100
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

    X_train, X_val, y_train, y_val = image_label_normed_splited
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=callbacks
    )
    model.save("my_model.keras")