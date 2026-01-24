import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # If sigmoid not in model, uncomment next line
    # y_pred = tf.sigmoid(y_pred)
    # Compute Dice per sample, then average
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    denominator = K.sum(y_true + y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return K.mean(dice)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        K.sum(y_true, axis=[1, 2, 3])
        + K.sum(y_pred, axis=[1, 2, 3])
        - intersection
    )
    return K.mean((intersection + smooth) / (union + smooth), axis=0)
