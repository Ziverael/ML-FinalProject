"""
Based on
https://github.com/ArkaJU/U-Net-Satellite?tab=readme-ov-file
"""
import numpy as np
import os
import cv2

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt



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
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (
        K.sum(y_true, [1, 2, 3])
        + K.sum(y_pred, [1, 2, 3])
        - intersection
    )
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)

    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)

    conv2 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)

    conv3 = Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)

    conv4 = Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool4)

    conv5 = Conv2D(
        1024,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5)

    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6])

    conv6 = Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge6)

    conv6 = Conv2D(
        512,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv6)

    up7 = Conv2D(
        256,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7])

    conv7 = Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge7)

    conv7 = Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv7)

    up8 = Conv2D(
        128,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(conv7))

    merge8 = concatenate([conv2, up8])

    conv8 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge8)

    conv8 = Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv8)

    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9])

    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)

    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)

    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)

    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[dice_coef, iou_coef],
    )

    return model
