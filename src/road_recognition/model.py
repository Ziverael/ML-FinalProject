import numpy as np
import os
import cv2

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, BatchNormalization, Conv2DTranspose, Concatenate
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
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        K.sum(y_true, axis=[1, 2, 3]) +
        K.sum(y_pred, axis=[1, 2, 3]) -
        intersection
    )
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def conv_block(x, filters, dilation=1):
    x = Conv2D(
        filters, 3, padding="same",
        dilation_rate=dilation, kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = Conv2D(
        filters, 3, padding="same",
        dilation_rate=dilation, kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def unet_modified1(input_shape=(256, 256, 3),  output_layer=1):
    """First modification proposition"""

    inputs = Input(input_shape)

    # Encoder (only 3 downsamples)
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D(2)(c3)

    # Bottleneck with dilation (critical for roads)
    b = conv_block(p3, 256, dilation=2)
    b = Dropout(0.4)(b)

    # Decoder
    u3 = UpSampling2D(2)(b)
    u3 = concatenate([u3, c3])
    c4 = conv_block(u3, 128)

    u2 = UpSampling2D(2)(c4)
    u2 = concatenate([u2, c2])
    c5 = conv_block(u2, 64)

    u1 = UpSampling2D(2)(c5)
    u1 = concatenate([u1, c1])
    c6 = conv_block(u1, 32)

    outputs = Conv2D(1, 1, activation="sigmoid")(c6)

    return Model(inputs, outputs)



def unet_base1(input_shape=(256, 256, 3), output_layer=1):
    """Base from: https://github.com/tsdinh442/road-extraction.git"""
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Dropout(0.1)(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Dropout(0.2)(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Dropout(0.2)(conv4)

    # Bottom
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = Dropout(0.3)(conv5)

    # Decoder
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = Concatenate()([conv4, up6])
    conv6 = Dropout(0.2)(merge6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Dropout(0.2)(merge7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Dropout(0.1)(merge8)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Dropout(0.1)(merge9)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)

    # Output
    output = Conv2D(output_layer, (1, 1), activation='sigmoid')(conv9)

    return  Model(inputs=inputs, outputs=output)


MODELS_MAP = {
    "unet_base_1_bc": {"model": unet_base1, "optimizer": Adam(), "loss": "binary_crossentropy"},
    "unet_base_1_dice": {"model": unet_base1, "optimizer": Adam(), "loss": dice_coef_loss},
    "unet_modified_1_bc": {"model": unet_modified1, "optimizer": Adam(), "loss": "binary_crossentropy"},
    "unet_modified_1_dice": {"model": unet_modified1, "optimizer": Adam(), "loss": dice_coef_loss},
}

def get_model(name: str) -> Model:
    if (model_config:= MODELS_MAP.get(name, None)) is not None:
        model_init = model_config["model"]
        model = model_init()
        model.compile(
            optimizer=model_config["optimizer"],
            loss=model_config["loss"],
            metrics=["accuracy", iou_coef, dice_coef]
        )
        return model