"""
Based on
https://github.com/ArkaJU/U-Net-Satellite?tab=readme-ov-file
"""
import numpy as np
import os
import cv2

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, BatchNormalization
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


def unet(input_shape=(256, 256, 3)):
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

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(1e-4),
        loss=dice_coef_loss,
        metrics=[dice_coef, iou_coef]
    )

    return model