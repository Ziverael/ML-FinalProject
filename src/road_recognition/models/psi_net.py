import tensorflow as tf
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Concatenate, Activation, Multiply, BatchNormalization
)
from keras.models import Model

def conv_block(x, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def attention_gate(skip, gating, filters):
    theta_x = Conv2D(filters, 1, padding="same")(skip)
    phi_g = Conv2D(filters, 1, padding="same")(gating)

    add = Activation("relu")(theta_x + phi_g)
    psi = Conv2D(1, 1, padding="same")(add)
    psi = Activation("sigmoid")(psi)

    return Multiply()([skip, psi])

def psinet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # -------- Main Encoder --------
    e1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(e1)

    e2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(e2)

    e3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(e3)

    # -------- Bottleneck --------
    b = conv_block(p3, 512)

    # -------- Shallow Path --------
    s1 = conv_block(inputs, 32)
    s2 = MaxPooling2D()(s1)
    s2 = conv_block(s2, 64)

    # -------- Decoder --------
    d3 = UpSampling2D()(b)
    a3 = attention_gate(e3, d3, 256)
    d3 = Concatenate()([d3, a3])
    d3 = conv_block(d3, 256)

    d2 = UpSampling2D()(d3)
    a2 = attention_gate(e2, d2, 128)
    d2 = Concatenate()([d2, a2])
    d2 = conv_block(d2, 128)

    d1 = UpSampling2D()(d2)

    # fusion (main + shallow)
    d1 = Concatenate()([d1, e1, s1])
    d1 = conv_block(d1, 64)

    # -------- Output --------
    outputs = Conv2D(1, 1, activation="sigmoid")(d1)

    return Model(inputs, outputs)
