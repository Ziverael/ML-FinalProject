from keras.models import load_model
from src.road_recognition import data
from src.road_recognition.model import dice_coef, dice_coef_loss, iou_coef
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.constans import PROJECT_PATHS

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

if __name__ == "__main__":
    m = load_model(
    # "models/unet_roads_1.keras",
    "my_model.keras",
    custom_objects={
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "iou_coef": iou_coef,
    },
)
    img = cv2.imread("image.png")
    # img = cv2.imread("boston_45.jpg")
    img = data.normalize_image(img)
    img = np.expand_dims(img, axis=0)
    pred = m.predict(img)
    pred = np.squeeze(pred)
    pred = (pred > 0.5).astype(np.uint8) * 255
    data.save_image(pred, PROJECT_PATHS.tmp / "prediction.png")


