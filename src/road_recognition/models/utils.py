from keras.models import Model
from keras.optimizers import Adam

from road_recognition.models.dedicated_u_net import unet_modified1
from road_recognition.models.metrics import dice_coef, dice_coef_loss, iou_coef
from road_recognition.models.psi_net import psinet
from road_recognition.models.u_net import unet_base1


MODELS_MAP = {
    "unet_base_1_bc": {
        "model": unet_base1,
        "optimizer": Adam(),
        "loss": "binary_crossentropy",
    },
    "unet_base_1_dice": {
        "model": unet_base1,
        "optimizer": Adam(),
        "loss": dice_coef_loss,
    },
    "unet_modified_1_bc": {
        "model": unet_modified1,
        "optimizer": Adam(),
        "loss": "binary_crossentropy",
    },
    "unet_modified_1_dice": {
        "model": unet_modified1,
        "optimizer": Adam(),
        "loss": dice_coef_loss,
    },
    "psinet_dice": {
        "model": psinet,
        "optimizer": Adam(),
        "loss": dice_coef_loss,
    },
    "psinet_bc": {
        "model": psinet,
        "optimizer": Adam(),
        "loss": "binary_crossentropy",
    },
}


def get_model(name: str) -> Model:
    if (model_config := MODELS_MAP.get(name)) is not None:
        model_init = model_config["model"]
        model = model_init()
        model.compile(
            optimizer=model_config["optimizer"],
            loss=model_config["loss"],
            metrics=["accuracy", iou_coef, dice_coef],
        )
        return model
    msg = f"Invalid model name: {name}"
    raise ValueError(msg)
