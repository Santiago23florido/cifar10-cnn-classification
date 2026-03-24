from .randomness import set_global_seed
from .runtime import (
    ADAM_EPSILON,
    Adam,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    K,
    MaxPool2D,
    SGD,
    Sequential,
    l2,
)

def build_optimizer(config: dict):
    if config["slug"] == "adam":
        return Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config.get("epsilon", ADAM_EPSILON),
        )
    return SGD(
        learning_rate=config["learning_rate"],
        momentum=config.get("momentum", 0.0),
    )
