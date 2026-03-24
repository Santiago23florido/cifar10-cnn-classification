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

from .model_build_model import build_model

def get_initial_weights(seed: int):
    K.clear_session()
    set_global_seed(seed)
    model = build_model()
    weights = model.get_weights()
    del model
    K.clear_session()
    return weights
