from .runtime import (
    ARCHITECTURE_CONFIGS,
    Any,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    K,
    MaxPool2D,
    Sequential,
    l2,
    set_global_seed,
)

from .model_build_model import build_model

def get_initial_weights(config: dict[str, Any], seed: int):
    K.clear_session()
    set_global_seed(seed)
    model = build_model(config)
    weights = model.get_weights()
    del model
    K.clear_session()
    return weights
