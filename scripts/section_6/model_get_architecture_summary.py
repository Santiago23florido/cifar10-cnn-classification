# Section 6 architecture helper used by the notebook pipeline.
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

def get_architecture_summary(config: dict[str, Any]) -> dict[str, Any]:
    K.clear_session()
    model = build_model(config)
    params = int(model.count_params())
    K.clear_session()
    stage_tokens = []
    for stage in config["conv_stages"]:
        conv_token = "-".join([str(stage["filters"])] * stage["convs"])
        stage_tokens.append(f"{conv_token}/pool")
    return {
        "name": config["name"],
        "slug": config["slug"],
        "title": config["title"],
        "description": config["description"],
        "num_conv_layers": sum(stage["convs"] for stage in config["conv_stages"]),
        "num_stages": len(config["conv_stages"]),
        "params": params,
        "layout": " | ".join(stage_tokens),
        "dense_units": config["dense_units"],
        "kernel_regularizer_l2": config["kernel_regularizer_l2"],
        "dropout_before_dense": config["dropout_before_dense"],
        "dropout_after_pool": [stage["dropout_after_pool"] for stage in config["conv_stages"]],
    }
