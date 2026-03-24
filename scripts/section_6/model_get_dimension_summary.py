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
from .model_shape_to_string import shape_to_string

def get_dimension_summary(config: dict[str, Any]) -> list[dict[str, Any]]:
    K.clear_session()
    model = build_model(config)
    rows = [{"layer_name": "input", "layer_type": "Input", "output_shape": shape_to_string(model.input_shape), "params": 0}]
    for layer in model.layers:
        if isinstance(layer, Dropout):
            continue
        rows.append({
            "layer_name": layer.name,
            "layer_type": layer.__class__.__name__,
            "output_shape": shape_to_string(layer.output.shape),
            "params": int(layer.count_params()),
        })
    K.clear_session()
    return rows
