# Section 5 inventory helper used by the notebook audit.
from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

def get_trainable_parameter_summary() -> dict[str, Any]:
    model = s4.build_model()
    layer_rows = []
    total_trainable = 0
    for layer in model.layers:
        trainable = sum(int(weight.shape.num_elements()) for weight in layer.trainable_weights)
        total_trainable += trainable
        layer_rows.append(
            {
                "layer": layer.__class__.__name__,
                "name": layer.name,
                "trainable_params": trainable,
            }
        )
    return {
        "total_trainable": total_trainable,
        "layers": layer_rows,
    }
