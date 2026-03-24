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

def shape_to_string(shape) -> str:
    dims = [int(dim) for dim in tuple(shape)[1:] if dim is not None]
    if not dims:
        return "-"
    if len(dims) == 1:
        return str(dims[0])
    return "x".join(str(dim) for dim in dims)
