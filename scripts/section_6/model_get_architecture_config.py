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

def get_architecture_config(slug: str) -> dict[str, Any]:
    for config in ARCHITECTURE_CONFIGS:
        if config["slug"] == slug:
            return config
    raise KeyError(f"Unknown architecture slug: {slug}")
