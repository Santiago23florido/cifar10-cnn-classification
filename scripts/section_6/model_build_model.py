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

def build_model(config: dict[str, Any]) -> Sequential:
    model = Sequential(name=config["slug"])
    model.add(Input(shape=(32, 32, 3), name="input"))
    regularizer = l2(config["kernel_regularizer_l2"])
    for stage_index, stage in enumerate(config["conv_stages"], start=1):
        for conv_index in range(1, stage["convs"] + 1):
            model.add(
                Conv2D(
                    filters=stage["filters"],
                    kernel_size=config["kernel_size"],
                    activation="relu",
                    padding=config["padding"],
                    kernel_regularizer=regularizer,
                    name=f"conv_s{stage_index}_{conv_index}",
                )
            )
        if stage["pool"]:
            model.add(MaxPool2D(pool_size=config["pool_size"], name=f"pool_s{stage_index}"))
        if stage["dropout_after_pool"] > 0.0:
            model.add(Dropout(stage["dropout_after_pool"], name=f"drop_s{stage_index}"))

    model.add(Flatten(name="flatten"))
    if config["dropout_before_dense"] > 0.0:
        model.add(Dropout(config["dropout_before_dense"], name="drop_pre_dense"))
    model.add(Dense(config["dense_units"], activation="relu", kernel_regularizer=regularizer, name="dense_hidden"))
    model.add(Dense(config["output_units"], activation="softmax", kernel_regularizer=regularizer, name="dense_output"))
    return model
