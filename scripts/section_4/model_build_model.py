# Section 4 model helper used by the notebook pipeline.
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

def build_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(
        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(0.00),
        )
    )
    model.add(Dropout(0.0))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.00)))
    model.add(Dropout(0.0))
    model.add(Dense(10, activation="softmax", kernel_regularizer=l2(0.00)))
    model.add(Dropout(0.0))
    return model
