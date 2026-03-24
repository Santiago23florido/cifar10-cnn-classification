import math
from .model import build_model, build_optimizer, get_initial_weights
from .randomness import set_global_seed
from .runtime import (
    Any,
    BATCH_SIZES,
    BATCH_STUDY_EPOCHS,
    CategoricalCrossentropy,
    K,
    LOSS_REDUCTION,
    OPTIMIZER_BATCH_SIZE,
    OPTIMIZER_CONFIGS,
    OPTIMIZER_STUDY_EPOCHS,
    TRAINING_SEEDS,
    np,
)
from .timing import TimingHistory

def run_training(
    data: dict[str, Any],
    initial_weights,
    optimizer_config: dict[str, Any],
    batch_size: int,
    epochs: int,
    training_seed: int,
) -> dict[str, Any]:
    K.clear_session()
    set_global_seed(training_seed)
    model = build_model()
    model.set_weights(initial_weights)
    model.compile(
        optimizer=build_optimizer(optimizer_config),
        loss=CategoricalCrossentropy(reduction=LOSS_REDUCTION),
        metrics=["accuracy"],
    )

    timing = TimingHistory()
    history = model.fit(
        data["x_train"],
        data["y_train"],
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        shuffle=True,
        validation_data=(data["x_val"], data["y_val"]),
        callbacks=[timing],
    )

    run = {
        "seed": training_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": math.ceil(len(data["x_train"]) / batch_size),
        "optimizer": optimizer_config,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "timing": {
            "step_times": [float(v) for v in timing.batch_times],
            "epoch_times": [float(v) for v in timing.epoch_times],
        },
    }

    del model
    K.clear_session()
    return run
