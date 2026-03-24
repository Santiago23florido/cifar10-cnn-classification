# Section 8 training helper used to build epoch checkpoints for activations.
from keras.callbacks import Callback

from .runtime import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_EPOCHS,
    CategoricalCrossentropy,
    K,
    LOSS_REDUCTION,
    OPTIMIZER_CONFIG,
    REFERENCE_EPOCHS,
    REFERENCE_MODEL_SEED,
    REFERENCE_MODEL_SLUG,
    build_model,
    build_optimizer,
    get_architecture_config,
    get_initial_weights,
)


def checkpoint_path_for_epoch(epoch: int):
    return CHECKPOINT_DIR / f"m3_epoch_{epoch:02d}.weights.h5"


class SelectedEpochCheckpoint(Callback):
    def __init__(self, selected_epochs: list[int]):
        super().__init__()
        self.selected_epochs = set(selected_epochs)

    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        if epoch_number in self.selected_epochs:
            self.model.save_weights(checkpoint_path_for_epoch(epoch_number))


def run_reference_training(data: dict, verbose: bool = True) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = get_architecture_config(REFERENCE_MODEL_SLUG)
    initial_weights = get_initial_weights(config, REFERENCE_MODEL_SEED)

    K.clear_session()
    model = build_model(config)
    model.set_weights(initial_weights)
    model.compile(
        optimizer=build_optimizer(OPTIMIZER_CONFIG),
        loss=CategoricalCrossentropy(reduction=LOSS_REDUCTION),
        metrics=["accuracy"],
    )
    model.save_weights(checkpoint_path_for_epoch(0))

    callbacks = [SelectedEpochCheckpoint(epoch for epoch in CHECKPOINT_EPOCHS if epoch > 0)]
    history = model.fit(
        data["x_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=REFERENCE_EPOCHS,
        verbose=1 if verbose else 0,
        shuffle=True,
        validation_data=(data["x_val"], data["y_val"]),
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    model.save_weights(checkpoint_path_for_epoch(REFERENCE_EPOCHS))
    del model
    K.clear_session()

    return {
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "checkpoint_paths": {str(epoch): str(checkpoint_path_for_epoch(epoch)) for epoch in CHECKPOINT_EPOCHS},
        "final_test_metrics": {"loss": float(test_loss), "accuracy": float(test_accuracy)},
    }


def load_reference_model(epoch: int):
    config = get_architecture_config(REFERENCE_MODEL_SLUG)
    model = build_model(config)
    model.load_weights(checkpoint_path_for_epoch(epoch))
    return model