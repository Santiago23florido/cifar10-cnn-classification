import math
from .aggregation import aggregate_runs, compute_confusion_matrices, rank_architectures
from .model import build_model, get_architecture_config, get_initial_weights
from .runtime import (
    ARCHITECTURE_CONFIGS,
    BATCH_SIZE,
    CategoricalCrossentropy,
    CLASS_NAMES,
    CONFIRMATION_EPOCHS,
    CONFIRMATION_SEEDS,
    K,
    LOSS_REDUCTION,
    OPTIMIZER_CONFIG,
    SCREENING_EPOCHS,
    SCREENING_SEEDS,
    TimingHistory,
    build_optimizer,
    np,
    set_global_seed,
)

def run_training(
    data: dict,
    config: dict,
    initial_weights,
    batch_size: int,
    epochs: int,
    training_seed: int,
    evaluate_test: bool = False,
) -> dict:
    K.clear_session()
    set_global_seed(training_seed)
    model = build_model(config)
    model.set_weights(initial_weights)
    model.compile(
        optimizer=build_optimizer(OPTIMIZER_CONFIG),
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
        "architecture_name": config["name"],
        "architecture_slug": config["slug"],
        "seed": training_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": math.ceil(len(data["x_train"]) / batch_size),
        "optimizer": OPTIMIZER_CONFIG,
        "model_params": int(model.count_params()),
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

    run["final_val_accuracy"] = float(run["history"]["val_accuracy"][-1])
    run["min_val_loss"] = float(min(run["history"]["val_loss"]))
    run["best_val_accuracy"] = float(max(run["history"]["val_accuracy"]))
    run["best_val_loss"] = float(min(run["history"]["val_loss"]))

    if evaluate_test:
        test_loss, test_accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
        probabilities = model.predict(data["x_test"], verbose=0)
        y_true = np.argmax(data["y_test"], axis=1)
        y_pred = np.argmax(probabilities, axis=1)
        confusion, confusion_normalized = compute_confusion_matrices(y_true, y_pred)
        per_class_accuracy = [
            float(confusion[index, index] / confusion[index, :].sum()) if confusion[index, :].sum() > 0 else 0.0
            for index in range(len(CLASS_NAMES))
        ]
        run["test"] = {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy),
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_normalized": confusion_normalized.tolist(),
            "per_class_accuracy": per_class_accuracy,
            "class_names": CLASS_NAMES,
        }

    del model
    K.clear_session()
    return run
