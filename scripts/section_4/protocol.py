from .runtime import (
    BATCH_SIZES,
    BATCH_STUDY_EPOCHS,
    LOSS_REDUCTION,
    N_TRAINING_SAMPLES,
    N_VALID_SAMPLES,
    OPTIMIZER_BATCH_SIZE,
    OPTIMIZER_CONFIGS,
    OPTIMIZER_STUDY_EPOCHS,
    SPLIT_SEED,
    TRAINING_SEEDS,
)


def get_section4_protocol() -> dict:
    return {
        "batch_sizes": BATCH_SIZES,
        "optimizers": [config["name"] for config in OPTIMIZER_CONFIGS],
        "training_seeds": TRAINING_SEEDS,
        "split_seed": SPLIT_SEED,
        "batch_study_epochs": BATCH_STUDY_EPOCHS,
        "optimizer_study_epochs": OPTIMIZER_STUDY_EPOCHS,
        "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
        "subset_sizes": {
            "train": N_TRAINING_SAMPLES,
            "validation": N_VALID_SAMPLES,
            "test": N_VALID_SAMPLES,
        },
        "optimizer_hyperparameters": {config["name"]: dict(config) for config in OPTIMIZER_CONFIGS},
        "loss_reduction": LOSS_REDUCTION,
        "step_time_definition": "mean over the recorded training-batch wall times",
        "epoch_time_definition": "per-epoch wall time including validation",
        "batch_size_comparison_budget": "fixed number of epochs",
        "optimizer_comparison_budget": "fixed number of epochs",
    }
