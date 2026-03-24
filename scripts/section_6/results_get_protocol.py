# Section 6 results helper used by the notebook pipeline.
from .model import get_architecture_config, get_architecture_summary, get_dimension_summary
from .runtime import (
    ARCHITECTURE_CONFIGS,
    BATCH_SIZE,
    CONFIRMATION_EPOCHS,
    CONFIRMATION_SEEDS,
    DEFAULT_RESULTS_PATH,
    OPTIMIZER_CONFIG,
    Path,
    SCREENING_EPOCHS,
    SCREENING_SEEDS,
    json,
    load_reduced_cifar10,
)
from .training import run_confirmation, run_screening, run_selected_representative

def get_protocol() -> dict:
    return {
        "dataset": "Reduced CIFAR-10 protocol reused from Section 4",
        "subset_sizes": {"train": 5000, "validation": 1000, "test": 1000},
        "preprocessing": "Per-image, per-channel standardization",
        "screening": {"seeds": SCREENING_SEEDS, "epochs": SCREENING_EPOCHS},
        "confirmation": {"seeds": CONFIRMATION_SEEDS, "epochs": CONFIRMATION_EPOCHS},
        "batch_size": BATCH_SIZE,
        "optimizer": OPTIMIZER_CONFIG,
        "selection_rule": [
            "Highest mean final validation accuracy",
            "Tie-break: lowest mean minimum validation loss",
            "Second tie-break: lowest mean epoch time",
        ],
    }
