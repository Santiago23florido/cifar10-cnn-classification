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

def get_audit_summary() -> dict:
    baseline = get_architecture_summary(get_architecture_config("m0_baseline"))
    dense_params = (2048 + 1) * 64 + (64 + 1) * 10
    return {
        "baseline_layout": "32x32x3 -> Conv8 -> Pool -> Flatten(2048) -> Dense64 -> Dense10",
        "baseline_params": baseline["params"],
        "baseline_dense_params": dense_params,
        "bottlenecks": [
            "Single convolution layer, hence shallow local feature extraction.",
            "Most trainable parameters are concentrated in the dense head.",
            "No active regularization in the baseline model.",
            "No notebook-centered refinement campaign existed for Section 6.",
        ],
    }
