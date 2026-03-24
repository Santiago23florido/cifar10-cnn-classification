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

from .training_run_training import run_training

def run_selected_representative(data: dict, selected_slug: str, verbose: bool = True) -> dict:
    config = get_architecture_config(selected_slug)
    seed = CONFIRMATION_SEEDS[0]
    if verbose:
        print(f"[section6 final] architecture={config['name']} seed={seed} epochs={CONFIRMATION_EPOCHS} test_evaluation=True")
    initial_weights = get_initial_weights(config, seed)
    return run_training(
        data=data,
        config=config,
        initial_weights=initial_weights,
        batch_size=BATCH_SIZE,
        epochs=CONFIRMATION_EPOCHS,
        training_seed=seed,
        evaluate_test=True,
    )
