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

def run_confirmation(data: dict, shortlisted_refined: list[str], verbose: bool = True) -> tuple[list[dict], list[dict], str]:
    candidate_slugs = ["m0_baseline", *shortlisted_refined]
    runs = []
    for slug in candidate_slugs:
        config = get_architecture_config(slug)
        for seed in CONFIRMATION_SEEDS:
            if verbose:
                print(f"[section6 confirmation] architecture={config['name']} seed={seed} epochs={CONFIRMATION_EPOCHS}")
            initial_weights = get_initial_weights(config, seed)
            runs.append(
                run_training(
                    data=data,
                    config=config,
                    initial_weights=initial_weights,
                    batch_size=BATCH_SIZE,
                    epochs=CONFIRMATION_EPOCHS,
                    training_seed=seed,
                )
            )
    summary = aggregate_runs(runs, candidate_slugs)
    selected_slug = rank_architectures(summary)[0]["architecture_slug"]
    return runs, summary, selected_slug
