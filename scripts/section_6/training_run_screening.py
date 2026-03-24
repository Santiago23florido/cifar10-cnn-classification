# Section 6 training helper called from the notebook experiments.
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

def run_screening(data: dict, verbose: bool = True) -> tuple[list[dict], list[dict], list[str]]:
    runs = []
    for config in ARCHITECTURE_CONFIGS:
        for seed in SCREENING_SEEDS:
            if verbose:
                print(f"[section6 screening] architecture={config['name']} seed={seed} epochs={SCREENING_EPOCHS}")
            initial_weights = get_initial_weights(config, seed)
            runs.append(
                run_training(
                    data=data,
                    config=config,
                    initial_weights=initial_weights,
                    batch_size=BATCH_SIZE,
                    epochs=SCREENING_EPOCHS,
                    training_seed=seed,
                )
            )
    ordered_slugs = [config["slug"] for config in ARCHITECTURE_CONFIGS]
    summary = aggregate_runs(runs, ordered_slugs)
    refined_ranked = rank_architectures([row for row in summary if row["architecture_slug"] != "m0_baseline"])
    shortlisted = [row["architecture_slug"] for row in refined_ranked[:2]]
    return runs, summary, shortlisted
