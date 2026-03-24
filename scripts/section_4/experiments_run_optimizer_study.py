# Section 4 experiment helper called from the notebook.
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

from .experiments_aggregate_runs import aggregate_runs
from .experiments_run_training import run_training

def run_optimizer_study(data: dict[str, Any], verbose: bool = True):
    runs = []
    optimizer_labels = [config["name"] for config in OPTIMIZER_CONFIGS]
    for seed in TRAINING_SEEDS:
        initial_weights = get_initial_weights(seed)
        for optimizer_config in OPTIMIZER_CONFIGS:
            if verbose:
                print(
                    "[optimizer-study] "
                    f"seed={seed} optimizer={optimizer_config['name']} epochs={OPTIMIZER_STUDY_EPOCHS}",
                    flush=True,
                )
            run = run_training(
                data=data,
                initial_weights=initial_weights,
                optimizer_config=optimizer_config,
                batch_size=OPTIMIZER_BATCH_SIZE,
                epochs=OPTIMIZER_STUDY_EPOCHS,
                training_seed=seed,
            )
            run["optimizer_name"] = optimizer_config["name"]
            runs.append(run)
    return runs, aggregate_runs(runs, "optimizer_name", optimizer_labels)
