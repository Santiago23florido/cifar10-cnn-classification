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

def run_batch_size_study(data: dict[str, Any], verbose: bool = True):
    runs = []
    for seed in TRAINING_SEEDS:
        initial_weights = get_initial_weights(seed)
        for batch_size in BATCH_SIZES:
            if verbose:
                print(
                    f"[batch-study] seed={seed} batch_size={batch_size} epochs={BATCH_STUDY_EPOCHS}",
                    flush=True,
                )
            run = run_training(
                data=data,
                initial_weights=initial_weights,
                optimizer_config=OPTIMIZER_CONFIGS[0],
                batch_size=batch_size,
                epochs=BATCH_STUDY_EPOCHS,
                training_seed=seed,
            )
            runs.append(run)
    return runs, aggregate_runs(runs, "batch_size", BATCH_SIZES)
