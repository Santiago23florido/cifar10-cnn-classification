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

def stack_metric(runs: list[dict[str, Any]], metric_name: str) -> np.ndarray:
    return np.array([run["history"][metric_name] for run in runs], dtype=np.float64)
