from .data import load_reduced_cifar10
from .experiments import run_batch_size_study, run_optimizer_study
from .runtime import (
    BATCH_SIZES,
    BATCH_STUDY_EPOCHS,
    DEFAULT_RESULTS_PATH,
    LOSS_REDUCTION,
    OPTIMIZER_BATCH_SIZE,
    OPTIMIZER_CONFIGS,
    OPTIMIZER_STUDY_EPOCHS,
    Path,
    SPLIT_SEED,
    TRAINING_SEEDS,
    json,
)

def load_results(path: Path | str = DEFAULT_RESULTS_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
