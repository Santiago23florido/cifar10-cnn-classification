# Section 4 results helper used by the notebook pipeline.
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

def save_results(payload: dict, path: Path | str = DEFAULT_RESULTS_PATH) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
