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

def load_results(path: Path = DEFAULT_RESULTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
