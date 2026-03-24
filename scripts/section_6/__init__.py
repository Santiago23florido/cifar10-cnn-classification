from .aggregation import aggregate_runs, compute_confusion_matrices, mean, rank_architectures, std
from .model import (
    build_model,
    get_architecture_config,
    get_architecture_summary,
    get_dimension_summary,
    get_initial_weights,
    shape_to_string,
)
from .results import (
    build_payload,
    get_audit_summary,
    get_protocol,
    load_results,
    run_section6_pipeline,
    save_results,
)
from .runtime import (
    ARCHITECTURE_CONFIGS,
    BATCH_SIZE,
    CLASS_NAMES,
    CONFIRMATION_EPOCHS,
    CONFIRMATION_SEEDS,
    DEFAULT_RESULTS_PATH,
    OPTIMIZER_CONFIG,
    RESULTS_DIR,
    SCREENING_EPOCHS,
    SCREENING_SEEDS,
)
from .training import run_confirmation, run_screening, run_selected_representative, run_training
