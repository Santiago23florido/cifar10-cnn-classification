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

from .results_load_results import load_results
from .results_make_results_payload import make_results_payload
from .results_save_results import save_results

def run_section4_pipeline(
    output_path: Path | str = DEFAULT_RESULTS_PATH,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"Loading existing Section 4 results from {output_path}")
        return load_results(output_path)

    payload = make_results_payload(verbose=verbose)
    saved_path = save_results(payload, output_path)
    if verbose:
        print(f"Section 4 results written to {saved_path}")
    return payload
