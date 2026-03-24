# Section 6 results helper used by the notebook pipeline.
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

from .results_build_payload import build_payload
from .results_load_results import load_results
from .results_save_results import save_results

def run_section6_pipeline(output_path: Path = DEFAULT_RESULTS_PATH, force_recompute: bool = False, verbose: bool = True) -> dict:
    if not force_recompute and Path(output_path).exists():
        return load_results(output_path)
    data = load_reduced_cifar10()
    screening_runs, screening_summary, shortlisted_refined = run_screening(data, verbose=verbose)
    confirmation_runs, confirmation_summary, selected_slug = run_confirmation(data, shortlisted_refined, verbose=verbose)
    representative_run = run_selected_representative(data, selected_slug, verbose=verbose)
    payload = build_payload(
        screening_runs=screening_runs,
        screening_summary=screening_summary,
        shortlisted_refined=shortlisted_refined,
        confirmation_runs=confirmation_runs,
        confirmation_summary=confirmation_summary,
        selected_slug=selected_slug,
        representative_run=representative_run,
    )
    save_results(payload, output_path)
    return payload
