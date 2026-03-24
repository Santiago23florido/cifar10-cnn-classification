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

from .results_get_audit_summary import get_audit_summary
from .results_get_protocol import get_protocol

def build_payload(
    screening_runs: list[dict],
    screening_summary: list[dict],
    shortlisted_refined: list[str],
    confirmation_runs: list[dict],
    confirmation_summary: list[dict],
    selected_slug: str,
    representative_run: dict,
) -> dict:
    selected_config = get_architecture_config(selected_slug)
    selected_summary = get_architecture_summary(selected_config)
    dimension_summary = get_dimension_summary(selected_config)
    return {
        "meta": {"protocol": get_protocol(), "audit": get_audit_summary()},
        "architecture_catalog": [get_architecture_summary(config) for config in ARCHITECTURE_CONFIGS],
        "screening": {"runs": screening_runs, "summary": screening_summary, "shortlisted_refined": shortlisted_refined},
        "confirmation": {"runs": confirmation_runs, "summary": confirmation_summary, "selected_architecture_slug": selected_slug},
        "selected_model": {
            "config": selected_config,
            "summary": selected_summary,
            "dimension_summary": dimension_summary,
            "representative_run": representative_run,
            "hyperparameters": {
                "optimizer": OPTIMIZER_CONFIG,
                "batch_size": BATCH_SIZE,
                "epochs": CONFIRMATION_EPOCHS,
                "kernel_size": list(selected_config["kernel_size"]),
                "padding": selected_config["padding"],
                "pool_size": list(selected_config["pool_size"]),
                "dense_units": selected_config["dense_units"],
                "kernel_regularizer_l2": selected_config["kernel_regularizer_l2"],
                "dropout_before_dense": selected_config["dropout_before_dense"],
                "dropout_after_pool": [stage["dropout_after_pool"] for stage in selected_config["conv_stages"]],
            },
        },
    }
