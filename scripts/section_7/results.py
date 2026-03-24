# Section 7 results helper used by the notebook pipeline.
from pathlib import Path

from .example import extract_overfitting_example
from .metrics import aggregate_study_runs
from .runtime import DEFAULT_RESULTS_PATH, OPTIMIZER_CONFIG, RESULTS_DIR, STUDY_EPOCHS, STUDY_SEEDS, json, load_reduced_cifar10
from .study import get_variant_configs, run_regularization_study


def save_results(payload: dict, output_path: Path = DEFAULT_RESULTS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_protocol() -> dict:
    return {
        "dataset": "Reduced CIFAR-10 protocol reused from Sections 4 and 6",
        "subset_sizes": {"train": 5000, "validation": 1000, "test": 1000},
        "preprocessing": "Per-image, per-channel standardization",
        "example_source": "Section 6 screening run of M2 with seed 42",
        "study_seeds": STUDY_SEEDS,
        "study_epochs": STUDY_EPOCHS,
        "batch_size": 32,
        "optimizer": OPTIMIZER_CONFIG,
        "mechanisms_tested": [
            "No explicit regularization",
            "Dropout",
            "Weight decay (L2)",
            "Dropout + Weight decay (L2)",
        ],
        "mechanisms_not_tested": [
            "Data augmentation",
            "Batch normalization",
            "Early stopping as a primary comparison mechanism",
        ],
    }


def get_audit_summary() -> list[str]:
    return [
        "The Section 6 screening run of M2 already exhibits a clear overfitting pattern on the validation loss curve.",
        "The regularized M3 model from Section 6 reduces the train/validation gap, but does not fully eliminate the phenomenon.",
        "Section 7 can therefore remain fully coherent with Section 6 by reusing the M2/M3 family and by comparing dropout and weight decay on the same backbone.",
    ]


def build_payload(example: dict, variants: list[dict], study_runs: list[dict], study_summary: list[dict]) -> dict:
    return {
        "meta": {"protocol": get_protocol(), "audit": get_audit_summary()},
        "overfitting_example": example,
        "regularization_study": {
            "variant_catalog": variants,
            "runs": study_runs,
            "summary": study_summary,
        },
    }


def run_section7_pipeline(output_path: Path = DEFAULT_RESULTS_PATH, force_recompute: bool = False, verbose: bool = True) -> dict:
    if not force_recompute and Path(output_path).exists():
        return load_results(output_path)
    example = extract_overfitting_example()
    variants = get_variant_configs()
    data = load_reduced_cifar10()
    study_runs = run_regularization_study(data, variants, verbose=verbose)
    study_summary = aggregate_study_runs(study_runs, variants)
    payload = build_payload(example=example, variants=variants, study_runs=study_runs, study_summary=study_summary)
    save_results(payload, output_path)
    return payload
