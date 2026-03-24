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

def make_results_payload(verbose: bool = True) -> dict:
    data = load_reduced_cifar10(SPLIT_SEED)
    batch_runs, batch_summary = run_batch_size_study(data, verbose=verbose)
    optimizer_runs, optimizer_summary = run_optimizer_study(data, verbose=verbose)
    return {
        "meta": {
            "split_seed": SPLIT_SEED,
            "training_seeds": TRAINING_SEEDS,
            "batch_sizes": BATCH_SIZES,
            "batch_study_epochs": BATCH_STUDY_EPOCHS,
            "optimizer_study_epochs": OPTIMIZER_STUDY_EPOCHS,
            "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
            "train_shape": data["train_shape"],
            "val_shape": data["val_shape"],
            "test_shape": data["test_shape"],
            "course_sources": [
                "cours/3_Classification_Apprentissage.pdf",
                "cours/Poly_Chap4_ClassifML.pdf",
            ],
            "optimizer_configs": OPTIMIZER_CONFIGS,
            "loss_reduction": LOSS_REDUCTION,
            "step_time_definition": "mean over the recorded training-batch wall times",
            "epoch_time_definition": "mean over per-epoch wall times including validation",
            "batch_size_comparison_budget": "fixed number of epochs",
            "optimizer_comparison_budget": "fixed number of epochs",
        },
        "batch_size_study": {
            "runs": batch_runs,
            "summary": batch_summary,
        },
        "optimizer_study": {
            "runs": optimizer_runs,
            "summary": optimizer_summary,
        },
    }
