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

from .experiments_stack_metric import stack_metric

def aggregate_runs(runs: list[dict[str, Any]], group_key: str, labels: list[Any]) -> list[dict[str, Any]]:
    aggregated = []
    for label in labels:
        matching = [run for run in runs if run[group_key] == label]
        step_times = np.concatenate(
            [np.array(run["timing"]["step_times"], dtype=np.float64) for run in matching]
        )
        epoch_times = np.concatenate(
            [np.array(run["timing"]["epoch_times"], dtype=np.float64) for run in matching]
        )
        aggregated.append(
            {
                group_key: label,
                "steps_per_epoch": int(matching[0]["steps_per_epoch"]),
                "mean_step_time": float(step_times.mean()),
                "std_step_time": float(step_times.std(ddof=0)),
                "mean_epoch_time": float(epoch_times.mean()),
                "std_epoch_time": float(epoch_times.std(ddof=0)),
                "mean_final_val_accuracy": float(
                    np.mean([run["history"]["val_accuracy"][-1] for run in matching])
                ),
                "mean_min_val_loss": float(
                    np.mean([min(run["history"]["val_loss"]) for run in matching])
                ),
                "curves": {
                    "loss_mean": stack_metric(matching, "loss").mean(axis=0).tolist(),
                    "loss_std": stack_metric(matching, "loss").std(axis=0, ddof=0).tolist(),
                    "val_loss_mean": stack_metric(matching, "val_loss").mean(axis=0).tolist(),
                    "val_loss_std": stack_metric(matching, "val_loss").std(axis=0, ddof=0).tolist(),
                    "accuracy_mean": stack_metric(matching, "accuracy").mean(axis=0).tolist(),
                    "accuracy_std": stack_metric(matching, "accuracy").std(axis=0, ddof=0).tolist(),
                    "val_accuracy_mean": stack_metric(matching, "val_accuracy").mean(axis=0).tolist(),
                    "val_accuracy_std": stack_metric(matching, "val_accuracy").std(axis=0, ddof=0).tolist(),
                },
            }
        )
    return aggregated
