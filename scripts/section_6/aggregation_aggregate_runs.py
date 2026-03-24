# Section 6 aggregation helper used by notebook comparisons.
from .model import get_architecture_config, get_architecture_summary
from .runtime import CLASS_NAMES, np

from .aggregation_mean import mean
from .aggregation_std import std

def aggregate_runs(runs: list[dict], ordered_slugs: list[str]) -> list[dict]:
    summaries = []
    for slug in ordered_slugs:
        slug_runs = [run for run in runs if run["architecture_slug"] == slug]
        if not slug_runs:
            continue
        config = get_architecture_config(slug)
        architecture_summary = get_architecture_summary(config)
        curve_keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
        curves = {}
        for key in curve_keys:
            stacked = np.array([run["history"][key] for run in slug_runs], dtype=np.float64)
            curves[f"{key}_mean"] = np.mean(stacked, axis=0).tolist()
            curves[f"{key}_std"] = np.std(stacked, axis=0).tolist()
        summaries.append(
            {
                "architecture_name": config["name"],
                "architecture_slug": slug,
                "title": architecture_summary["title"],
                "description": architecture_summary["description"],
                "num_conv_layers": architecture_summary["num_conv_layers"],
                "params": architecture_summary["params"],
                "layout": architecture_summary["layout"],
                "mean_step_time": mean([time for run in slug_runs for time in run["timing"]["step_times"]]),
                "std_step_time": std([time for run in slug_runs for time in run["timing"]["step_times"]]),
                "mean_epoch_time": mean([time for run in slug_runs for time in run["timing"]["epoch_times"]]),
                "std_epoch_time": std([time for run in slug_runs for time in run["timing"]["epoch_times"]]),
                "mean_final_val_accuracy": mean([run["final_val_accuracy"] for run in slug_runs]),
                "std_final_val_accuracy": std([run["final_val_accuracy"] for run in slug_runs]),
                "mean_min_val_loss": mean([run["min_val_loss"] for run in slug_runs]),
                "std_min_val_loss": std([run["min_val_loss"] for run in slug_runs]),
                "curves": curves,
            }
        )
    return summaries
