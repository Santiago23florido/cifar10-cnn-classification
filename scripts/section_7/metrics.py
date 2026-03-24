import numpy as np


def compute_overfitting_indicators(run: dict) -> dict:
    history = run["history"]
    val_loss = history["val_loss"]
    val_accuracy = history["val_accuracy"]
    min_val_loss = float(min(val_loss))
    epoch_at_min_val_loss = int(val_loss.index(min_val_loss) + 1)
    final_val_loss = float(val_loss[-1])
    final_train_accuracy = float(history["accuracy"][-1])
    final_val_accuracy = float(val_accuracy[-1])
    best_val_accuracy = float(max(val_accuracy))
    return {
        "min_val_loss": min_val_loss,
        "epoch_at_min_val_loss": epoch_at_min_val_loss,
        "final_val_loss": final_val_loss,
        "overfit_loss_increase": final_val_loss - min_val_loss,
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "final_accuracy_gap": final_train_accuracy - final_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
    }


def aggregate_study_runs(runs: list[dict], variants: list[dict]) -> list[dict]:
    summaries = []
    ordered_slugs = [variant["slug"] for variant in variants]
    variant_by_slug = {variant["slug"]: variant for variant in variants}
    for slug in ordered_slugs:
        slug_runs = [run for run in runs if run["variant_slug"] == slug]
        if not slug_runs:
            continue
        variant = variant_by_slug[slug]
        curve_keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
        curves = {}
        for key in curve_keys:
            stacked = np.array([run["history"][key] for run in slug_runs], dtype=np.float64)
            curves[f"{key}_mean"] = np.mean(stacked, axis=0).tolist()
            curves[f"{key}_std"] = np.std(stacked, axis=0).tolist()
        indicator_rows = [run["indicators"] for run in slug_runs]
        summaries.append(
            {
                "variant_name": variant["name"],
                "variant_slug": variant["slug"],
                "title": variant["title"],
                "mechanism": variant["mechanism"],
                "description": variant["description"],
                "params": variant["params"],
                "layout": variant["layout"],
                "mean_final_val_accuracy": float(np.mean([row["final_val_accuracy"] for row in indicator_rows])),
                "std_final_val_accuracy": float(np.std([row["final_val_accuracy"] for row in indicator_rows])),
                "mean_best_val_accuracy": float(np.mean([row["best_val_accuracy"] for row in indicator_rows])),
                "mean_min_val_loss": float(np.mean([row["min_val_loss"] for row in indicator_rows])),
                "std_min_val_loss": float(np.std([row["min_val_loss"] for row in indicator_rows])),
                "mean_epoch_at_min_val_loss": float(np.mean([row["epoch_at_min_val_loss"] for row in indicator_rows])),
                "mean_final_val_loss": float(np.mean([row["final_val_loss"] for row in indicator_rows])),
                "mean_overfit_loss_increase": float(np.mean([row["overfit_loss_increase"] for row in indicator_rows])),
                "std_overfit_loss_increase": float(np.std([row["overfit_loss_increase"] for row in indicator_rows])),
                "mean_final_train_accuracy": float(np.mean([row["final_train_accuracy"] for row in indicator_rows])),
                "mean_final_accuracy_gap": float(np.mean([row["final_accuracy_gap"] for row in indicator_rows])),
                "std_final_accuracy_gap": float(np.std([row["final_accuracy_gap"] for row in indicator_rows])),
                "mean_epoch_time": float(np.mean([time for run in slug_runs for time in run["timing"]["epoch_times"]])),
                "std_epoch_time": float(np.std([time for run in slug_runs for time in run["timing"]["epoch_times"]])),
                "curves": curves,
            }
        )
    return summaries
