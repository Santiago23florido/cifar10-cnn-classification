from .markdown_markdown_table import markdown_table

def format_batch_summary_markdown(payload: dict) -> str:
    rows = []
    for row in payload["batch_size_study"]["summary"]:
        rows.append(
            [
                row["batch_size"],
                row["steps_per_epoch"],
                f"{row['mean_step_time']:.4f} +/- {row['std_step_time']:.4f}",
                f"{row['mean_epoch_time']:.3f} +/- {row['std_epoch_time']:.3f}",
                f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                f"{row['mean_min_val_loss']:.4f}",
            ]
        )
    return markdown_table(
        [
            "Batch size",
            "Steps/epoch",
            "Temps/step (s)",
            "Temps/epoch (s)",
            "Accuracy finale val. (%)",
            "Min. val. loss",
        ],
        rows,
    )
