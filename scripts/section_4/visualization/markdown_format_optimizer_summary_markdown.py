from .markdown_markdown_table import markdown_table

def format_optimizer_summary_markdown(payload: dict) -> str:
    rows = []
    for row in payload["optimizer_study"]["summary"]:
        rows.append(
            [
                row["optimizer_name"],
                f"{row['mean_epoch_time']:.3f} +/- {row['std_epoch_time']:.3f}",
                f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                f"{row['mean_min_val_loss']:.4f}",
            ]
        )
    return markdown_table(
        ["Optimiseur", "Temps/epoch (s)", "Accuracy finale val. (%)", "Min. val. loss"],
        rows,
    )
