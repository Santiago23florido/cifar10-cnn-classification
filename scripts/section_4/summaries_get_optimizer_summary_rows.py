def get_optimizer_summary_rows(payload: dict) -> list[dict[str, str]]:
    rows = []
    for row in payload["optimizer_study"]["summary"]:
        rows.append(
            {
                "Optimizer": row["optimizer_name"],
                "Batch size": payload["meta"]["optimizer_batch_size"],
                "Mean epoch time (s)": f"{row['mean_epoch_time']:.3f}",
                "Std epoch time (s)": f"{row['std_epoch_time']:.3f}",
                "Final val. acc. (%)": f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                "Min val. loss": f"{row['mean_min_val_loss']:.4f}",
            }
        )
    return rows
