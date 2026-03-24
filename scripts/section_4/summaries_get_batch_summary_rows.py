def get_batch_summary_rows(payload: dict) -> list[dict[str, str]]:
    rows = []
    for row in payload["batch_size_study"]["summary"]:
        rows.append(
            {
                "Batch size": row["batch_size"],
                "Steps/epoch": row["steps_per_epoch"],
                "Mean step time (s)": f"{row['mean_step_time']:.4f}",
                "Std step time (s)": f"{row['std_step_time']:.4f}",
                "Mean epoch time (s)": f"{row['mean_epoch_time']:.3f}",
                "Std epoch time (s)": f"{row['std_epoch_time']:.3f}",
                "Final val. acc. (%)": f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                "Min val. loss": f"{row['mean_min_val_loss']:.4f}",
            }
        )
    return rows
