from ..runtime import np

from .markdown_markdown_table import markdown_table

def _format_comparison_table(summary: list[dict]) -> str:
    rows = []
    for row in summary:
        rows.append([row["architecture_name"], row["num_conv_layers"], row["params"], f"{100.0 * row['mean_final_val_accuracy']:.2f}", f"{row['mean_min_val_loss']:.4f}", f"{row['mean_epoch_time']:.3f}"])
    return markdown_table(["Modele", "Nb. conv", "Params", "Val. acc. finale (%)", "Min. val. loss", "Temps/epoch (s)"], rows)
