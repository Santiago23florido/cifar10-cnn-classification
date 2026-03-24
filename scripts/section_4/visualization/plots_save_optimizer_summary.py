# Section 4 visualization helper used by the notebook and report.
import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def save_optimizer_summary(summary: list[dict[str, object]], figure_dir: Path) -> Path:
    labels = [row["optimizer_name"] for row in summary]
    final_val_acc = [100.0 * row["mean_final_val_accuracy"] for row in summary]
    min_val_loss = [row["mean_min_val_loss"] for row in summary]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    bars1 = ax1.bar(x - width / 2, final_val_acc, width=width, label="Final validation accuracy (%)")
    ax1.set_ylabel("Final validation accuracy (%)")
    ax1.set_xticks(x, labels)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        min_val_loss,
        width=width,
        label="Minimum validation loss",
        color="#dd8452",
    )
    ax2.set_ylabel("Minimum validation loss")

    handles = [bars1, bars2]
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper center")
    plt.title("Optimizer comparison summary")
    plt.tight_layout()
    output_path = figure_dir / "section4_optimizer_summary.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
