# Section 4 visualization helper used by the notebook and report.
import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def save_batch_curves(summary: list[dict[str, object]], figure_dir: Path) -> Path:
    epochs = np.arange(1, len(summary[0]["curves"]["loss_mean"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summary)))

    for color, row in zip(colors, summary):
        label = f"B={row['batch_size']}"
        axes[0, 0].plot(epochs, row["curves"]["loss_mean"], color=color, linewidth=2, label=label)
        axes[0, 1].plot(epochs, row["curves"]["val_loss_mean"], color=color, linewidth=2, label=label)
        axes[1, 0].plot(epochs, row["curves"]["accuracy_mean"], color=color, linewidth=2, label=label)
        axes[1, 1].plot(epochs, row["curves"]["val_accuracy_mean"], color=color, linewidth=2, label=label)

    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Validation loss")
    axes[1, 0].set_title("Training accuracy")
    axes[1, 1].set_title("Validation accuracy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, 1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path = figure_dir / "section4_batch_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
