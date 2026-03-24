# Section 6 visualization helper used by the notebook and report.
import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def save_final_curves(payload: dict, figure_dir: Path) -> Path:
    history = payload["selected_model"]["representative_run"]["history"]
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    axes[0, 0].plot(epochs, history["loss"], linewidth=2, color="#1f77b4")
    axes[0, 1].plot(epochs, history["val_loss"], linewidth=2, color="#ff7f0e")
    axes[1, 0].plot(epochs, history["accuracy"], linewidth=2, color="#2ca02c")
    axes[1, 1].plot(epochs, history["val_accuracy"], linewidth=2, color="#d62728")

    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Validation loss")
    axes[1, 0].set_title("Training accuracy")
    axes[1, 1].set_title("Validation accuracy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")

    plt.tight_layout()
    output_path = figure_dir / "section6_final_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
