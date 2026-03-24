# Section 7 visualization helper used by the notebook and report.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import DEFAULT_FIGURE_DIR
from .payload import coerce_payload


def save_overfitting_example(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    example = payload["overfitting_example"]
    history = example["run"]["history"]
    indicators = example["indicators"]
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    axes[0].plot(epochs, history["loss"], label="Training loss", linewidth=2, color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss", linewidth=2, color="#ff7f0e")
    axes[0].axvline(indicators["epoch_at_min_val_loss"], linestyle="--", color="black", linewidth=1.2, label="Best val. loss")
    axes[0].set_title("Overfitting example: M2 loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, history["accuracy"], label="Training accuracy", linewidth=2, color="#2ca02c")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation accuracy", linewidth=2, color="#d62728")
    axes[1].axvline(indicators["epoch_at_min_val_loss"], linestyle="--", color="black", linewidth=1.2, label="Best val. loss")
    axes[1].set_title("Overfitting example: M2 accuracies")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    output_path = figure_dir / "section7_overfitting_example.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_regularization_comparison(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    summary = payload["regularization_study"]["summary"]
    epochs = np.arange(1, len(summary[0]["curves"]["loss_mean"]) + 1)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(summary)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    for color, row in zip(colors, summary):
        label = row["variant_name"]
        axes[0, 0].plot(epochs, row["curves"]["loss_mean"], color=color, linewidth=2, label=label)
        axes[0, 1].plot(epochs, row["curves"]["val_loss_mean"], color=color, linewidth=2, label=label)
        axes[1, 0].plot(epochs, row["curves"]["accuracy_mean"], color=color, linewidth=2, label=label)
        axes[1, 1].plot(epochs, row["curves"]["val_accuracy_mean"], color=color, linewidth=2, label=label)

    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Validation loss")
    axes[1, 0].set_title("Training accuracy")
    axes[1, 1].set_title("Validation accuracy")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, 1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    output_path = figure_dir / "section7_regularization_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_regularization_summary(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    summary = payload["regularization_study"]["summary"]
    labels = [row["variant_name"] for row in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    final_val_acc = [100.0 * row["mean_final_val_accuracy"] for row in summary]
    final_val_acc_std = [100.0 * row["std_final_val_accuracy"] for row in summary]
    axes[0].bar(x, final_val_acc, yerr=final_val_acc_std, capsize=4, color="#4c72b0")
    axes[0].set_title("Final validation accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, labels)
    axes[0].grid(True, axis="y", alpha=0.3)

    min_val_loss = [row["mean_min_val_loss"] for row in summary]
    min_val_loss_std = [row["std_min_val_loss"] for row in summary]
    axes[1].bar(x, min_val_loss, yerr=min_val_loss_std, capsize=4, color="#dd8452")
    axes[1].set_title("Minimum validation loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(x, labels)
    axes[1].grid(True, axis="y", alpha=0.3)

    overfit_increase = [row["mean_overfit_loss_increase"] for row in summary]
    overfit_increase_std = [row["std_overfit_loss_increase"] for row in summary]
    axes[2].bar(x, overfit_increase, yerr=overfit_increase_std, capsize=4, color="#55a868")
    axes[2].set_title("Overfitting increase")
    axes[2].set_ylabel("Final val. loss - min val. loss")
    axes[2].set_xticks(x, labels)
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = figure_dir / "section7_regularization_summary.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_all_section7_figures(results_or_path, figure_dir=DEFAULT_FIGURE_DIR) -> dict:
    return {
        "overfitting_example": save_overfitting_example(results_or_path, figure_dir),
        "regularization_comparison": save_regularization_comparison(results_or_path, figure_dir),
        "regularization_summary": save_regularization_summary(results_or_path, figure_dir),
    }
