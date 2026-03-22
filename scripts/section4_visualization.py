import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.section4_pipeline import DEFAULT_RESULTS_PATH, load_results


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "rappport" / "imgs"


def _coerce_payload(results_or_path: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(results_or_path, dict):
        return results_or_path
    return load_results(results_or_path)


def save_batch_step_time(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
    batch_sizes = [row["batch_size"] for row in summary]
    means = [row["mean_step_time"] for row in summary]
    stds = [row["std_step_time"] for row in summary]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(batch_sizes, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    plt.xlabel("Batch size")
    plt.ylabel("Mean time per step (s)")
    plt.title("Batch size comparison: step time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = figure_dir / "section4_batch_step_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_batch_epoch_time(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
    batch_sizes = [row["batch_size"] for row in summary]
    means = [row["mean_epoch_time"] for row in summary]
    stds = [row["std_epoch_time"] for row in summary]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(batch_sizes, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    plt.xlabel("Batch size")
    plt.ylabel("Mean time per epoch (s)")
    plt.title("Batch size comparison: epoch time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = figure_dir / "section4_batch_epoch_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_batch_curves(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
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


def save_optimizer_curves(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
    epochs = np.arange(1, len(summary[0]["curves"]["loss_mean"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(summary)))

    for color, row in zip(colors, summary):
        label = row["optimizer_name"]
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
    axes[0, 1].legend(loc="best", fontsize=9)
    plt.tight_layout()
    output_path = figure_dir / "section4_optimizer_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_optimizer_summary(summary: list[dict[str, Any]], figure_dir: Path) -> Path:
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


def save_batch_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = _coerce_payload(results_or_path)
    summary = payload["batch_size_study"]["summary"]
    return {
        "batch_step_time": save_batch_step_time(summary, figure_dir),
        "batch_epoch_time": save_batch_epoch_time(summary, figure_dir),
        "batch_curves": save_batch_curves(summary, figure_dir),
    }


def save_optimizer_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = _coerce_payload(results_or_path)
    summary = payload["optimizer_study"]["summary"]
    return {
        "optimizer_curves": save_optimizer_curves(summary, figure_dir),
        "optimizer_summary": save_optimizer_summary(summary, figure_dir),
    }


def save_all_section4_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    paths = {}
    paths.update(save_batch_figures(results_or_path, figure_dir))
    paths.update(save_optimizer_figures(results_or_path, figure_dir))
    return paths


def load_image_paths(figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    return {
        "batch_step_time": figure_dir / "section4_batch_step_time.png",
        "batch_epoch_time": figure_dir / "section4_batch_epoch_time.png",
        "batch_curves": figure_dir / "section4_batch_curves.png",
        "optimizer_curves": figure_dir / "section4_optimizer_curves.png",
        "optimizer_summary": figure_dir / "section4_optimizer_summary.png",
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def format_protocol_markdown(protocol: dict[str, Any]) -> str:
    optimizer_params = protocol["optimizer_hyperparameters"]
    rows = [
        ["Sections de reference", "Section II pour les donn?es et la standardisation, Section III pour l'architecture"],
        ["Sous-ensembles", f"{protocol['subset_sizes']['train']} train / {protocol['subset_sizes']['validation']} validation / {protocol['subset_sizes']['test']} test"],
        ["Graine de partition", protocol["split_seed"]],
        ["Graines d'entrainement", ", ".join(str(seed) for seed in protocol["training_seeds"])],
        ["Comparaison batch size", f"{protocol['batch_study_epochs']} epochs fixes, batch sizes {', '.join(str(value) for value in protocol['batch_sizes'])}"],
        ["Comparaison optimiseurs", f"{protocol['optimizer_study_epochs']} epochs fixes, batch size {protocol['optimizer_batch_size']}"],
        ["SGD", f"lr={optimizer_params['SGD']['learning_rate']}, momentum={optimizer_params['SGD']['momentum']}"],
        ["SGD+Momentum", f"lr={optimizer_params['SGD+Momentum']['learning_rate']}, momentum={optimizer_params['SGD+Momentum']['momentum']}"],
        ["Adam", f"lr={optimizer_params['Adam']['learning_rate']}, beta_1={optimizer_params['Adam']['beta_1']}, beta_2={optimizer_params['Adam']['beta_2']}, epsilon={optimizer_params['Adam']['epsilon']}"],
        ["Convention de loss", protocol['loss_reduction']],
        ["Temps/step", protocol['step_time_definition']],
        ["Temps/epoch", protocol['epoch_time_definition']],
    ]
    return _markdown_table(["Parametre", "Valeur"], rows)


def format_batch_summary_markdown(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["batch_size_study"]["summary"]:
        rows.append(
            [
                row["batch_size"],
                row["steps_per_epoch"],
                f"{row['mean_step_time']:.4f} ± {row['std_step_time']:.4f}",
                f"{row['mean_epoch_time']:.3f} ± {row['std_epoch_time']:.3f}",
                f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                f"{row['mean_min_val_loss']:.4f}",
            ]
        )
    return _markdown_table(
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


def format_optimizer_summary_markdown(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["optimizer_study"]["summary"]:
        rows.append(
            [
                row["optimizer_name"],
                f"{row['mean_epoch_time']:.3f} ± {row['std_epoch_time']:.3f}",
                f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                f"{row['mean_min_val_loss']:.4f}",
            ]
        )
    return _markdown_table(
        ["Optimiseur", "Temps/epoch (s)", "Accuracy finale val. (%)", "Min. val. loss"],
        rows,
    )


__all__ = [
    "DEFAULT_FIGURE_DIR",
    "DEFAULT_RESULTS_PATH",
    "format_batch_summary_markdown",
    "format_optimizer_summary_markdown",
    "format_protocol_markdown",
    "load_image_paths",
    "save_all_section4_figures",
    "save_batch_figures",
    "save_optimizer_figures",
]
