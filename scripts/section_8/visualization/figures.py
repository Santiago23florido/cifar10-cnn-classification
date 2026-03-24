# Section 8 visualization helper used by the notebook and report.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import DEFAULT_FIGURE_DIR
from .payload import coerce_payload


def _as_image(array_like):
    array = np.asarray(array_like, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 3:
        return np.clip(array / 255.0, 0.0, 1.0)
    return array


def save_first_layer_figure(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    payload = coerce_payload(results_or_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    panel = payload["first_layer"]["panel"]
    channels = panel["channels"]

    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    axes = axes.flatten()
    axes[0].imshow(_as_image(panel["input_image"]))
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")

    for axis, channel in zip(axes[1:], channels):
        axis.imshow(np.asarray(channel["map"]), cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(f"ch {channel['channel_index']:02d}", fontsize=9)
        axis.axis("off")

    for axis in axes[1 + len(channels) :]:
        axis.axis("off")

    fig.suptitle("First-layer activation maps (conv_s1_1)", fontsize=13)
    plt.tight_layout()
    output_path = figure_dir / "section8_first_layer_activation_maps.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_deep_layer_figure(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    payload = coerce_payload(results_or_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    panel = payload["deep_layers"]
    layer_2 = next(row for row in panel["layers"] if row["layer_name"] == "conv_s2_2")
    layer_3 = next(row for row in panel["layers"] if row["layer_name"] == "conv_s3_2")

    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    axes = axes.flatten()
    axes[0].imshow(_as_image(panel["input_image"]))
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")

    channel_items = [("conv_s2_2", channel) for channel in layer_2["selected_channels"]]
    channel_items += [("conv_s3_2", channel) for channel in layer_3["selected_channels"]]

    for axis, (layer_name, channel) in zip(axes[1:], channel_items):
        axis.imshow(np.asarray(channel["map"]), cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(f"{layer_name} ch {channel['channel_index']:02d}", fontsize=8)
        axis.axis("off")

    for axis in axes[1 + len(channel_items) :]:
        axis.axis("off")

    fig.suptitle("Deeper activation maps in the refined model", fontsize=13)
    plt.tight_layout()
    output_path = figure_dir / "section8_deep_layer_activation_maps.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_evolution_figure(results_or_path, figure_dir=DEFAULT_FIGURE_DIR):
    payload = coerce_payload(results_or_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    panel = payload["evolution"]["panel"]
    rows = panel["tracked_rows"]
    epochs = [row["epoch"] for row in rows[0]["epochs"]]

    fig, axes = plt.subplots(len(rows), len(epochs), figsize=(11, 7.5))
    if len(rows) == 1:
        axes = np.asarray([axes])

    for row_axis, row in zip(axes, rows):
        for axis, epoch_row in zip(row_axis, row["epochs"]):
            axis.imshow(np.asarray(epoch_row["map"]), cmap="magma", vmin=0.0, vmax=1.0)
            axis.set_title(f"epoch {epoch_row['epoch']}", fontsize=9)
            axis.axis("off")
        row_axis[0].set_ylabel(f"{row['layer_name']}\nch {row['channel_index']:02d}", fontsize=9)

    fig.suptitle("Activation map evolution across training epochs", fontsize=13)
    plt.tight_layout()
    output_path = figure_dir / "section8_activation_evolution.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_all_section8_figures(results_or_path, figure_dir=DEFAULT_FIGURE_DIR) -> dict[str, str]:
    return {
        "first_layer": str(save_first_layer_figure(results_or_path, figure_dir)),
        "deep_layers": str(save_deep_layer_figure(results_or_path, figure_dir)),
        "evolution": str(save_evolution_figure(results_or_path, figure_dir)),
    }