# Section 6 visualization helper used by the notebook and report.
import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def save_final_architecture(payload: dict, figure_dir: Path) -> Path:
    rows = payload["selected_model"]["dimension_summary"]
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.axis("off")

    y_positions = np.linspace(0.95, 0.07, len(rows))
    colors = {
        "Input": "#d9edf7",
        "Conv2D": "#fcf3cf",
        "MaxPool2D": "#d5f5e3",
        "Flatten": "#fdebd0",
        "Dense": "#f5eef8",
    }

    for index, (row, y) in enumerate(zip(rows, y_positions)):
        layer_type = row["layer_type"]
        facecolor = colors.get(layer_type, "#ebedef")
        label = f"{row['layer_name']}\n{layer_type}\nOutput: {row['output_shape']}\nParams: {row['params']}"
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=facecolor, edgecolor="#566573"),
        )
        if index < len(rows) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.05),
                xytext=(0.5, y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#566573"),
            )

    ax.set_title("Final multi-layer CNN architecture", fontsize=13)
    plt.tight_layout()
    output_path = figure_dir / "section6_final_architecture.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
