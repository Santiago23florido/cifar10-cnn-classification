import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def draw_confusion(ax, matrix: np.ndarray, labels: list[str], title: str, value_format: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    threshold = np.max(matrix) / 2 if np.max(matrix) > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = format(matrix[i, j], value_format)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=7,
            )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
