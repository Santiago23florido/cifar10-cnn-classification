# Section 6 visualization helper used by the notebook and report.
import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

from .figures_draw_confusion import draw_confusion

def save_final_confusion_matrices(payload: dict, figure_dir: Path) -> Path:
    test_payload = payload["selected_model"]["representative_run"]["test"]
    class_names = test_payload["class_names"]
    confusion = np.array(test_payload["confusion_matrix"], dtype=np.float64)
    confusion_normalized = np.array(test_payload["confusion_matrix_normalized"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    draw_confusion(axes[0], confusion, class_names, "Confusion matrix (counts)", ".0f")
    draw_confusion(axes[1], confusion_normalized, class_names, "Confusion matrix (normalized)", ".2f")
    plt.tight_layout()
    output_path = figure_dir / "section6_final_confusion_matrices.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
