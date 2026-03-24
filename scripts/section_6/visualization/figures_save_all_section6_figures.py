import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

from .figures_save_final_architecture import save_final_architecture
from .figures_save_final_confusion_matrices import save_final_confusion_matrices
from .figures_save_final_curves import save_final_curves

def save_all_section6_figures(results_or_path, figure_dir: Path) -> dict[str, Path]:
    from .payload import coerce_payload

    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    return {
        "final_architecture": save_final_architecture(payload, figure_dir),
        "final_curves": save_final_curves(payload, figure_dir),
        "final_confusion_matrices": save_final_confusion_matrices(payload, figure_dir),
    }
