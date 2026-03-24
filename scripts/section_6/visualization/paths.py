from ..runtime import REPO_ROOT, Path

DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "rappport" / "imgs"


def load_image_paths(figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    return {
        "final_architecture": figure_dir / "section6_final_architecture.png",
        "final_curves": figure_dir / "section6_final_curves.png",
        "final_confusion_matrices": figure_dir / "section6_final_confusion_matrices.png",
    }
