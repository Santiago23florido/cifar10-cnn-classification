from scripts.section_4 import REPO_ROOT

DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "rappport" / "imgs"


def load_image_paths(figure_dir=DEFAULT_FIGURE_DIR) -> dict:
    return {
        "overfitting_example": figure_dir / "section7_overfitting_example.png",
        "regularization_comparison": figure_dir / "section7_regularization_comparison.png",
        "regularization_summary": figure_dir / "section7_regularization_summary.png",
    }
