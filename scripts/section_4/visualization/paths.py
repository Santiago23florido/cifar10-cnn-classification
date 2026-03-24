from ..runtime import REPO_ROOT, Path

DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "rappport" / "imgs"


def load_image_paths(figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    return {
        "batch_step_time": figure_dir / "section4_batch_step_time.png",
        "batch_epoch_time": figure_dir / "section4_batch_epoch_time.png",
        "batch_curves": figure_dir / "section4_batch_curves.png",
        "optimizer_curves": figure_dir / "section4_optimizer_curves.png",
        "optimizer_summary": figure_dir / "section4_optimizer_summary.png",
    }
