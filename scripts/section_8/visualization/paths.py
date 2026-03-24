# Section 8 visualization helper used by the notebook and report.
from pathlib import Path

from scripts.section_4 import REPO_ROOT

DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "rappport" / "imgs"


def load_image_paths(figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    return {
        "first_layer": figure_dir / "section8_first_layer_activation_maps.png",
        "deep_layers": figure_dir / "section8_deep_layer_activation_maps.png",
        "evolution": figure_dir / "section8_activation_evolution.png",
    }