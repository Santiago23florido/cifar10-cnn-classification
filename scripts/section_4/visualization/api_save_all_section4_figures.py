from ..runtime import Any, Path
from .paths import DEFAULT_FIGURE_DIR
from .payload import coerce_payload
from .plots import (
    save_batch_curves,
    save_batch_epoch_time,
    save_batch_step_time,
    save_optimizer_curves,
    save_optimizer_summary,
)

from .api_save_batch_figures import save_batch_figures
from .api_save_optimizer_figures import save_optimizer_figures

def save_all_section4_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    paths = {}
    paths.update(save_batch_figures(results_or_path, figure_dir))
    paths.update(save_optimizer_figures(results_or_path, figure_dir))
    return paths
