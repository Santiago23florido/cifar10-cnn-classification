# Section 4 visualization helper used by the notebook and report.
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

def save_batch_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    summary = payload["batch_size_study"]["summary"]
    return {
        "batch_step_time": save_batch_step_time(summary, figure_dir),
        "batch_epoch_time": save_batch_epoch_time(summary, figure_dir),
        "batch_curves": save_batch_curves(summary, figure_dir),
    }
