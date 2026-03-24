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

def save_optimizer_figures(results_or_path: dict[str, Any] | Path | str, figure_dir: Path = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    payload = coerce_payload(results_or_path)
    summary = payload["optimizer_study"]["summary"]
    return {
        "optimizer_curves": save_optimizer_curves(summary, figure_dir),
        "optimizer_summary": save_optimizer_summary(summary, figure_dir),
    }
