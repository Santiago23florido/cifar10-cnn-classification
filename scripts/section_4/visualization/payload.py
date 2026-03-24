# Section 4 visualization helper used by the notebook and report.
from ..results import load_results
from ..runtime import Any, Path


def coerce_payload(results_or_path: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(results_or_path, dict):
        return results_or_path
    return load_results(results_or_path)
