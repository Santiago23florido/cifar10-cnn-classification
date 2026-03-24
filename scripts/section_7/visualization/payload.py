from pathlib import Path

from ..results import load_results


def coerce_payload(results_or_path):
    if isinstance(results_or_path, dict):
        return results_or_path
    return load_results(Path(results_or_path))
