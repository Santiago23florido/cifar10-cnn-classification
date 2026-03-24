# Section 8 visualization helper used by the notebook and report.
from pathlib import Path

from ..results import load_results


def coerce_payload(results_or_path):
    if isinstance(results_or_path, (str, Path)):
        return load_results(Path(results_or_path))
    return results_or_path