from scripts.section_6 import load_results as load_section6_results

from .metrics import compute_overfitting_indicators
from .runtime import OVERFITTING_SOURCE, SECTION6_RESULTS_PATH


def extract_overfitting_example(section6_payload: dict | None = None) -> dict:
    payload = section6_payload or load_section6_results(SECTION6_RESULTS_PATH)
    target_slug = OVERFITTING_SOURCE["architecture_slug"]
    target_seed = OVERFITTING_SOURCE["seed"]
    for run in payload["screening"]["runs"]:
        if run["architecture_slug"] == target_slug and run["seed"] == target_seed:
            return {
                "source": OVERFITTING_SOURCE,
                "run": run,
                "indicators": compute_overfitting_indicators(run),
            }
    raise KeyError("Unable to locate the Section 6 overfitting example in section6_results.json")
