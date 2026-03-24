# Section 6 aggregation helper used by notebook comparisons.
from .model import get_architecture_config, get_architecture_summary
from .runtime import CLASS_NAMES, np

def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
