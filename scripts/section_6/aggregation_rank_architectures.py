# Section 6 aggregation helper used by notebook comparisons.
from .model import get_architecture_config, get_architecture_summary
from .runtime import CLASS_NAMES, np

def rank_architectures(summary: list[dict]) -> list[dict]:
    return sorted(summary, key=lambda row: (-row["mean_final_val_accuracy"], row["mean_min_val_loss"], row["mean_epoch_time"]))
