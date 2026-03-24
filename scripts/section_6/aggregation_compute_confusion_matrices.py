# Section 6 aggregation helper used by notebook comparisons.
from .model import get_architecture_config, get_architecture_summary
from .runtime import CLASS_NAMES, np

def compute_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    normalized = matrix.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized), where=row_sums != 0)
    return matrix, normalized
