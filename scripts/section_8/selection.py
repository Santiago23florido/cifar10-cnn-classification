# Section 8 image-selection helper used by the notebook pipeline.
from .runtime import CLASS_NAMES, PRIMARY_CLASS_NAME, SECONDARY_CLASS_NAME, np


def _select_correct_example(probabilities, y_true_int, target_class_name: str) -> dict:
    class_index = CLASS_NAMES.index(target_class_name)
    y_pred_int = np.argmax(probabilities, axis=1)
    valid_ids = np.where((y_true_int == class_index) & (y_pred_int == class_index))[0]
    if len(valid_ids) == 0:
        raise ValueError(f"No correctly classified sample found for class '{target_class_name}'.")
    class_scores = probabilities[valid_ids, class_index]
    selected_index = int(valid_ids[int(np.argmax(class_scores))])
    return {
        "index": selected_index,
        "true_class_name": target_class_name,
        "predicted_class_name": target_class_name,
        "confidence": float(probabilities[selected_index, class_index]),
    }


def select_representative_images(probabilities, y_true_int) -> dict:
    return {
        "primary": _select_correct_example(probabilities, y_true_int, PRIMARY_CLASS_NAME),
        "secondary": _select_correct_example(probabilities, y_true_int, SECONDARY_CLASS_NAME),
    }
