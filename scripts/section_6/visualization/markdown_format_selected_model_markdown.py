# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

def format_selected_model_markdown(payload: dict) -> str:
    selected = payload["selected_model"]
    test_payload = selected["representative_run"]["test"]
    per_class = test_payload["per_class_accuracy"]
    best_class_index = int(np.argmax(per_class))
    worst_class_index = int(np.argmin(per_class))
    return (
        "### VII.5. Modele final retenu\n\n"
        f"- Architecture : `{selected['summary']['name']}` ({selected['summary']['title']}).\n"
        f"- Profondeur convolutionnelle : `{selected['summary']['num_conv_layers']}` couches.\n"
        f"- Nombre total de parametres : `{selected['summary']['params']}`.\n"
        f"- Accuracy test du run representatif : `{100.0 * test_payload['accuracy']:.2f}%`.\n"
        f"- Loss test du run representatif : `{test_payload['loss']:.4f}`.\n"
        f"- Meilleure classe : `{test_payload['class_names'][best_class_index]}` ({100.0 * per_class[best_class_index]:.2f}%).\n"
        f"- Classe la plus fragile : `{test_payload['class_names'][worst_class_index]}` ({100.0 * per_class[worst_class_index]:.2f}%)."
    )
