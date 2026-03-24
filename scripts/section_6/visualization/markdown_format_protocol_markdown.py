from ..runtime import np

from .markdown_markdown_table import markdown_table

def format_protocol_markdown(payload: dict) -> str:
    protocol = payload["meta"]["protocol"]
    rows = [
        ["Split et pretraitement", f"{protocol['subset_sizes']['train']} train / {protocol['subset_sizes']['validation']} val / {protocol['subset_sizes']['test']} test ; {protocol['preprocessing']}"],
        ["Optimiseur fixe", f"Adam, lr={protocol['optimizer']['learning_rate']}, beta_1={protocol['optimizer']['beta_1']}, beta_2={protocol['optimizer']['beta_2']}, epsilon={protocol['optimizer']['epsilon']}"],
        ["Batch size fixe", protocol["batch_size"]],
        ["Screening", f"seed(s) {', '.join(str(seed) for seed in protocol['screening']['seeds'])}, {protocol['screening']['epochs']} epochs"],
        ["Confirmation", f"seed(s) {', '.join(str(seed) for seed in protocol['confirmation']['seeds'])}, {protocol['confirmation']['epochs']} epochs"],
        ["Selection", " > ".join(protocol["selection_rule"])],
    ]
    return markdown_table(["Parametre", "Valeur"], rows)
