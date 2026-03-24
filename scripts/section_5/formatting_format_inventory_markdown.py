from typing import Any
from .inventory import (
    get_experimental_conditions,
    get_hyperparameter_inventory,
    get_non_hyperparameter_constants,
    get_section5_audit,
    get_trainable_parameter_summary,
    group_inventory_by_nature,
)
from .references import NATURE_ORDER

from .formatting__structure_inventory_rows import _structure_inventory_rows
from .formatting__training_inventory_rows import _training_inventory_rows
from .formatting_markdown_table import markdown_table

def format_inventory_markdown(items: list[dict[str, str]] | None = None) -> str:
    items = items or get_hyperparameter_inventory()
    structural = [item for item in items if item["nature_precise"] in {"structurel impose par la tache", "ajustable de conception", "regularisation"}]
    training = [item for item in items if item["nature_precise"] == "entrainement"]
    blocks = ["### VI.4. Inventaire analytique des hyperparametres", ""]
    blocks.append("#### Hyperparametres structurels, de conception et de regularisation")
    blocks.append(markdown_table(["Nom", "Exemple de code", "Nature precise", "Valeur retenue", "Influence attendue", "Emplacement"], _structure_inventory_rows(structural)))
    blocks.append("")
    blocks.append("#### Hyperparametres d'entrainement")
    blocks.append(markdown_table(["Nom", "Exemple de code", "Nature precise", "Valeur initiale", "Valeurs explorees", "Valeur retenue", "Influence attendue", "Emplacement"], _training_inventory_rows(training)))
    return "\n".join(blocks)
