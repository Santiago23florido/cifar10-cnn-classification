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

from .formatting_markdown_table import markdown_table

def format_non_hyperparameter_markdown(experimental_conditions: list[dict[str, str]] | None = None, excluded: list[dict[str, str]] | None = None) -> str:
    if experimental_conditions and excluded is None:
        first_status = experimental_conditions[0].get("status")
        if first_status != "condition experimentale":
            excluded = experimental_conditions
            experimental_conditions = None
    experimental_conditions = experimental_conditions or get_experimental_conditions()
    excluded = excluded or get_non_hyperparameter_constants()
    blocks = ["### VI.5. Conditions externes et quantites ecartees", ""]
    blocks.append("#### Conditions experimentales externes")
    blocks.append(markdown_table(["Nom", "Exemple de code", "Statut", "Valeur", "Influence", "Emplacement"], [[item["name"], item["code_example"], item["nature_precise"], item["retained_value"], item["influence"], item["location"]] for item in experimental_conditions]))
    blocks.append("")
    blocks.append("#### Quantites explicitement exclues de l'inventaire")
    blocks.append(markdown_table(["Nom", "Exemple de code", "Statut", "Justification", "Emplacement"], [[item["name"], item["code_example"], item["status"], item["influence"], item["location"]] for item in excluded]))
    return "\n".join(blocks)
