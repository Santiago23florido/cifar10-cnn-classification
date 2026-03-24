# Section 5 formatting helper used by the notebook summaries.
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

def format_trainable_parameter_markdown(summary: dict[str, Any] | None = None) -> str:
    summary = summary or get_trainable_parameter_summary()
    rows = [[row["layer"], row["name"], row["trainable_params"]] for row in summary["layers"]]
    table = markdown_table(["Couche", "Nom", "Parametres trainables"], rows)
    intro = (
        "### VI.2. Distinction entre parametres appris et hyperparametres\n\n"
        f"Le modele contient **{summary['total_trainable']} parametres trainables**. "
        "Ces poids et biais sont ajustes par l'optimisation ; ils ne figurent donc pas dans l'inventaire des hyperparametres. "
        "La Section 5 distingue maintenant les hyperparametres du modele, les conditions experimentales externes et les constantes d'implementation."
    )
    return intro + "\n\n" + table
