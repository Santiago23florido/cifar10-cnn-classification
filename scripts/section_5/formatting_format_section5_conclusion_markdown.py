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

def format_section5_conclusion_markdown() -> str:
    return "\n".join([
        "### VI.6. Observations principales",
        "",
        "- L'inventaire analytique doit se faire hyperparametre par hyperparametre : code exact, nature precise, valeur initiale, valeurs explorees et influence attendue.",
        "- Les hyperparametres structurels imposes par la tache doivent etre distingues des hyperparametres ajustables de conception et des hyperparametres d'entrainement.",
        "- Les tailles de sous-ensembles, les graines et le standardize influencent l'experience, mais ils sont traites ici comme conditions experimentales externes et non comme hyperparametres du modele.",
    ])
