# Section 5 inventory helper used by the notebook audit.
from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

from .inventory_inventory_item import inventory_item

def get_experimental_conditions() -> list[dict[str, str]]:
    return [
        inventory_item(name="standardize", code_example="standardize(img_data)", category="condition experimentale", nature_precise="pretraitement externe au modele", location=f"{NOTEBOOK_REFS['data_split']} ; {SCRIPT_REFS['data']}", initial_value="standardisation image par image et canal par canal", retained_value="standardisation image par image et canal par canal", influence="homogeneise l'echelle numerique des entrees et stabilise l'optimisation, sans constituer un hyperparametre du modele au sens strict", status="condition experimentale"),
        inventory_item(name="subset_sizes", code_example="n_training_samples, n_other_samples, n_valid", category="condition experimentale", nature_precise="protocole externe au modele", location=f"{NOTEBOOK_REFS['data_split']} ; {SCRIPT_REFS['data']}", initial_value="5000 train / 1000 validation / 1000 test", retained_value="5000 train / 1000 validation / 1000 test", influence="conditionne la quantite d'information disponible, la variance des mesures et le cout calculatoire, sans etre un hyperparametre sintonisable du modele", status="condition experimentale"),
        inventory_item(name="seeds", code_example="split_seed, TRAINING_SEEDS", category="condition experimentale", nature_precise="controle de reproductibilite", location=SCRIPT_REFS['section4'], initial_value="split seed 42", explored_values="training seeds 42 et 314", retained_value="42 pour la partition ; 42 et 314 pour les campagnes controlees", influence="agit sur la variance experimentale et la comparabilite des campagnes, sans modifier le modele lui-meme", status="condition experimentale"),
    ]
