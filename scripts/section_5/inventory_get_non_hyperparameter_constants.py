# Section 5 inventory helper used by the notebook audit.
from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

from .inventory_inventory_item import inventory_item

def get_non_hyperparameter_constants() -> list[dict[str, str]]:
    return [
        inventory_item(name="poids et biais appris", code_example="weights / biases trainables", category="parametres appris", nature_precise="parametre appris", location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}", initial_value="132010 parametres trainables au total", influence="ils encodent la solution apprise a partir des donnees et ne sont donc pas fixes a priori", status="parametre appris"),
        inventory_item(name="logging_checkpoint", code_example="verbose, filepath, monitor, save_best_only, mode, save_freq", category="constante d'implementation", nature_precise="journalisation et sauvegarde", location=NOTEBOOK_REFS['callbacks'], influence="n'influence pas la dynamique d'apprentissage du modele", status="constante d'implementation"),
        inventory_item(name="metrics", code_example="metrics=['accuracy']", category="constante d'implementation", nature_precise="suivi de performance", location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['train']}", influence="sert a evaluer l'apprentissage, mais ne modifie pas les mises a jour des poids", status="constante d'implementation"),
        inventory_item(name="loss_reduction", code_example="CategoricalCrossentropy(reduction='sum_over_batch_size')", category="constante d'implementation", nature_precise="convention de calcul", location=SCRIPT_REFS['train'], influence="fixe une convention de reduction de la loss dans la pipeline, sans constituer un hyperparametre du modele au sens strict de la Section 5", status="constante d'implementation"),
        inventory_item(name="pipeline_visualisation", code_example="force_recompute, noms de fichiers, styles de figure", category="constante d'implementation", nature_precise="pilotage de l'execution", location=f"TP3_CNN.ipynb / sections 4 et 6 ; {SCRIPT_REFS['visualization']}", influence="sans effet sur le modele ni sur la dynamique d'optimisation", status="constante d'implementation"),
    ]
