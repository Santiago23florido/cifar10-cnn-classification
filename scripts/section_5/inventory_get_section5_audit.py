# Section 5 inventory helper used by the notebook audit.
from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

def get_section5_audit() -> list[str]:
    return [
        "L'inventaire doit distinguer les hyperparametres un par un, et non seulement par familles.",
        "Les quantites imposees par la tache, les hyperparametres ajustables et les conditions experimentales doivent etre separees plus strictement.",
        "Les hyperparametres d'optimisation deja etudies en Section 4 doivent etre rappeles ici sans dupliquer l'analyse experimentale complete.",
        "Le notebook et le helper peuvent fournir un inventaire analytique code par code, ancre explicitement dans les appels Keras utilises.",
    ]
