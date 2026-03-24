from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

def inventory_item(
    *,
    name: str,
    code_example: str,
    category: str,
    nature_precise: str,
    location: str,
    influence: str,
    initial_value: str = "-",
    explored_values: str = "-",
    retained_value: str = "-",
    status: str = "hyperparametre",
) -> dict[str, str]:
    return {
        "name": name,
        "code_example": code_example,
        "category": category,
        "nature_precise": nature_precise,
        "location": location,
        "initial_value": initial_value,
        "explored_values": explored_values,
        "retained_value": retained_value,
        "influence": influence,
        "status": status,
    }
