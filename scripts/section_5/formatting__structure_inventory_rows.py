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

def _structure_inventory_rows(items: list[dict[str, str]]) -> list[list[str]]:
    return [[item["name"], item["code_example"], item["nature_precise"], item["retained_value"], item["influence"], item["location"]] for item in items]
