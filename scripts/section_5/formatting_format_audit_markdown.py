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

def format_audit_markdown() -> str:
    lines = ["### VI.1. Diagnostic", ""]
    for item in get_section5_audit():
        lines.append(f"- {item}")
    return "\n".join(lines)
