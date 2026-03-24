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

def format_classification_summary_markdown(items: list[dict[str, str]] | None = None) -> str:
    items = items or get_hyperparameter_inventory()
    grouped = group_inventory_by_nature(items)
    rows = []
    for nature in NATURE_ORDER:
        records = grouped.get(nature, [])
        if not records:
            continue
        rows.append([
            nature,
            len(records),
            ", ".join(record["name"] for record in records[:4]) + ("..." if len(records) > 4 else ""),
        ])
    return "### VI.3. Classification precise\n\n" + markdown_table(["Nature precise", "Nombre", "Exemples"], rows)
