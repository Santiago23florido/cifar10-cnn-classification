# Section 5 inventory helper used by the notebook audit.
from __future__ import annotations
from collections import defaultdict
from typing import Any
import scripts.section_4 as s4
from .references import NATURE_ORDER, NOTEBOOK_REFS, SCRIPT_REFS

def group_inventory_by_nature(items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        grouped[item["nature_precise"]].append(item)
    return dict(grouped)
