# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

from .markdown__format_comparison_table import _format_comparison_table

def format_confirmation_summary_markdown(payload: dict) -> str:
    body = _format_comparison_table(payload["confirmation"]["summary"])
    selected_slug = payload["confirmation"]["selected_architecture_slug"]
    selected = next(row for row in payload["confirmation"]["summary"] if row["architecture_slug"] == selected_slug)
    return body + "\n\n" + f"Modele selectionne : {selected['architecture_name']}."
