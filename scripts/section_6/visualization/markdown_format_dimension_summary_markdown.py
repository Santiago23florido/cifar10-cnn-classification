# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

from .markdown_markdown_table import markdown_table

def format_dimension_summary_markdown(payload: dict) -> str:
    rows = []
    for row in payload["selected_model"]["dimension_summary"]:
        rows.append([row["layer_name"], row["layer_type"], row["output_shape"], row["params"]])
    return markdown_table(["Layer", "Type", "Output shape", "Params"], rows)
