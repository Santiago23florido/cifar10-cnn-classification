# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

from .markdown_markdown_table import markdown_table

def format_architecture_catalog_markdown(payload: dict) -> str:
    rows = []
    for row in payload["architecture_catalog"]:
        rows.append([row["name"], row["num_conv_layers"], row["layout"], row["params"], row["kernel_regularizer_l2"], row["dropout_before_dense"]])
    return markdown_table(["Modele", "Nb. conv", "Backbone", "Params", "L2", "Dropout before dense"], rows)
