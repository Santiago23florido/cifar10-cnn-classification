# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

from .markdown_markdown_table import markdown_table

def format_hyperparameter_summary_markdown(payload: dict) -> str:
    selected = payload["selected_model"]
    hyper = selected["hyperparameters"]
    stage_filters = [" -> ".join([str(stage["filters"])] * stage["convs"]) for stage in selected["config"]["conv_stages"]]
    rows = [
        ["Conv stages", " | ".join(stage_filters)],
        ["Kernel size", tuple(hyper["kernel_size"])],
        ["Padding", hyper["padding"]],
        ["Pool size", tuple(hyper["pool_size"])],
        ["Dense units", hyper["dense_units"]],
        ["Batch size", hyper["batch_size"]],
        ["Epochs", hyper["epochs"]],
        ["Optimizer", hyper["optimizer"]["name"]],
        ["Learning rate", hyper["optimizer"]["learning_rate"]],
        ["beta_1", hyper["optimizer"]["beta_1"]],
        ["beta_2", hyper["optimizer"]["beta_2"]],
        ["Adam epsilon", hyper["optimizer"]["epsilon"]],
        ["L2", hyper["kernel_regularizer_l2"]],
        ["Dropout before dense", hyper["dropout_before_dense"]],
        ["Dropout after pools", hyper["dropout_after_pool"]],
    ]
    return markdown_table(["Hyperparametre", "Valeur retenue"], rows)
