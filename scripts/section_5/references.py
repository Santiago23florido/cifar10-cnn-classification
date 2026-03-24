# Section 5 shared references for notebook-side hyperparameter classification.
NOTEBOOK_REFS = {
    "data_split": "TP3_CNN.ipynb / cell 10",
    "model": "TP3_CNN.ipynb / cell 16",
    "compile": "TP3_CNN.ipynb / cell 20",
    "callbacks": "TP3_CNN.ipynb / cell 25",
    "fit_main": "TP3_CNN.ipynb / cell 26",
    "fit_reload": "TP3_CNN.ipynb / cell 39",
}

SCRIPT_REFS = {
    "data": "scripts/section_4/data.py / load_reduced_cifar10",
    "model": "scripts/section_4/model.py / build_model",
    "optimizer": "scripts/section_4/model.py / build_optimizer ; scripts/section_4/runtime.py / OPTIMIZER_CONFIGS",
    "train": "scripts/section_4/experiments.py / run_training",
    "section4": "scripts/section_4/experiments.py / run_batch_size_study + run_optimizer_study",
    "visualization": "scripts/section_4/visualization",
}

NATURE_ORDER = [
    "structurel impose par la tache",
    "ajustable de conception",
    "entrainement",
    "regularisation",
]
