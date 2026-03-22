from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts import section4_pipeline as s4

NOTEBOOK_REFS = {
    "data_split": "TP3_CNN.ipynb / cell 10",
    "model": "TP3_CNN.ipynb / cell 16",
    "compile": "TP3_CNN.ipynb / cell 20",
    "callbacks": "TP3_CNN.ipynb / cell 25",
    "fit_main": "TP3_CNN.ipynb / cell 26",
    "fit_reload": "TP3_CNN.ipynb / cell 39",
}

SCRIPT_REFS = {
    "data": "scripts/section4_pipeline.py / load_reduced_cifar10",
    "model": "scripts/section4_pipeline.py / build_model",
    "optimizer": "scripts/section4_pipeline.py / OPTIMIZER_CONFIGS + build_optimizer",
    "train": "scripts/section4_pipeline.py / run_training",
    "section4": "scripts/section4_pipeline.py / run_batch_size_study + run_optimizer_study",
}

NATURE_ORDER = [
    "structurel impose par la tache",
    "ajustable de conception",
    "entrainement",
    "regularisation",
]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])



def _inventory_item(
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



def get_section5_audit() -> list[str]:
    return [
        "L'inventaire doit distinguer les hyperparametres un par un, et non seulement par familles.",
        "Les quantites imposees par la tache, les hyperparametres ajustables et les conditions experimentales doivent etre separees plus strictement.",
        "Les hyperparametres d'optimisation deja etudies en Section 4 doivent etre rappeles ici sans dupliquer l'analyse experimentale complete.",
        "Le notebook et le helper peuvent fournir un inventaire analytique code par code, ancre explicitement dans les appels Keras utilises.",
    ]



def get_trainable_parameter_summary() -> dict[str, Any]:
    model = s4.build_model()
    layer_rows = []
    total_trainable = 0
    for layer in model.layers:
        trainable = sum(int(weight.shape.num_elements()) for weight in layer.trainable_weights)
        total_trainable += trainable
        layer_rows.append(
            {
                "layer": layer.__class__.__name__,
                "name": layer.name,
                "trainable_params": trainable,
            }
        )
    return {
        "total_trainable": total_trainable,
        "layers": layer_rows,
    }



def get_hyperparameter_inventory() -> list[dict[str, str]]:
    return [
        _inventory_item(
            name="input_shape",
            code_example="Input(shape=(32,32,3))",
            category="architecture",
            nature_precise="structurel impose par la tache",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="32 x 32 x 3",
            retained_value="32 x 32 x 3",
            influence="fixe la forme des donnees d'entree et contraint la premiere couche du reseau",
        ),
        _inventory_item(
            name="output_units",
            code_example="Dense(10, activation='softmax')",
            category="architecture",
            nature_precise="structurel impose par la tache",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="10",
            retained_value="10",
            influence="impose la dimension de sortie du classifieur et doit coïncider avec le nombre de classes",
        ),
        _inventory_item(
            name="num_conv_layers",
            code_example="model.add(Conv2D(...))",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="1 couche convolutionnelle",
            retained_value="1 couche convolutionnelle",
            influence="controle la profondeur de l'extraction locale et la croissance du champ recepteur effectif",
        ),
        _inventory_item(
            name="filters",
            code_example="Conv2D(filters=8, ...)",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="8",
            retained_value="8",
            influence="modifie la largeur de la representation convolutionnelle et le nombre de poids appris",
        ),
        _inventory_item(
            name="kernel_size",
            code_example="Conv2D(..., kernel_size=(3,3), ...)",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="(3, 3)",
            retained_value="(3, 3)",
            influence="controle le voisinage local explore et participe directement au champ recepteur",
        ),
        _inventory_item(
            name="padding",
            code_example="Conv2D(..., padding='same')",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="same",
            retained_value="same",
            influence="preserve la resolution spatiale de sortie et conditionne la taille des tenseurs transmis aux couches suivantes",
        ),
        _inventory_item(
            name="pool_size",
            code_example="MaxPool2D(pool_size=(2,2))",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="(2, 2)",
            retained_value="(2, 2)",
            influence="reduit la resolution spatiale, diminue le cout calculatoire et agrandit le champ recepteur effectif des couches profondes",
        ),
        _inventory_item(
            name="conv_to_dense_bridge",
            code_example="Flatten()",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="Flatten",
            retained_value="Flatten",
            influence="fixe la transition convolution-dense et donc la taille d'entree de la tete de classification",
        ),
        _inventory_item(
            name="dense_units",
            code_example="Dense(64, activation='relu')",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="64",
            retained_value="64",
            influence="controle la largeur de la tete de classification et la capacite expressive de la partie dense",
        ),
        _inventory_item(
            name="activation_hidden",
            code_example="activation='relu'",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="ReLU",
            retained_value="ReLU",
            influence="introduit la non-linearite et limite en pratique l'attenuation du gradient dans les premieres couches",
        ),
        _inventory_item(
            name="activation_output",
            code_example="activation='softmax'",
            category="architecture",
            nature_precise="ajustable de conception",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="softmax",
            retained_value="softmax",
            influence="rend la sortie interpretable comme distribution de probabilites sur les classes",
        ),
        _inventory_item(
            name="loss",
            code_example="compile(loss='categorical_crossentropy', ...)",
            category="optimisation",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['train']}",
            initial_value="categorical_crossentropy",
            retained_value="categorical_crossentropy",
            influence="definit la quantite dont le gradient pilote l'apprentissage ; ici elle est adaptee a une sortie softmax et a des labels one-hot",
        ),
        _inventory_item(
            name="optimizer",
            code_example="SGD(...), Adam(...)",
            category="optimisation",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['optimizer']}",
            initial_value="SGD",
            explored_values="SGD, SGD+Momentum, Adam",
            retained_value="comparaison explicite en Section 4",
            influence="modifie la dynamique de convergence, le lissage des gradients et la sensibilite au pas d'apprentissage",
        ),
        _inventory_item(
            name="learning_rate",
            code_example="SGD(learning_rate=0.01), Adam(learning_rate=0.001)",
            category="optimisation",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['optimizer']}",
            initial_value="0.01",
            explored_values="0.01 pour SGD et SGD+Momentum ; 0.001 pour Adam",
            retained_value="0.01 (SGD), 0.001 (Adam)",
            influence="multiplie directement le gradient dans les mises a jour ; trop grand il destabilise la descente, trop faible il ralentit fortement la convergence",
        ),
        _inventory_item(
            name="momentum",
            code_example="SGD(..., momentum=0.0) / SGD(..., momentum=0.9)",
            category="optimisation",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['optimizer']}",
            initial_value="0.0",
            explored_values="0.0 et 0.9",
            retained_value="0.9 pour la comparaison SGD+Momentum",
            influence="ajoute une inertie directionnelle qui lisse les fluctuations du gradient entre deux mini-batches",
        ),
        _inventory_item(
            name="beta_1",
            code_example="Adam(..., beta_1=0.9, ...)",
            category="optimisation",
            nature_precise="entrainement",
            location=SCRIPT_REFS['optimizer'],
            explored_values="0.9",
            retained_value="0.9",
            influence="controle le lissage du premier moment dans Adam",
        ),
        _inventory_item(
            name="beta_2",
            code_example="Adam(..., beta_2=0.999, ...)",
            category="optimisation",
            nature_precise="entrainement",
            location=SCRIPT_REFS['optimizer'],
            explored_values="0.999",
            retained_value="0.999",
            influence="controle le lissage du second moment dans Adam",
        ),
        _inventory_item(
            name="adam_eta",
            code_example="Adam(..., epsilon=1e-7)",
            category="optimisation",
            nature_precise="entrainement",
            location=SCRIPT_REFS['optimizer'],
            explored_values="1e-7",
            retained_value="1e-7",
            influence="stabilise numeriquement le denominateur de l'update d'Adam ; elle correspond au terme note eta dans le rapport",
        ),
        _inventory_item(
            name="batch_size",
            code_example="fit(..., batch_size=...)",
            category="protocole",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['fit_main']} ; {NOTEBOOK_REFS['fit_reload']} ; {SCRIPT_REFS['section4']}",
            initial_value="32",
            explored_values="8 ; {8,16,32,64,128}",
            retained_value="32 pour la baseline et l'etude optimiseurs ; grille {8,16,32,64,128} pour l'etude batch-size",
            influence="agit sur le bruit du gradient, le nombre d'updates par epoch et le compromis entre stabilite et cout calculatoire",
        ),
        _inventory_item(
            name="epochs",
            code_example="fit(..., epochs=...)",
            category="protocole",
            nature_precise="entrainement",
            location=f"{NOTEBOOK_REFS['fit_main']} ; {NOTEBOOK_REFS['fit_reload']} ; {SCRIPT_REFS['section4']}",
            initial_value="20",
            explored_values="10 ; 8",
            retained_value="8 pour les comparaisons controlees de la Section 4",
            influence="fixe le budget d'apprentissage en passages sur la base et conditionne le niveau de convergence atteint",
        ),
        _inventory_item(
            name="shuffle",
            code_example="fit(..., shuffle=True)",
            category="protocole",
            nature_precise="entrainement",
            location=SCRIPT_REFS['train'],
            initial_value="defaut Keras sur le notebook",
            explored_values="True dans la pipeline reproductible",
            retained_value="True",
            influence="limite les effets d'ordre entre epochs et stabilise l'estimation du gradient stochastique",
        ),
        _inventory_item(
            name="dropout_rate",
            code_example="Dropout(0.0)",
            category="regularisation",
            nature_precise="regularisation",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="0.0",
            retained_value="0.0",
            influence="si le taux etait strictement positif, il reduirait la co-adaptation des neurones et le risque de sur-apprentissage ; ici son effet empirique est nul",
        ),
        _inventory_item(
            name="l2_lambda",
            code_example="kernel_regularizer=l2(0.00)",
            category="regularisation",
            nature_precise="regularisation",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="0.00",
            retained_value="0.00",
            influence="si le coefficient etait positif, il penaliserait les poids trop grands et lisserait le modele ; ici son effet empirique est nul",
        ),
    ]



def get_experimental_conditions() -> list[dict[str, str]]:
    return [
        _inventory_item(
            name="standardize",
            code_example="standardize(img_data)",
            category="condition experimentale",
            nature_precise="pretraitement externe au modele",
            location=f"{NOTEBOOK_REFS['data_split']} ; {SCRIPT_REFS['data']}",
            initial_value="standardisation image par image et canal par canal",
            retained_value="standardisation image par image et canal par canal",
            influence="homogeneise l'echelle numerique des entrees et stabilise l'optimisation, sans constituer un hyperparametre du modele au sens strict",
            status="condition experimentale",
        ),
        _inventory_item(
            name="subset_sizes",
            code_example="n_training_samples, n_other_samples, n_valid",
            category="condition experimentale",
            nature_precise="protocole externe au modele",
            location=f"{NOTEBOOK_REFS['data_split']} ; {SCRIPT_REFS['data']}",
            initial_value="5000 train / 1000 validation / 1000 test",
            retained_value="5000 train / 1000 validation / 1000 test",
            influence="conditionne la quantite d'information disponible, la variance des mesures et le cout calculatoire, sans etre un hyperparametre sintonisable du modele",
            status="condition experimentale",
        ),
        _inventory_item(
            name="seeds",
            code_example="split_seed, TRAINING_SEEDS",
            category="condition experimentale",
            nature_precise="controle de reproductibilite",
            location=SCRIPT_REFS['section4'],
            initial_value="split seed 42",
            explored_values="training seeds 42 et 314",
            retained_value="42 pour la partition ; 42 et 314 pour les campagnes controlees",
            influence="agit sur la variance experimentale et la comparabilite des campagnes, sans modifier le modele lui-meme",
            status="condition experimentale",
        ),
    ]



def get_non_hyperparameter_constants() -> list[dict[str, str]]:
    return [
        _inventory_item(
            name="poids et biais appris",
            code_example="weights / biases trainables",
            category="parametres appris",
            nature_precise="parametre appris",
            location=f"{NOTEBOOK_REFS['model']} ; {SCRIPT_REFS['model']}",
            initial_value="132010 parametres trainables au total",
            influence="ils encodent la solution apprise a partir des donnees et ne sont donc pas fixes a priori",
            status="parametre appris",
        ),
        _inventory_item(
            name="logging_checkpoint",
            code_example="verbose, filepath, monitor, save_best_only, mode, save_freq",
            category="constante d'implementation",
            nature_precise="journalisation et sauvegarde",
            location=NOTEBOOK_REFS['callbacks'],
            influence="n'influence pas la dynamique d'apprentissage du modele",
            status="constante d'implementation",
        ),
        _inventory_item(
            name="metrics",
            code_example="metrics=['acc'] / ['accuracy']",
            category="constante d'implementation",
            nature_precise="suivi de performance",
            location=f"{NOTEBOOK_REFS['compile']} ; {SCRIPT_REFS['train']}",
            influence="sert a evaluer l'apprentissage, mais ne modifie pas les mises a jour des poids",
            status="constante d'implementation",
        ),
        _inventory_item(
            name="loss_reduction",
            code_example="CategoricalCrossentropy(reduction='sum_over_batch_size')",
            category="constante d'implementation",
            nature_precise="convention de calcul",
            location=SCRIPT_REFS['train'],
            influence="fixe une convention de reduction de la loss dans la pipeline, sans constituer un hyperparametre du modele au sens strict de la Section 5",
            status="constante d'implementation",
        ),
        _inventory_item(
            name="pipeline_visualisation",
            code_example="force_recompute, noms de fichiers, styles de figure",
            category="constante d'implementation",
            nature_precise="pilotage de l'execution",
            location="TP3_CNN.ipynb / section 4 ; scripts/section4_visualization.py",
            influence="sans effet sur le modele ni sur la dynamique d'optimisation",
            status="constante d'implementation",
        ),
    ]



def group_inventory_by_nature(items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        grouped[item["nature_precise"]].append(item)
    return dict(grouped)



def format_audit_markdown() -> str:
    lines = ["### VI.1. Diagnostic", ""]
    for item in get_section5_audit():
        lines.append(f"- {item}")
    return "\n".join(lines)



def format_trainable_parameter_markdown(summary: dict[str, Any] | None = None) -> str:
    summary = summary or get_trainable_parameter_summary()
    rows = [[row["layer"], row["name"], row["trainable_params"]] for row in summary["layers"]]
    table = _markdown_table(["Couche", "Nom", "Parametres trainables"], rows)
    intro = (
        "### VI.2. Distinction entre parametres appris et hyperparametres\n\n"
        f"Le modele contient **{summary['total_trainable']} parametres trainables**. "
        "Ces poids et biais sont ajustes par l'optimisation ; ils ne figurent donc pas dans l'inventaire des hyperparametres. "
        "La Section 5 distingue maintenant les hyperparametres du modele, les conditions experimentales externes et les constantes d'implementation."
    )
    return intro + "\n\n" + table



def format_classification_summary_markdown(items: list[dict[str, str]] | None = None) -> str:
    items = items or get_hyperparameter_inventory()
    grouped = group_inventory_by_nature(items)
    rows = []
    for nature in NATURE_ORDER:
        records = grouped.get(nature, [])
        if not records:
            continue
        rows.append(
            [
                nature,
                len(records),
                ", ".join(record["name"] for record in records[:4]) + ("..." if len(records) > 4 else ""),
            ]
        )
    return "### VI.3. Classification precise\n\n" + _markdown_table(
        ["Nature precise", "Nombre", "Exemples"],
        rows,
    )



def _structure_inventory_rows(items: list[dict[str, str]]) -> list[list[str]]:
    rows = []
    for item in items:
        rows.append(
            [
                item["name"],
                item["code_example"],
                item["nature_precise"],
                item["retained_value"],
                item["influence"],
                item["location"],
            ]
        )
    return rows



def _training_inventory_rows(items: list[dict[str, str]]) -> list[list[str]]:
    rows = []
    for item in items:
        rows.append(
            [
                item["name"],
                item["code_example"],
                item["nature_precise"],
                item["initial_value"],
                item["explored_values"],
                item["retained_value"],
                item["influence"],
                item["location"],
            ]
        )
    return rows



def format_inventory_markdown(items: list[dict[str, str]] | None = None) -> str:
    items = items or get_hyperparameter_inventory()
    structural = [
        item
        for item in items
        if item["nature_precise"] in {"structurel impose par la tache", "ajustable de conception", "regularisation"}
    ]
    training = [item for item in items if item["nature_precise"] == "entrainement"]

    blocks = ["### VI.4. Inventaire analytique des hyperparametres", ""]
    blocks.append("#### Hyperparametres structurels, de conception et de regularisation")
    blocks.append(
        _markdown_table(
            [
                "Nom",
                "Exemple de code",
                "Nature precise",
                "Valeur retenue",
                "Influence attendue",
                "Emplacement",
            ],
            _structure_inventory_rows(structural),
        )
    )
    blocks.append("")
    blocks.append("#### Hyperparametres d'entrainement")
    blocks.append(
        _markdown_table(
            [
                "Nom",
                "Exemple de code",
                "Nature precise",
                "Valeur initiale",
                "Valeurs explorees",
                "Valeur retenue",
                "Influence attendue",
                "Emplacement",
            ],
            _training_inventory_rows(training),
        )
    )
    return "\n".join(blocks)



def format_non_hyperparameter_markdown(
    experimental_conditions: list[dict[str, str]] | None = None,
    excluded: list[dict[str, str]] | None = None,
) -> str:
    if experimental_conditions and excluded is None:
        first_status = experimental_conditions[0].get("status")
        if first_status != "condition experimentale":
            excluded = experimental_conditions
            experimental_conditions = None

    experimental_conditions = experimental_conditions or get_experimental_conditions()
    excluded = excluded or get_non_hyperparameter_constants()

    blocks = ["### VI.5. Conditions externes et quantites ecartees", ""]
    blocks.append("#### Conditions experimentales externes")
    blocks.append(
        _markdown_table(
            ["Nom", "Exemple de code", "Statut", "Valeur", "Influence", "Emplacement"],
            [
                [
                    item["name"],
                    item["code_example"],
                    item["nature_precise"],
                    item["retained_value"],
                    item["influence"],
                    item["location"],
                ]
                for item in experimental_conditions
            ],
        )
    )
    blocks.append("")
    blocks.append("#### Quantites explicitement exclues de l'inventaire")
    blocks.append(
        _markdown_table(
            ["Nom", "Exemple de code", "Statut", "Justification", "Emplacement"],
            [
                [
                    item["name"],
                    item["code_example"],
                    item["status"],
                    item["influence"],
                    item["location"],
                ]
                for item in excluded
            ],
        )
    )
    return "\n".join(blocks)



def format_section5_conclusion_markdown() -> str:
    return "\n".join(
        [
            "### VI.6. Observations principales",
            "",
            "- L'inventaire analytique doit se faire hyperparametre par hyperparametre : code exact, nature precise, valeur initiale, valeurs explorees et influence attendue.",
            "- Les hyperparametres structurels imposes par la tache doivent etre distingues des hyperparametres ajustables de conception et des hyperparametres d'entrainement.",
            "- Les tailles de sous-ensembles, les graines et le standardize influencent l'experience, mais ils sont traites ici comme conditions experimentales externes et non comme hyperparametres du modele.",
        ]
    )



def get_section5_payload() -> dict[str, Any]:
    return {
        "audit": get_section5_audit(),
        "trainable_summary": get_trainable_parameter_summary(),
        "hyperparameters": get_hyperparameter_inventory(),
        "experimental_conditions": get_experimental_conditions(),
        "excluded": get_non_hyperparameter_constants(),
    }


__all__ = [
    "format_audit_markdown",
    "format_classification_summary_markdown",
    "format_inventory_markdown",
    "format_non_hyperparameter_markdown",
    "format_section5_conclusion_markdown",
    "format_trainable_parameter_markdown",
    "get_experimental_conditions",
    "get_hyperparameter_inventory",
    "get_non_hyperparameter_constants",
    "get_section5_audit",
    "get_section5_payload",
    "get_trainable_parameter_summary",
    "group_inventory_by_nature",
]
