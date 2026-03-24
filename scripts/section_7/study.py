# Section 7 regularization study helper called from the notebook.
from copy import deepcopy

from scripts.section_6 import build_model, get_architecture_config, get_initial_weights, run_training

from .metrics import compute_overfitting_indicators
from .runtime import BATCH_SIZE, STUDY_EPOCHS, STUDY_SEEDS


def _layout_from_config(config: dict) -> str:
    stage_tokens = []
    for stage in config["conv_stages"]:
        conv_token = "-".join([str(stage["filters"])] * stage["convs"])
        stage_tokens.append(f"{conv_token}/pool")
    return " | ".join(stage_tokens)


def _count_params(config: dict) -> int:
    model = build_model(config)
    params = int(model.count_params())
    del model
    return params


def get_variant_configs() -> list[dict]:
    base_m2 = deepcopy(get_architecture_config("m2_three_stage"))
    variant_r0 = deepcopy(base_m2)
    variant_r0.update({"name": "R0", "slug": "r0_m2_baseline", "title": "M2 baseline"})

    variant_r1 = deepcopy(base_m2)
    variant_r1.update({"name": "R1", "slug": "r1_m2_dropout", "title": "M2 + Dropout", "dropout_before_dense": 0.5})
    for stage in variant_r1["conv_stages"]:
        stage["dropout_after_pool"] = 0.25

    variant_r2 = deepcopy(base_m2)
    variant_r2.update({"name": "R2", "slug": "r2_m2_l2", "title": "M2 + L2", "kernel_regularizer_l2": 1e-4})

    variant_r3 = deepcopy(get_architecture_config("m3_three_stage_regularized"))
    variant_r3.update({"name": "R3", "slug": "r3_m3_full_regularization", "title": "M2 + Dropout + L2"})

    variants = [
        {
            "name": "R0",
            "slug": "r0_m2_baseline",
            "title": "M2 baseline",
            "mechanism": "No explicit regularization",
            "description": "Three-stage M2 backbone without dropout and without weight decay.",
            "reference_backbone": "M2",
            "model_config": variant_r0,
        },
        {
            "name": "R1",
            "slug": "r1_m2_dropout",
            "title": "M2 + Dropout",
            "mechanism": "Dropout after pooling and before the dense layer",
            "description": "Same M2 backbone with dropout only.",
            "reference_backbone": "M2",
            "model_config": variant_r1,
        },
        {
            "name": "R2",
            "slug": "r2_m2_l2",
            "title": "M2 + L2",
            "mechanism": "Weight decay only",
            "description": "Same M2 backbone with L2 regularization only.",
            "reference_backbone": "M2",
            "model_config": variant_r2,
        },
        {
            "name": "R3",
            "slug": "r3_m3_full_regularization",
            "title": "M2 + Dropout + L2",
            "mechanism": "Combined dropout and weight decay",
            "description": "Section 6 regularized deep model M3.",
            "reference_backbone": "M3",
            "model_config": variant_r3,
        },
    ]
    for variant in variants:
        variant["params"] = _count_params(variant["model_config"])
        variant["layout"] = _layout_from_config(variant["model_config"])
        variant["dropout_before_dense"] = variant["model_config"]["dropout_before_dense"]
        variant["dropout_after_pool"] = [stage["dropout_after_pool"] for stage in variant["model_config"]["conv_stages"]]
        variant["kernel_regularizer_l2"] = variant["model_config"]["kernel_regularizer_l2"]
    return variants


def run_regularization_study(data: dict, variants: list[dict], verbose: bool = True) -> list[dict]:
    runs = []
    reference_config = variants[0]["model_config"]
    initial_weights_by_seed = {seed: get_initial_weights(reference_config, seed) for seed in STUDY_SEEDS}
    for seed in STUDY_SEEDS:
        for variant in variants:
            if verbose:
                print(f"[section7 study] variant={variant['name']} seed={seed} epochs={STUDY_EPOCHS}")
            run = run_training(
                data=data,
                config=variant["model_config"],
                initial_weights=[weight.copy() for weight in initial_weights_by_seed[seed]],
                batch_size=BATCH_SIZE,
                epochs=STUDY_EPOCHS,
                training_seed=seed,
                evaluate_test=False,
            )
            run["variant_name"] = variant["name"]
            run["variant_slug"] = variant["slug"]
            run["variant_title"] = variant["title"]
            run["mechanism"] = variant["mechanism"]
            run["reference_backbone"] = variant["reference_backbone"]
            run["indicators"] = compute_overfitting_indicators(run)
            runs.append(run)
    return runs
