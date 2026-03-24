# Section 8 results helper used by the notebook pipeline.
from .activations import (
    build_deep_layer_panel,
    build_epoch_evolution_panel,
    build_first_layer_panel,
    compute_first_layer_diagnostics,
    extract_layer_activations,
    select_tracked_channels,
)
from .data import load_reduced_cifar10_with_raw
from .runtime import (
    CHECKPOINT_EPOCHS,
    DEEP_LAYER_NAMES,
    DEEP_LAYER_TOP_K,
    DEFAULT_RESULTS_PATH,
    EVOLUTION_LAYER_NAMES,
    FIRST_LAYER_NAME,
    PRIMARY_CLASS_NAME,
    REFERENCE_EPOCHS,
    REFERENCE_MODEL_SEED,
    REFERENCE_MODEL_SLUG,
    RESULTS_DIR,
    SECONDARY_CLASS_NAME,
    K,
    json,
    np,
)
from .selection import select_representative_images
from .training import load_reference_model, run_reference_training


def save_results(payload: dict, output_path=DEFAULT_RESULTS_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_results(path=DEFAULT_RESULTS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_section8_protocol() -> dict:
    return {
        "reference_model": "M3 / R3 three-stage CNN with dropout and L2",
        "reference_slug": REFERENCE_MODEL_SLUG,
        "reference_seed": REFERENCE_MODEL_SEED,
        "reference_epochs": REFERENCE_EPOCHS,
        "batch_size": 32,
        "optimizer": "Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)",
        "selected_layers": [FIRST_LAYER_NAME, *DEEP_LAYER_NAMES],
        "evolution_layers": EVOLUTION_LAYER_NAMES,
        "selected_epochs": CHECKPOINT_EPOCHS,
        "selected_classes": {"primary": PRIMARY_CLASS_NAME, "secondary": SECONDARY_CLASS_NAME},
        "visual_normalization": {
            "static_panels": "Per-map min-max scaling to [0,1]",
            "epoch_evolution": "Fixed per-channel scaling across the selected epochs",
        },
    }


def get_section8_audit() -> list[str]:
    return [
        "The original Part IV of the notebook only visualizes an aggregated first-layer mask from the baseline model.",
        "No usable deep-model checkpoints existed before Section 8, so a dedicated M3 reference run is needed to compare epochs 0, 1, 5 and 15.",
        "The best Section 6 / Section 7 model family can be reused directly through M3, which keeps Section 8 coherent with the rest of the TP.",
    ]


def _serialize_selected_image(image_info: dict, raw_image) -> dict:
    return {**image_info, "input_image": raw_image.tolist()}


def _build_summary_table(first_layer: dict, deep_layers: dict, evolution: dict) -> list[dict]:
    conv_s2 = next(row for row in deep_layers["layers"] if row["layer_name"] == "conv_s2_2")
    conv_s3 = next(row for row in deep_layers["layers"] if row["layer_name"] == "conv_s3_2")
    first_positive = np.mean([channel["positive_ratio"] for channel in first_layer["channels"]])
    first_entropy = np.mean([channel["spatial_entropy"] for channel in first_layer["channels"]])

    tracked_conv_s1 = [row for row in evolution["tracked_rows"] if row["layer_name"] == "conv_s1_1"]
    tracked_conv_s3 = [row for row in evolution["tracked_rows"] if row["layer_name"] == "conv_s3_2"]

    def evolution_delta(rows, key: str) -> float:
        start = np.mean([row["epochs"][0][key] for row in rows])
        end = np.mean([row["epochs"][-1][key] for row in rows])
        return float(end - start)

    return [
        {
            "analysis": "First-layer maps",
            "layers": "conv_s1_1",
            "epochs": "15",
            "displayed_channels": "16",
            "observation": (
                f"Mean positive ratio {first_positive:.3f}; normalized entropy {first_entropy:.3f}; "
                "maps remain spatially dense and locally interpretable."
            ),
        },
        {
            "analysis": "Deep-layer maps",
            "layers": "conv_s2_2, conv_s3_2",
            "epochs": "15",
            "displayed_channels": "8 + 8",
            "observation": (
                f"Mean positive ratio decreases from {conv_s2['mean_positive_ratio']:.3f} to {conv_s3['mean_positive_ratio']:.3f}; "
                f"entropy decreases from {conv_s2['mean_spatial_entropy']:.3f} to {conv_s3['mean_spatial_entropy']:.3f}."
            ),
        },
        {
            "analysis": "Evolution during training",
            "layers": "conv_s1_1, conv_s3_2",
            "epochs": ", ".join(str(epoch) for epoch in CHECKPOINT_EPOCHS),
            "displayed_channels": "2 + 2",
            "observation": (
                f"Activation std increases by {evolution_delta(tracked_conv_s1, 'activation_std'):.3f} in conv_s1_1 "
                f"and by {evolution_delta(tracked_conv_s3, 'activation_std'):.3f} in conv_s3_2."
            ),
        },
    ]


def run_section8_pipeline(output_path=DEFAULT_RESULTS_PATH, force_recompute: bool = False, verbose: bool = True) -> dict:
    if not force_recompute and output_path.exists():
        return load_results(output_path)

    data = load_reduced_cifar10_with_raw()
    reference_run = run_reference_training(data, verbose=verbose)

    final_model = load_reference_model(REFERENCE_EPOCHS)
    probabilities = final_model.predict(data["x_test"], verbose=0)
    selected_images = select_representative_images(probabilities, data["y_test_int"])

    primary_index = selected_images["primary"]["index"]
    secondary_index = selected_images["secondary"]["index"]
    primary_image = data["x_test"][primary_index]
    primary_image_raw = data["x_test_raw"][primary_index]
    secondary_image_raw = data["x_test_raw"][secondary_index]

    final_activations = extract_layer_activations(final_model, primary_image, [FIRST_LAYER_NAME, *DEEP_LAYER_NAMES])
    first_layer_panel = build_first_layer_panel(primary_image_raw, final_activations[FIRST_LAYER_NAME])
    first_layer_diagnostics = compute_first_layer_diagnostics(primary_image_raw, final_activations[FIRST_LAYER_NAME])
    deep_layer_panel = build_deep_layer_panel(
        primary_image_raw,
        {name: final_activations[name] for name in DEEP_LAYER_NAMES},
        top_k=DEEP_LAYER_TOP_K,
    )
    tracked_channels = select_tracked_channels(
        {
            FIRST_LAYER_NAME: final_activations[FIRST_LAYER_NAME],
            "conv_s3_2": final_activations["conv_s3_2"],
        }
    )
    del final_model
    K.clear_session()

    activations_by_epoch = {}
    for epoch in CHECKPOINT_EPOCHS:
        epoch_model = load_reference_model(epoch)
        activations_by_epoch[epoch] = extract_layer_activations(epoch_model, primary_image, EVOLUTION_LAYER_NAMES)
        del epoch_model
        K.clear_session()

    evolution_panel = build_epoch_evolution_panel(primary_image_raw, activations_by_epoch, tracked_channels)

    payload = {
        "meta": {"audit": get_section8_audit(), "protocol": get_section8_protocol()},
        "reference_run": reference_run,
        "selected_images": {
            "primary": _serialize_selected_image(selected_images["primary"], primary_image_raw),
            "secondary": _serialize_selected_image(selected_images["secondary"], secondary_image_raw),
        },
        "first_layer": {
            "panel": first_layer_panel,
            "diagnostics": first_layer_diagnostics,
        },
        "deep_layers": deep_layer_panel,
        "evolution": {
            "tracked_channels": tracked_channels,
            "panel": evolution_panel,
        },
    }
    payload["summary_table"] = _build_summary_table(
        first_layer=payload["first_layer"]["panel"],
        deep_layers=payload["deep_layers"],
        evolution=payload["evolution"]["panel"],
    )
    save_results(payload, output_path)
    return payload