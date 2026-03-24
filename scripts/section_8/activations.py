# Section 8 activation helper used by the notebook pipeline.
from .runtime import EVOLUTION_TOP_K, Model, np


def extract_layer_activations(model, image: np.ndarray, layer_names: list[str]) -> dict[str, np.ndarray]:
    extractor = Model(inputs=model.inputs, outputs=[model.get_layer(name).output for name in layer_names])
    outputs = extractor.predict(image[None, ...], verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]
    activations = {name: np.asarray(output[0], dtype=np.float64) for name, output in zip(layer_names, outputs)}
    del extractor
    return activations


def normalize_map(map_2d: np.ndarray) -> np.ndarray:
    minimum = float(np.min(map_2d))
    maximum = float(np.max(map_2d))
    if maximum - minimum < 1e-12:
        return np.zeros_like(map_2d, dtype=np.float64)
    return (map_2d - minimum) / (maximum - minimum)


def rank_channels_by_mean(activation_tensor: np.ndarray, top_k: int) -> list[int]:
    channel_means = np.mean(activation_tensor, axis=(0, 1))
    return [int(index) for index in np.argsort(channel_means)[::-1][:top_k]]


def positive_ratio(map_2d: np.ndarray) -> float:
    return float(np.mean(map_2d > 0.0))


def spatial_entropy(map_2d: np.ndarray) -> float:
    positive = np.maximum(map_2d, 0.0)
    total = float(np.sum(positive))
    if total <= 1e-12:
        return 0.0
    probabilities = positive.reshape(-1) / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(map_2d.size))


def activation_std(map_2d: np.ndarray) -> float:
    return float(np.std(map_2d))


def _box_blur(gray_image: np.ndarray) -> np.ndarray:
    padded = np.pad(gray_image, 1, mode="reflect")
    windows = []
    for dy in range(3):
        for dx in range(3):
            windows.append(padded[dy : dy + gray_image.shape[0], dx : dx + gray_image.shape[1]])
    return np.mean(np.stack(windows, axis=0), axis=0)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    a_std = float(np.std(a_flat))
    b_std = float(np.std(b_flat))
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_first_layer_diagnostics(raw_image: np.ndarray, activation_tensor: np.ndarray) -> dict:
    image = raw_image.astype(np.float64) / 255.0
    gray = np.mean(image, axis=2)
    grad_y, grad_x = np.gradient(gray)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    horizontal_edges = np.abs(grad_x)
    vertical_edges = np.abs(grad_y)
    local_contrast = np.abs(gray - _box_blur(gray))
    color_contrast = np.std(image, axis=2)

    rows = []
    for channel_index in range(activation_tensor.shape[-1]):
        normalized = normalize_map(activation_tensor[:, :, channel_index])
        rows.append(
            {
                "channel_index": int(channel_index),
                "mean_activation": float(np.mean(activation_tensor[:, :, channel_index])),
                "positive_ratio": positive_ratio(activation_tensor[:, :, channel_index]),
                "edge_corr": _corrcoef(normalized, edge_magnitude),
                "horizontal_corr": _corrcoef(normalized, horizontal_edges),
                "vertical_corr": _corrcoef(normalized, vertical_edges),
                "local_contrast_corr": _corrcoef(normalized, local_contrast),
                "color_corr": _corrcoef(normalized, color_contrast),
            }
        )

    def top_channels(metric: str, limit: int = 3) -> list[int]:
        ordered = sorted(rows, key=lambda row: row[metric], reverse=True)
        return [int(row["channel_index"]) for row in ordered[:limit]]

    return {
        "per_channel": rows,
        "top_edge_channels": top_channels("edge_corr"),
        "top_horizontal_channels": top_channels("horizontal_corr"),
        "top_vertical_channels": top_channels("vertical_corr"),
        "top_color_channels": top_channels("color_corr"),
        "top_local_contrast_channels": top_channels("local_contrast_corr"),
    }


def build_first_layer_panel(raw_image: np.ndarray, activation_tensor: np.ndarray) -> dict:
    channels = []
    for channel_index in range(activation_tensor.shape[-1]):
        map_2d = activation_tensor[:, :, channel_index]
        channels.append(
            {
                "channel_index": int(channel_index),
                "map": normalize_map(map_2d).tolist(),
                "positive_ratio": positive_ratio(map_2d),
                "spatial_entropy": spatial_entropy(map_2d),
            }
        )
    return {
        "input_image": raw_image.tolist(),
        "layer_name": "conv_s1_1",
        "shape": list(activation_tensor.shape),
        "channels": channels,
    }


def build_deep_layer_panel(raw_image: np.ndarray, activation_dict: dict[str, np.ndarray], top_k: int) -> dict:
    layer_summaries = []
    for layer_name, activation_tensor in activation_dict.items():
        top_channels = rank_channels_by_mean(activation_tensor, top_k)
        channels = []
        for channel_index in top_channels:
            map_2d = activation_tensor[:, :, channel_index]
            channels.append(
                {
                    "channel_index": int(channel_index),
                    "map": normalize_map(map_2d).tolist(),
                    "mean_activation": float(np.mean(map_2d)),
                    "positive_ratio": positive_ratio(map_2d),
                    "spatial_entropy": spatial_entropy(map_2d),
                }
            )
        layer_summaries.append(
            {
                "layer_name": layer_name,
                "shape": list(activation_tensor.shape),
                "selected_channels": channels,
                "mean_positive_ratio": float(np.mean([channel["positive_ratio"] for channel in channels])),
                "mean_spatial_entropy": float(np.mean([channel["spatial_entropy"] for channel in channels])),
            }
        )
    return {"input_image": raw_image.tolist(), "layers": layer_summaries}


def select_tracked_channels(final_activations: dict[str, np.ndarray]) -> dict[str, list[int]]:
    tracked = {}
    for layer_name, activation_tensor in final_activations.items():
        tracked[layer_name] = rank_channels_by_mean(activation_tensor, EVOLUTION_TOP_K)
    return tracked


def build_epoch_evolution_panel(raw_image: np.ndarray, activations_by_epoch: dict[int, dict[str, np.ndarray]], tracked_channels: dict[str, list[int]]) -> dict:
    layer_rows = []
    for layer_name, channel_indices in tracked_channels.items():
        for channel_index in channel_indices:
            raw_maps = {epoch: activations_by_epoch[epoch][layer_name][:, :, channel_index] for epoch in activations_by_epoch}
            global_min = min(float(np.min(map_2d)) for map_2d in raw_maps.values())
            global_max = max(float(np.max(map_2d)) for map_2d in raw_maps.values())
            epoch_rows = []
            for epoch in sorted(raw_maps):
                map_2d = raw_maps[epoch]
                if global_max - global_min < 1e-12:
                    normalized = np.zeros_like(map_2d, dtype=np.float64)
                else:
                    normalized = (map_2d - global_min) / (global_max - global_min)
                epoch_rows.append(
                    {
                        "epoch": int(epoch),
                        "map": normalized.tolist(),
                        "positive_ratio": positive_ratio(map_2d),
                        "spatial_entropy": spatial_entropy(map_2d),
                        "activation_std": activation_std(map_2d),
                    }
                )
            layer_rows.append({"layer_name": layer_name, "channel_index": int(channel_index), "epochs": epoch_rows})
    return {"input_image": raw_image.tolist(), "tracked_rows": layer_rows}