import json
import math
import os
import pickle
import random
import tarfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras import utils as keras_utils
from keras.callbacks import Callback
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.regularizers import l2


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "section4"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "section4_results.json"

SPLIT_SEED = 42
TRAINING_SEEDS = [42, 314]
N_TRAINING_SAMPLES = 5000
N_OTHER_SAMPLES = 2000
N_VALID_SAMPLES = N_OTHER_SAMPLES // 2
BATCH_SIZES = [8, 16, 32, 64, 128]
BATCH_STUDY_EPOCHS = 8
OPTIMIZER_STUDY_EPOCHS = 8
OPTIMIZER_BATCH_SIZE = 32
LOSS_REDUCTION = "sum_over_batch_size"
ADAM_EPSILON = 1e-7

OPTIMIZER_CONFIGS = [
    {
        "name": "SGD",
        "slug": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.0,
    },
    {
        "name": "SGD+Momentum",
        "slug": "sgd_momentum",
        "learning_rate": 0.01,
        "momentum": 0.9,
    },
    {
        "name": "Adam",
        "slug": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": ADAM_EPSILON,
    },
]


class TimingHistory(Callback):
    def on_train_begin(self, logs=None):
        self.batch_times = []
        self.epoch_times = []
        self._batch_start = None
        self._epoch_start = None

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is None:
            return
        self.batch_times.append(time.perf_counter() - self._batch_start)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start is None:
            return
        self.epoch_times.append(time.perf_counter() - self._epoch_start)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras_utils.set_random_seed(seed)


def standardize(img_data: np.ndarray) -> np.ndarray:
    img_data_mean = np.mean(img_data, axis=(1, 2), keepdims=True)
    img_data_std = np.std(img_data, axis=(1, 2), keepdims=True)
    return (img_data - img_data_mean) / img_data_std


def get_cifar_archive_path() -> Path:
    archive_candidates = [
        Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target_archive",
        Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target.tar.gz",
    ]
    archive_path = next((path for path in archive_candidates if path.exists()), None)
    if archive_path is None:
        raise FileNotFoundError(
            "Unable to find a local CIFAR-10 archive in the user Keras cache."
        )
    return archive_path


def load_cifar_batch_from_tar(archive: tarfile.TarFile, member_name: str):
    member = archive.getmember(member_name)
    with archive.extractfile(member) as file_obj:
        if file_obj is None:
            raise FileNotFoundError(f"Unable to read member {member_name} from CIFAR-10 archive.")
        batch = pickle.load(file_obj, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"], dtype="uint8")
    return data, labels


def load_cifar10_from_local_archive():
    archive_path = get_cifar_archive_path()
    x_train = np.empty((50000, 3, 32, 32), dtype="uint8")
    y_train = np.empty((50000,), dtype="uint8")

    with tarfile.open(archive_path, "r:*") as archive:
        for batch_index in range(1, 6):
            start = (batch_index - 1) * 10000
            stop = batch_index * 10000
            data, labels = load_cifar_batch_from_tar(
                archive,
                f"cifar-10-batches-py/data_batch_{batch_index}",
            )
            x_train[start:stop] = data
            y_train[start:stop] = labels

        x_test, y_test = load_cifar_batch_from_tar(archive, "cifar-10-batches-py/test_batch")

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return (x_train, y_train), (x_test, y_test)


def load_reduced_cifar10(split_seed: int = SPLIT_SEED) -> dict[str, Any]:
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_cifar10_from_local_archive()
    rng = np.random.default_rng(split_seed)

    train_ids = rng.choice(len(x_train_full), size=N_TRAINING_SAMPLES, replace=False)
    other_ids = rng.choice(len(x_test_full), size=N_OTHER_SAMPLES, replace=False)
    val_ids = other_ids[:N_VALID_SAMPLES]
    test_ids = other_ids[N_VALID_SAMPLES:]

    x_train_initial = x_train_full[train_ids]
    y_train = y_train_full[train_ids]
    x_val_initial = x_test_full[val_ids]
    y_val = y_test_full[val_ids]
    x_test_initial = x_test_full[test_ids]
    y_test = y_test_full[test_ids]

    x_train = standardize(x_train_initial)
    x_val = standardize(x_val_initial)
    x_test = standardize(x_test_initial)

    y_train = keras_utils.to_categorical(y_train)
    y_val = keras_utils.to_categorical(y_val)
    y_test = keras_utils.to_categorical(y_test)

    return {
        "x_train": x_train.astype("float32"),
        "y_train": y_train.astype("float32"),
        "x_val": x_val.astype("float32"),
        "y_val": y_val.astype("float32"),
        "x_test": x_test.astype("float32"),
        "y_test": y_test.astype("float32"),
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "test_shape": list(x_test.shape),
    }


def build_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(
        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(0.00),
        )
    )
    model.add(Dropout(0.0))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.00)))
    model.add(Dropout(0.0))
    model.add(Dense(10, activation="softmax", kernel_regularizer=l2(0.00)))
    model.add(Dropout(0.0))
    return model


def build_optimizer(config: dict[str, Any]):
    if config["slug"] == "adam":
        return Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
        )
    return SGD(
        learning_rate=config["learning_rate"],
        momentum=config.get("momentum", 0.0),
    )


def get_initial_weights(seed: int):
    K.clear_session()
    set_global_seed(seed)
    model = build_model()
    weights = model.get_weights()
    del model
    K.clear_session()
    return weights


def run_training(
    data: dict[str, Any],
    initial_weights,
    optimizer_config: dict[str, Any],
    batch_size: int,
    epochs: int,
    training_seed: int,
) -> dict[str, Any]:
    K.clear_session()
    set_global_seed(training_seed)
    model = build_model()
    model.set_weights(initial_weights)
    model.compile(
        optimizer=build_optimizer(optimizer_config),
        loss=CategoricalCrossentropy(reduction=LOSS_REDUCTION),
        metrics=["accuracy"],
    )

    timing = TimingHistory()
    history = model.fit(
        data["x_train"],
        data["y_train"],
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        shuffle=True,
        validation_data=(data["x_val"], data["y_val"]),
        callbacks=[timing],
    )

    run = {
        "seed": training_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": math.ceil(len(data["x_train"]) / batch_size),
        "optimizer": optimizer_config,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "timing": {
            "step_times": [float(v) for v in timing.batch_times],
            "epoch_times": [float(v) for v in timing.epoch_times],
        },
    }

    del model
    K.clear_session()
    return run


def stack_metric(runs: list[dict[str, Any]], metric_name: str) -> np.ndarray:
    return np.array([run["history"][metric_name] for run in runs], dtype=np.float64)


def aggregate_runs(runs: list[dict[str, Any]], group_key: str, labels: list[Any]) -> list[dict[str, Any]]:
    aggregated = []
    for label in labels:
        matching = [run for run in runs if run[group_key] == label]
        step_times = np.concatenate(
            [np.array(run["timing"]["step_times"], dtype=np.float64) for run in matching]
        )
        epoch_times = np.concatenate(
            [np.array(run["timing"]["epoch_times"], dtype=np.float64) for run in matching]
        )
        aggregated.append(
            {
                group_key: label,
                "steps_per_epoch": int(matching[0]["steps_per_epoch"]),
                "mean_step_time": float(step_times.mean()),
                "std_step_time": float(step_times.std(ddof=0)),
                "mean_epoch_time": float(epoch_times.mean()),
                "std_epoch_time": float(epoch_times.std(ddof=0)),
                "mean_final_val_accuracy": float(
                    np.mean([run["history"]["val_accuracy"][-1] for run in matching])
                ),
                "mean_min_val_loss": float(
                    np.mean([min(run["history"]["val_loss"]) for run in matching])
                ),
                "curves": {
                    "loss_mean": stack_metric(matching, "loss").mean(axis=0).tolist(),
                    "loss_std": stack_metric(matching, "loss").std(axis=0, ddof=0).tolist(),
                    "val_loss_mean": stack_metric(matching, "val_loss").mean(axis=0).tolist(),
                    "val_loss_std": stack_metric(matching, "val_loss").std(axis=0, ddof=0).tolist(),
                    "accuracy_mean": stack_metric(matching, "accuracy").mean(axis=0).tolist(),
                    "accuracy_std": stack_metric(matching, "accuracy").std(axis=0, ddof=0).tolist(),
                    "val_accuracy_mean": stack_metric(matching, "val_accuracy")
                    .mean(axis=0)
                    .tolist(),
                    "val_accuracy_std": stack_metric(matching, "val_accuracy")
                    .std(axis=0, ddof=0)
                    .tolist(),
                },
            }
        )
    return aggregated


def run_batch_size_study(data: dict[str, Any], verbose: bool = True):
    runs = []
    for seed in TRAINING_SEEDS:
        initial_weights = get_initial_weights(seed)
        for batch_size in BATCH_SIZES:
            if verbose:
                print(
                    f"[batch-study] seed={seed} batch_size={batch_size} epochs={BATCH_STUDY_EPOCHS}",
                    flush=True,
                )
            run = run_training(
                data=data,
                initial_weights=initial_weights,
                optimizer_config=OPTIMIZER_CONFIGS[0],
                batch_size=batch_size,
                epochs=BATCH_STUDY_EPOCHS,
                training_seed=seed,
            )
            runs.append(run)
    return runs, aggregate_runs(runs, "batch_size", BATCH_SIZES)


def run_optimizer_study(data: dict[str, Any], verbose: bool = True):
    runs = []
    optimizer_labels = [config["name"] for config in OPTIMIZER_CONFIGS]
    for seed in TRAINING_SEEDS:
        initial_weights = get_initial_weights(seed)
        for optimizer_config in OPTIMIZER_CONFIGS:
            if verbose:
                print(
                    "[optimizer-study] "
                    f"seed={seed} optimizer={optimizer_config['name']} epochs={OPTIMIZER_STUDY_EPOCHS}",
                    flush=True,
                )
            run = run_training(
                data=data,
                initial_weights=initial_weights,
                optimizer_config=optimizer_config,
                batch_size=OPTIMIZER_BATCH_SIZE,
                epochs=OPTIMIZER_STUDY_EPOCHS,
                training_seed=seed,
            )
            run["optimizer_name"] = optimizer_config["name"]
            runs.append(run)
    return runs, aggregate_runs(runs, "optimizer_name", optimizer_labels)


def make_results_payload(verbose: bool = True) -> dict[str, Any]:
    data = load_reduced_cifar10(SPLIT_SEED)
    batch_runs, batch_summary = run_batch_size_study(data, verbose=verbose)
    optimizer_runs, optimizer_summary = run_optimizer_study(data, verbose=verbose)
    return {
        "meta": {
            "split_seed": SPLIT_SEED,
            "training_seeds": TRAINING_SEEDS,
            "batch_sizes": BATCH_SIZES,
            "batch_study_epochs": BATCH_STUDY_EPOCHS,
            "optimizer_study_epochs": OPTIMIZER_STUDY_EPOCHS,
            "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
            "train_shape": data["train_shape"],
            "val_shape": data["val_shape"],
            "test_shape": data["test_shape"],
            "course_sources": [
                "cours/3_Classification_Apprentissage.pdf",
                "cours/Poly_Chap4_ClassifML.pdf",
            ],
            "optimizer_configs": OPTIMIZER_CONFIGS,
            "loss_reduction": LOSS_REDUCTION,
            "step_time_definition": "mean over the recorded training-batch wall times",
            "epoch_time_definition": "mean over per-epoch wall times including validation",
            "batch_size_comparison_budget": "fixed number of epochs",
            "optimizer_comparison_budget": "fixed number of epochs",
        },
        "batch_size_study": {
            "runs": batch_runs,
            "summary": batch_summary,
        },
        "optimizer_study": {
            "runs": optimizer_runs,
            "summary": optimizer_summary,
        },
    }


def load_results(path: Path | str = DEFAULT_RESULTS_PATH) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(payload: dict[str, Any], path: Path | str = DEFAULT_RESULTS_PATH) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_section4_pipeline(
    output_path: Path | str = DEFAULT_RESULTS_PATH,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and not force_recompute:
        if verbose:
            print(f"Loading existing Section 4 results from {output_path}")
        return load_results(output_path)

    payload = make_results_payload(verbose=verbose)
    saved_path = save_results(payload, output_path)
    if verbose:
        print(f"Section 4 results written to {saved_path}")
    return payload


def get_batch_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload["batch_size_study"]["summary"]:
        rows.append(
            {
                "Batch size": row["batch_size"],
                "Steps/epoch": row["steps_per_epoch"],
                "Mean step time (s)": f"{row['mean_step_time']:.4f}",
                "Std step time (s)": f"{row['std_step_time']:.4f}",
                "Mean epoch time (s)": f"{row['mean_epoch_time']:.3f}",
                "Std epoch time (s)": f"{row['std_epoch_time']:.3f}",
                "Final val. acc. (%)": f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                "Min val. loss": f"{row['mean_min_val_loss']:.4f}",
            }
        )
    return rows


def get_optimizer_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload["optimizer_study"]["summary"]:
        rows.append(
            {
                "Optimizer": row["optimizer_name"],
                "Batch size": payload["meta"]["optimizer_batch_size"],
                "Mean epoch time (s)": f"{row['mean_epoch_time']:.3f}",
                "Std epoch time (s)": f"{row['std_epoch_time']:.3f}",
                "Final val. acc. (%)": f"{100.0 * row['mean_final_val_accuracy']:.2f}",
                "Min val. loss": f"{row['mean_min_val_loss']:.4f}",
            }
        )
    return rows


def get_section4_protocol() -> dict[str, Any]:
    return {
        "batch_sizes": BATCH_SIZES,
        "optimizers": [config["name"] for config in OPTIMIZER_CONFIGS],
        "training_seeds": TRAINING_SEEDS,
        "split_seed": SPLIT_SEED,
        "batch_study_epochs": BATCH_STUDY_EPOCHS,
        "optimizer_study_epochs": OPTIMIZER_STUDY_EPOCHS,
        "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
        "subset_sizes": {
            "train": N_TRAINING_SAMPLES,
            "validation": N_VALID_SAMPLES,
            "test": N_VALID_SAMPLES,
        },
        "optimizer_hyperparameters": {config["name"]: dict(config) for config in OPTIMIZER_CONFIGS},
        "loss_reduction": LOSS_REDUCTION,
        "step_time_definition": "mean over the recorded training-batch wall times",
        "epoch_time_definition": "per-epoch wall time including validation",
        "batch_size_comparison_budget": "fixed number of epochs",
        "optimizer_comparison_budget": "fixed number of epochs",
    }


def ensure_determinism() -> None:
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


__all__ = [
    "BATCH_SIZES",
    "BATCH_STUDY_EPOCHS",
    "DEFAULT_RESULTS_PATH",
    "OPTIMIZER_BATCH_SIZE",
    "OPTIMIZER_CONFIGS",
    "OPTIMIZER_STUDY_EPOCHS",
    "RESULTS_DIR",
    "TRAINING_SEEDS",
    "build_model",
    "ensure_determinism",
    "get_batch_summary_rows",
    "get_optimizer_summary_rows",
    "get_section4_protocol",
    "load_results",
    "run_section4_pipeline",
    "save_results",
]
