from .runtime import (
    Any,
    N_OTHER_SAMPLES,
    N_TRAINING_SAMPLES,
    N_VALID_SAMPLES,
    SPLIT_SEED,
    Path,
    keras_utils,
    np,
    pickle,
    tarfile,
)

from .data_load_cifar10_from_local_archive import load_cifar10_from_local_archive
from .data_standardize import standardize

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
