# Section 8 data helper used to keep raw and standardized images aligned.
from .runtime import (
    CLASS_NAMES,
    N_OTHER_SAMPLES,
    N_TRAINING_SAMPLES,
    N_VALID_SAMPLES,
    SPLIT_SEED,
    load_cifar10_from_local_archive,
    np,
    standardize,
)


def load_reduced_cifar10_with_raw(split_seed: int = SPLIT_SEED) -> dict:
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_cifar10_from_local_archive()
    rng = np.random.default_rng(split_seed)

    train_ids = rng.choice(len(x_train_full), size=N_TRAINING_SAMPLES, replace=False)
    other_ids = rng.choice(len(x_test_full), size=N_OTHER_SAMPLES, replace=False)
    val_ids = other_ids[:N_VALID_SAMPLES]
    test_ids = other_ids[N_VALID_SAMPLES:]

    x_train_raw = x_train_full[train_ids].astype("uint8")
    y_train_int = y_train_full[train_ids].astype("int64").reshape(-1)
    x_val_raw = x_test_full[val_ids].astype("uint8")
    y_val_int = y_test_full[val_ids].astype("int64").reshape(-1)
    x_test_raw = x_test_full[test_ids].astype("uint8")
    y_test_int = y_test_full[test_ids].astype("int64").reshape(-1)

    x_train = standardize(x_train_raw).astype("float32")
    x_val = standardize(x_val_raw).astype("float32")
    x_test = standardize(x_test_raw).astype("float32")

    y_train = np.eye(len(CLASS_NAMES), dtype="float32")[y_train_int]
    y_val = np.eye(len(CLASS_NAMES), dtype="float32")[y_val_int]
    y_test = np.eye(len(CLASS_NAMES), dtype="float32")[y_test_int]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_int": y_train_int,
        "x_val": x_val,
        "y_val": y_val,
        "y_val_int": y_val_int,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_int": y_test_int,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
    }