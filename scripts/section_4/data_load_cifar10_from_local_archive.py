# Section 4 data helper reused by the notebook experiments.
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

from .data_get_cifar_archive_path import get_cifar_archive_path
from .data_load_cifar_batch_from_tar import load_cifar_batch_from_tar

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
