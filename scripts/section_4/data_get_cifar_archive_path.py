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
