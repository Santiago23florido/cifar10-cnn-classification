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

def standardize(img_data: np.ndarray) -> np.ndarray:
    img_data_mean = np.mean(img_data, axis=(1, 2), keepdims=True)
    img_data_std = np.std(img_data, axis=(1, 2), keepdims=True)
    return (img_data - img_data_mean) / img_data_std
