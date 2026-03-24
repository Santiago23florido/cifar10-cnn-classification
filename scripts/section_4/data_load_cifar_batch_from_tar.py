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

def load_cifar_batch_from_tar(archive: tarfile.TarFile, member_name: str):
    member = archive.getmember(member_name)
    with archive.extractfile(member) as file_obj:
        if file_obj is None:
            raise FileNotFoundError(
                f"Unable to read member {member_name} from CIFAR-10 archive."
            )
        batch = pickle.load(file_obj, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"], dtype="uint8")
    return data, labels
