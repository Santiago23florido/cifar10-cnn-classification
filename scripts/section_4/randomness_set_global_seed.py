from .runtime import keras_utils, np, random, tf

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras_utils.set_random_seed(seed)
