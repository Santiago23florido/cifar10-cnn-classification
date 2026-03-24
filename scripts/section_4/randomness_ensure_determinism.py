from .runtime import keras_utils, np, random, tf

def ensure_determinism() -> None:
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
