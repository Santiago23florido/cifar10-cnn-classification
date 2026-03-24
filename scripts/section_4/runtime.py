# Shared constants for the notebook-driven Section 4 workflow.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
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
