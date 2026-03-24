# Shared constants for the notebook-driven Section 8 workflow.
import json
from pathlib import Path
from typing import Any

import numpy as np
from keras import Model
from keras import backend as K
from keras.losses import CategoricalCrossentropy

from scripts.section_4 import (
    LOSS_REDUCTION,
    N_OTHER_SAMPLES,
    N_TRAINING_SAMPLES,
    N_VALID_SAMPLES,
    REPO_ROOT,
    SPLIT_SEED,
    build_optimizer,
    load_cifar10_from_local_archive,
    standardize,
)
from scripts.section_6 import CLASS_NAMES, build_model, get_architecture_config, get_initial_weights
from scripts.section_7 import BATCH_SIZE as SECTION7_BATCH_SIZE
from scripts.section_7 import OPTIMIZER_CONFIG as SECTION7_OPTIMIZER_CONFIG
from scripts.section_7 import STUDY_EPOCHS as SECTION7_STUDY_EPOCHS

RESULTS_DIR = REPO_ROOT / "results" / "section8"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "section8_results.json"

REFERENCE_MODEL_SLUG = "m3_three_stage_regularized"
REFERENCE_MODEL_SEED = 42
REFERENCE_EPOCHS = SECTION7_STUDY_EPOCHS
BATCH_SIZE = SECTION7_BATCH_SIZE
OPTIMIZER_CONFIG = SECTION7_OPTIMIZER_CONFIG
CHECKPOINT_EPOCHS = [0, 1, 5, 15]

PRIMARY_CLASS_NAME = "truck"
SECONDARY_CLASS_NAME = "frog"

FIRST_LAYER_NAME = "conv_s1_1"
DEEP_LAYER_NAMES = ["conv_s2_2", "conv_s3_2"]
EVOLUTION_LAYER_NAMES = ["conv_s1_1", "conv_s3_2"]
FIRST_LAYER_CHANNELS = 16
DEEP_LAYER_TOP_K = 8
EVOLUTION_TOP_K = 2
