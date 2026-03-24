# Shared constants for the notebook-driven Section 6 workflow.
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from keras import Input
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.regularizers import l2

from scripts.section_4 import (
    LOSS_REDUCTION,
    REPO_ROOT,
    TimingHistory,
    build_optimizer,
    load_reduced_cifar10,
    set_global_seed,
)

RESULTS_DIR = REPO_ROOT / "results" / "section6"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "section6_results.json"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

SCREENING_SEEDS = [42]
SCREENING_EPOCHS = 10
CONFIRMATION_SEEDS = [42, 314]
CONFIRMATION_EPOCHS = 12
BATCH_SIZE = 32

OPTIMIZER_CONFIG = {
    "name": "Adam",
    "slug": "adam",
    "learning_rate": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
}

ARCHITECTURE_CONFIGS = [
    {
        "name": "M0",
        "slug": "m0_baseline",
        "title": "Baseline",
        "description": "Single convolution baseline used in the previous sections.",
        "conv_stages": [{"filters": 8, "convs": 1, "pool": True, "dropout_after_pool": 0.0}],
        "dense_units": 64,
        "kernel_size": (3, 3),
        "padding": "same",
        "pool_size": (2, 2),
        "kernel_regularizer_l2": 0.0,
        "dropout_before_dense": 0.0,
        "output_units": 10,
    },
    {
        "name": "M1",
        "slug": "m1_two_stage",
        "title": "Two-stage CNN",
        "description": "Two convolutional stages with modest depth increase.",
        "conv_stages": [
            {"filters": 16, "convs": 2, "pool": True, "dropout_after_pool": 0.0},
            {"filters": 32, "convs": 2, "pool": True, "dropout_after_pool": 0.0},
        ],
        "dense_units": 64,
        "kernel_size": (3, 3),
        "padding": "same",
        "pool_size": (2, 2),
        "kernel_regularizer_l2": 0.0,
        "dropout_before_dense": 0.0,
        "output_units": 10,
    },
    {
        "name": "M2",
        "slug": "m2_three_stage",
        "title": "Three-stage CNN",
        "description": "Three convolutional stages to enlarge the receptive field and reduce the dense bottleneck.",
        "conv_stages": [
            {"filters": 16, "convs": 2, "pool": True, "dropout_after_pool": 0.0},
            {"filters": 32, "convs": 2, "pool": True, "dropout_after_pool": 0.0},
            {"filters": 64, "convs": 2, "pool": True, "dropout_after_pool": 0.0},
        ],
        "dense_units": 64,
        "kernel_size": (3, 3),
        "padding": "same",
        "pool_size": (2, 2),
        "kernel_regularizer_l2": 0.0,
        "dropout_before_dense": 0.0,
        "output_units": 10,
    },
    {
        "name": "M3",
        "slug": "m3_three_stage_regularized",
        "title": "Regularized deep CNN",
        "description": "Same backbone as M2 with dropout and L2 regularization.",
        "conv_stages": [
            {"filters": 16, "convs": 2, "pool": True, "dropout_after_pool": 0.25},
            {"filters": 32, "convs": 2, "pool": True, "dropout_after_pool": 0.25},
            {"filters": 64, "convs": 2, "pool": True, "dropout_after_pool": 0.25},
        ],
        "dense_units": 64,
        "kernel_size": (3, 3),
        "padding": "same",
        "pool_size": (2, 2),
        "kernel_regularizer_l2": 1e-4,
        "dropout_before_dense": 0.5,
        "output_units": 10,
    },
]
