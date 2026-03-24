# Shared constants for the notebook-driven Section 7 workflow.
import json
from pathlib import Path
from typing import Any

from scripts.section_4 import REPO_ROOT, load_reduced_cifar10
from scripts.section_6 import BATCH_SIZE as SECTION6_BATCH_SIZE
from scripts.section_6 import DEFAULT_RESULTS_PATH as SECTION6_RESULTS_PATH
from scripts.section_6 import OPTIMIZER_CONFIG as SECTION6_OPTIMIZER_CONFIG

RESULTS_DIR = REPO_ROOT / "results" / "section7"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "section7_results.json"

STUDY_SEEDS = [42, 314]
STUDY_EPOCHS = 15
BATCH_SIZE = SECTION6_BATCH_SIZE
OPTIMIZER_CONFIG = SECTION6_OPTIMIZER_CONFIG

OVERFITTING_SOURCE = {
    "section": 6,
    "stage": "screening",
    "architecture_slug": "m2_three_stage",
    "architecture_name": "M2",
    "seed": 42,
    "epochs": 10,
}
