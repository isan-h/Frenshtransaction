"""
config.py
---------
Single source of truth for every path and setting the API needs.

Why this matters: without it, "models/feature_pipeline.joblib" would be
typed as a literal string in 3-4 different files. Change the folder
structure once (e.g. deploying inside Docker where paths differ) and
you'd have to hunt down every occurrence. One object, imported
everywhere, means one place to change.
"""

from pathlib import Path

# BASE_DIR = the project root (frenchtransaction/), computed from this
# file's location -- NOT hardcoded -- so this works regardless of which
# directory you run `uvicorn` from.
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
CATEGORY_ENCODER_PATH = MODELS_DIR / "label_encoder_category.joblib"
MERCHANT_ENCODER_PATH = MODELS_DIR / "label_encoder_merchant.joblib"

# The final chosen model -- change this ONE line if you ever retrain
# and pick a different model, nothing else in the API needs to change.
MODEL_NAME = "logistic_regression"
CATEGORY_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_category.joblib"
MERCHANT_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_merchant.joblib"

# API metadata (shown in the auto-generated docs at /docs)
API_TITLE = "Transaction Classification API"
API_DESCRIPTION = (
    "Classifies bank transaction descriptions into a category and merchant, "
    "in real time, simulating an Open Banking transaction feed."
)
API_VERSION = "1.0.0"

# Validation rules
MIN_DESCRIPTION_LENGTH = 3
MAX_DESCRIPTION_LENGTH = 500

LOG_LEVEL = "INFO"
