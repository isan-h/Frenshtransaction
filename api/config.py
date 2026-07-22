from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
CATEGORY_ENCODER_PATH = MODELS_DIR / "label_encoder_category.joblib"
MERCHANT_ENCODER_PATH = MODELS_DIR / "label_encoder_merchant.joblib"

MODEL_NAME = "logistic_regression"
CATEGORY_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_category.joblib"
MERCHANT_MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_merchant.joblib"

API_TITLE = "Transaction Classification API"
API_DESCRIPTION = (
    "Classifies bank transaction descriptions into a category and merchant, "
    "in real time, simulating an Open Banking transaction feed."
)
API_VERSION = "1.0.0"

MIN_DESCRIPTION_LENGTH = 3
MAX_DESCRIPTION_LENGTH = 500

LOG_LEVEL = "INFO"
