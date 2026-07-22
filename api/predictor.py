import logging
import sys
from pathlib import Path
from api.model_loader import ModelBundle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import clean_text  

logger = logging.getLogger(__name__)


def predict(description: str, bundle: ModelBundle) -> dict:
    cleaned = clean_text(description)
    logger.info(f"Predicting for description: {description!r} (cleaned: {cleaned!r})")

    X = bundle.feature_pipeline.transform([cleaned])


    category_pred_encoded = bundle.category_model.predict(X)[0]
    full_category = bundle.category_encoder.inverse_transform([category_pred_encoded])[0]
    if " / " in full_category:
        primary_category, detailed_category = full_category.split(" / ", 1)
    else:
        primary_category, detailed_category = full_category, full_category

    category_confidence = None
    if hasattr(bundle.category_model, "predict_proba"):
        category_confidence = float(bundle.category_model.predict_proba(X)[0].max())


    merchant_pred_encoded = bundle.merchant_model.predict(X)[0]
    merchant = bundle.merchant_encoder.inverse_transform([merchant_pred_encoded])[0]

    merchant_confidence = None
    if hasattr(bundle.merchant_model, "predict_proba"):
        merchant_confidence = float(bundle.merchant_model.predict_proba(X)[0].max())

    result = {
        "description": description,
        "category": full_category,
        "primary_category": primary_category,
        "detailed_category": detailed_category,
        "category_confidence": category_confidence,
        "merchant": merchant,
        "merchant_confidence": merchant_confidence,
    }
    logger.info(f"Prediction result: category={full_category!r}, merchant={merchant!r}")
    return result
