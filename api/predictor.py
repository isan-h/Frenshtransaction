"""
predictor.py
------------
The actual prediction logic. Deliberately has ZERO knowledge of
FastAPI, HTTP, or JSON -- it's a plain function: a cleaned description
in, a prediction out. This separation is what makes the logic testable
without spinning up a server, and keeps "business logic" and "web
framework" as two genuinely separate concerns.

Reuses the EXACT SAME clean_text() function from src/preprocessing.py
that the model was trained with -- using anything different here would
silently skew every prediction, since the model expects text cleaned
the same way its training data was.
"""

import logging
import sys
from pathlib import Path

from api.model_loader import ModelBundle

# src/ isn't a package -- add it to the path so we can import the EXACT
# same clean_text() used during training, not a reimplementation that
# could quietly drift out of sync with it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import clean_text  # noqa: E402

logger = logging.getLogger(__name__)


def predict(description: str, bundle: ModelBundle) -> dict:
    """
    Runs one description through the shared TF-IDF features, then both
    trained heads (category + merchant).

    Takes the ModelBundle as an explicit argument rather than reaching
    for a global variable -- this is what makes the function easy to
    call directly in a test with a fake/mock bundle, without needing
    the real models loaded.
    """
    cleaned = clean_text(description)
    logger.info(f"Predicting for description: {description!r} (cleaned: {cleaned!r})")

    X = bundle.feature_pipeline.transform([cleaned])

    # --- category head ---
    category_pred_encoded = bundle.category_model.predict(X)[0]
    full_category = bundle.category_encoder.inverse_transform([category_pred_encoded])[0]
    if " / " in full_category:
        primary_category, detailed_category = full_category.split(" / ", 1)
    else:
        primary_category, detailed_category = full_category, full_category

    category_confidence = None
    if hasattr(bundle.category_model, "predict_proba"):
        category_confidence = float(bundle.category_model.predict_proba(X)[0].max())

    # --- merchant head ---
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
