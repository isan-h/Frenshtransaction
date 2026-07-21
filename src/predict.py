import joblib
import pandas as pd
from pathlib import Path

from preprocessing import clean_text

MODELS_DIR = Path("models")

AVAILABLE_MODELS = {
    "Linear SVM": "linear_svm",
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def list_available_models() -> dict:
    return {
        label: fname for label, fname in AVAILABLE_MODELS.items()
        if (MODELS_DIR / f"{fname}_category.joblib").exists()
        and (MODELS_DIR / f"{fname}_merchant.joblib").exists()
    }


def get_categories() -> list:
    label_encoder = joblib.load(MODELS_DIR / "label_encoder_category.joblib")
    return sorted(label_encoder.classes_.tolist())


def get_merchants() -> list:
    """All merchant labels the model was trained on, sorted alphabetically."""
    label_encoder = joblib.load(MODELS_DIR / "label_encoder_merchant.joblib")
    return sorted(label_encoder.classes_.tolist())


def load_artifacts(model_key: str):
    pipeline = joblib.load(MODELS_DIR / "feature_pipeline.joblib")
    cat_encoder = joblib.load(MODELS_DIR / "label_encoder_category.joblib")
    merch_encoder = joblib.load(MODELS_DIR / "label_encoder_merchant.joblib")
    cat_model = joblib.load(MODELS_DIR / f"{model_key}_category.joblib")
    merch_model = joblib.load(MODELS_DIR / f"{model_key}_merchant.joblib")
    return pipeline, cat_encoder, merch_encoder, cat_model, merch_model


def predict_transaction(description: str, model_key: str = "linear_svm") -> dict:
    pipeline, cat_encoder, merch_encoder, cat_model, merch_model = load_artifacts(model_key)

    cleaned = clean_text(description)
    X = pipeline.transform([cleaned])

    # --- category head ---
    cat_pred_encoded = cat_model.predict(X)[0]
    full_category = cat_encoder.inverse_transform([cat_pred_encoded])[0]
    if " / " in full_category:
        primary_category, detailed_category = full_category.split(" / ", 1)
    else:
        primary_category, detailed_category = full_category, full_category
    category_confidence = None
    if hasattr(cat_model, "predict_proba"):
        category_confidence = float(cat_model.predict_proba(X)[0].max())

    # --- merchant head ---
    merch_pred_encoded = merch_model.predict(X)[0]
    merchant = merch_encoder.inverse_transform([merch_pred_encoded])[0]
    merchant_confidence = None
    if hasattr(merch_model, "predict_proba"):
        merchant_confidence = float(merch_model.predict_proba(X)[0].max())

    return {
        "full_category": full_category,
        "primary_category": primary_category,
        "detailed_category": detailed_category,
        "category_confidence": category_confidence,
        "merchant": merchant,
        "merchant_confidence": merchant_confidence,
    }


# Kept for anything still calling the old, category-only name.
def predict_category(description: str, model_key: str = "linear_svm"):
    pipeline, cat_encoder, merch_encoder, cat_model, merch_model = load_artifacts(model_key)

    cleaned = clean_text(description)
    X = pipeline.transform([cleaned])

    pred_encoded = cat_model.predict(X)[0]
    full_category = cat_encoder.inverse_transform([pred_encoded])[0]

    if " / " in full_category:
        primary_category, detailed_category = full_category.split(" / ", 1)
    else:
        primary_category, detailed_category = full_category, full_category

    if hasattr(cat_model, "predict_proba"):
        proba = cat_model.predict_proba(X)[0]
        confidence = float(proba.max())
        all_probs = dict(zip(cat_encoder.classes_, [float(p) for p in proba]))
    else:
        confidence, all_probs = None, None

    return full_category, primary_category, detailed_category, confidence, all_probs


if __name__ == "__main__":
    result = predict_transaction("CB ZODIO 7615 REIMS", model_key="linear_svm")
    print("Full category:", result["full_category"])
    print("Primary:", result["primary_category"])
    print("Detailed:", result["detailed_category"])
    print("Category confidence:", result["category_confidence"])
    print("Merchant:", result["merchant"])
    print("Merchant confidence:", result["merchant_confidence"])
