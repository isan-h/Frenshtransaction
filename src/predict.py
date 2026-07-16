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
        if (MODELS_DIR / f"{fname}.joblib").exists()
    }


def get_categories() -> list:
    """All full-category labels the model was trained on, sorted
    alphabetically. Loaded from the saved label encoder -- not
    hardcoded -- so this stays correct if the taxonomy changes."""
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return sorted(label_encoder.classes_.tolist())


def load_artifacts(model_key: str):
    pipeline = joblib.load(MODELS_DIR / "feature_pipeline.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    model = joblib.load(MODELS_DIR / f"{model_key}.joblib")
    return pipeline, label_encoder, model


def predict_category(description: str, model_key: str = "linear_svm"):
    """
    Returns (full_category, primary_category, detailed_category, confidence, all_probs)
    """
    pipeline, label_encoder, model = load_artifacts(model_key)

    cleaned = clean_text(description)
    X = pipeline.transform([cleaned])

    pred_encoded = model.predict(X)[0]
    full_category = label_encoder.inverse_transform([pred_encoded])[0]

    if " / " in full_category:
        primary_category, detailed_category = full_category.split(" / ", 1)
    else:
        primary_category, detailed_category = full_category, full_category

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba.max())
        all_probs = dict(zip(label_encoder.classes_, [float(p) for p in proba]))
    else:
        confidence, all_probs = None, None

    return full_category, primary_category, detailed_category, confidence, all_probs


if __name__ == "__main__":
    full, primary, detailed, confidence, all_probs = predict_category(
        "CB LIDL 4658 CERGY", model_key="linear_svm"
    )
    print("Full category:", full)
    print("Primary:", primary)
    print("Detailed:", detailed)
    print("Confidence:", confidence)