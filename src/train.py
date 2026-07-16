"""
train.py
--------
Phase 4: train every candidate model using ONLY description text,
predicting the full `category` field (37 classes: "Primary / Detailed",
e.g. "Food & Drink / Groceries"). No primary_category feature this time.
"""

import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_engineering import build_feature_pipeline, TEXT_COL
from models import get_models

DATA_PATH = Path("data/processed/cleaned_transactions.csv")
MODELS_DIR = Path("models")


def load_features():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    y_raw = df["category"]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(X_train_raw[TEXT_COL])  # fit ONLY on train
    X_test = pipeline.transform(X_test_raw[TEXT_COL])        # reuse, never re-fit

    return X_train, X_test, y_train, y_test, pipeline, label_encoder, X_test_raw


def train_all_models():
    X_train, X_test, y_train, y_test, pipeline, label_encoder, X_test_raw = load_features()
    models = get_models()

    results = {}
    MODELS_DIR.mkdir(exist_ok=True)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "train_time": train_time,
            "predict_time": predict_time,
        }
        print(f"[{name}] trained in {train_time:.2f}s, predicted in {predict_time:.3f}s")

    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    for name, r in results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(r["model"], MODELS_DIR / f"{safe_name}.joblib")

    return results, X_test, y_test, label_encoder, X_test_raw


if __name__ == "__main__":
    train_all_models()