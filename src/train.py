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
TARGETS = ["category", "merchant"]


def load_features():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    X_train_raw, X_test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )

    encoders, y_train, y_test = {}, {}, {}
    for target in TARGETS:
        le = LabelEncoder()
        le.fit(df[target])
        y_train[target] = le.transform(X_train_raw[target])
        y_test[target] = le.transform(X_test_raw[target])
        encoders[target] = le

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(X_train_raw[TEXT_COL])  
    X_test = pipeline.transform(X_test_raw[TEXT_COL])        

    return X_train, X_test, y_train, y_test, pipeline, encoders, X_test_raw


def train_all_models():
    X_train, X_test, y_train, y_test, pipeline, encoders, X_test_raw = load_features()

    results = {target: {} for target in TARGETS}
    skipped = {target: [] for target in TARGETS}
    MODELS_DIR.mkdir(exist_ok=True)

    for target in TARGETS:
        print(f"\n=== Training all models on target: {target} "
              f"({len(encoders[target].classes_)} classes) ===")
        for name, model in get_models().items():
            try:
                start = time.time()
                model.fit(X_train, y_train[target])
                train_time = time.time() - start

                start = time.time()
                y_pred = model.predict(X_test)
                predict_time = time.time() - start
            except Exception as exc:
                print(f"  [{name}] SKIPPED on target '{target}': {exc}")
                skipped[target].append(name)
                continue

            results[target][name] = {
                "model": model,
                "y_pred": y_pred,
                "train_time": train_time,
                "predict_time": predict_time,
            }
            print(f"  [{name}] trained in {train_time:.2f}s, predicted in {predict_time:.3f}s")

    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    for target in TARGETS:
        joblib.dump(encoders[target], MODELS_DIR / f"label_encoder_{target}.joblib")
        for name, r in results[target].items():
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(r["model"], MODELS_DIR / f"{safe_name}_{target}.joblib")

    return results, X_test, y_test, encoders, X_test_raw, skipped


if __name__ == "__main__":
    train_all_models()
