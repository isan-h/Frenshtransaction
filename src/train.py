"""
train.py
--------
Phase 4: train every candidate model on TWO targets extracted from the
SAME description text -- `category` (37 classes, "Primary / Detailed")
and `merchant` (292 classes, the payee name, e.g. "Zodio"). Both targets
share the exact same cleaned text, the same train/test row split, and
the same TF-IDF features -- only the label column differs. This is
deliberate: category and merchant are trained together in one run, not
as two disconnected projects, so every result is directly comparable
model-for-model, task-for-task.
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

# The two things we predict from description text. Order matters only
# for print order -- both are trained/evaluated every run.
TARGETS = ["category", "merchant"]


def load_features():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    # ONE split for both targets, stratified on `category` (perfectly
    # balanced, 60 rows/class -- safe to stratify on). `merchant` has a
    # handful of merchants with only 1 example, which train_test_split
    # can't stratify on directly -- so instead of splitting twice (which
    # would put category and merchant on different rows and make results
    # incomparable), merchant just rides the same category-stratified
    # split. Net effect: a few rare merchants end up test-only and the
    # model will never have learned them -- this is a real, honestly
    # reported limitation, not hidden (see the report's merchant section).
    X_train_raw, X_test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )

    encoders, y_train, y_test = {}, {}, {}
    for target in TARGETS:
        le = LabelEncoder()
        # fit on the FULL column (not just train) so a merchant/category
        # that only appears in the test split still has a valid encoding
        # instead of crashing .transform() on an unseen label
        le.fit(df[target])
        y_train[target] = le.transform(X_train_raw[target])
        y_test[target] = le.transform(X_test_raw[target])
        encoders[target] = le

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(X_train_raw[TEXT_COL])  # fit ONLY on train
    X_test = pipeline.transform(X_test_raw[TEXT_COL])        # reuse, never re-fit

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
                # Some models (e.g. XGBoost) require every class index
                # 0..N-1 to appear in the TRAINING split. With 292 merchant
                # classes and a handful of rare merchants, a few classes
                # only landed in the test split -- that model is skipped
                # for THIS target only, and it's reported, not hidden.
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

    # One shared feature pipeline, but one label encoder + one set of
    # model files per target -- e.g. linear_svm_category.joblib and
    # linear_svm_merchant.joblib live side by side.
    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    for target in TARGETS:
        joblib.dump(encoders[target], MODELS_DIR / f"label_encoder_{target}.joblib")
        for name, r in results[target].items():
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(r["model"], MODELS_DIR / f"{safe_name}_{target}.joblib")

    return results, X_test, y_test, encoders, X_test_raw, skipped


if __name__ == "__main__":
    train_all_models()
