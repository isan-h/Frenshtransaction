"""
generate_predictions.py
------------------------
Produces the honest, out-of-fold prediction comparison for BOTH
targets (category and merchant) in one file, using the real unlabeled
copy of the dataset. See README for why cross_val_predict is required
here instead of a simple train/predict pass.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from feature_engineering import build_feature_pipeline, TEXT_COL
from preprocessing import clean_text
from models import get_models

LABELED_PATH = Path("data/raw/transactions_fr_balanced.csv")
UNLABELED_PATH = Path("data/raw/transactions_fr_balanced - Copie.csv")
OUT_PATH = Path("data/processed/predictions_comparison.csv")


def split_category(cat: str):
    if " / " in cat:
        primary, detailed = cat.split(" / ", 1)
        return primary, detailed
    return cat, cat


def generate():
    labeled = pd.read_csv(LABELED_PATH)
    labeled[TEXT_COL] = labeled["description"].apply(clean_text)

    oof_predictions = pd.DataFrame({"transaction_id": labeled["transaction_id"]})

    cv_by_target = {
        # category is balanced (60/class) -> stratified 5-fold is safe
        "category": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        # merchant has classes with as few as 1 example -> can't stratify
        "merchant": KFold(n_splits=5, shuffle=True, random_state=42),
    }

    for target, cv in cv_by_target.items():
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labeled[target])

        for name, model in get_models().items():
            if name == "Majority Class Baseline":
                continue
            pipe = Pipeline([
                ("features", build_feature_pipeline()),
                ("clf", model),
            ])
            print(f"Generating honest out-of-fold {target} predictions for {name}...")
            try:
                y_pred_encoded = cross_val_predict(pipe, labeled[TEXT_COL], y, cv=cv)
            except ValueError as exc:
                # e.g. XGBoost requires every class 0..N-1 present in every
                # training fold -- with 292 merchant classes (some with only
                # 1-2 rows) that can fail on some folds. Skip this model for
                # this target only rather than crash the whole run.
                print(f"  SKIPPED {name} on '{target}': {exc}")
                continue
            y_pred = label_encoder.inverse_transform(y_pred_encoded)

            safe_name = name.lower().replace(" ", "_")
            if target == "category":
                oof_predictions[f"predicted_category_{safe_name}"] = y_pred
                split = [split_category(c) for c in y_pred]
                oof_predictions[f"predicted_primary_category_{safe_name}"] = [s[0] for s in split]
                oof_predictions[f"predicted_detailed_category_{safe_name}"] = [s[1] for s in split]
            else:
                oof_predictions[f"predicted_merchant_{safe_name}"] = y_pred

    unlabeled = pd.read_csv(UNLABELED_PATH)
    out = unlabeled[["transaction_id", "date", "description", "amount", "type"]].copy()
    out = out.merge(oof_predictions, on="transaction_id", how="left")
    truth = labeled[["transaction_id", "merchant", "category"]].rename(
        columns={"merchant": "actual_merchant", "category": "actual_category"}
    )
    out = out.merge(truth, on="transaction_id", how="left")

    for name in get_models():
        if name == "Majority Class Baseline":
            continue
        safe_name = name.lower().replace(" ", "_")
        if f"predicted_category_{safe_name}" in out.columns:
            out[f"match_category_{safe_name}"] = (
                out[f"predicted_category_{safe_name}"] == out["actual_category"]
            ).astype(int)
        if f"predicted_merchant_{safe_name}" in out.columns:
            out[f"match_merchant_{safe_name}"] = (
                out[f"predicted_merchant_{safe_name}"] == out["actual_merchant"]
            ).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved to {OUT_PATH}")
    print("\nHonest (out-of-fold) match rate per model:")
    for target in ["category", "merchant"]:
        print(f"  -- {target} --")
        for col in out.columns:
            if col.startswith(f"match_{target}_"):
                print(f"    {col.replace(f'match_{target}_', '')}: {out[col].mean():.4f}")


if __name__ == "__main__":
    generate()
