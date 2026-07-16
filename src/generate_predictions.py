import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labeled["category"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Honest out-of-fold predictions, keyed by transaction_id
    oof_predictions = pd.DataFrame({"transaction_id": labeled["transaction_id"]})

    for name, model in get_models().items():
        if name == "Majority Class Baseline":
            continue
        pipe = Pipeline([
            ("features", build_feature_pipeline()),
            ("clf", model),
        ])
        print(f"Generating honest out-of-fold predictions for {name}...")
        y_pred_encoded = cross_val_predict(pipe, labeled[TEXT_COL], y, cv=cv)
        y_pred_category = label_encoder.inverse_transform(y_pred_encoded)

        safe_name = name.lower().replace(" ", "_")
        oof_predictions[f"predicted_category_{safe_name}"] = y_pred_category
        split = [split_category(c) for c in y_pred_category]
        oof_predictions[f"predicted_primary_category_{safe_name}"] = [s[0] for s in split]
        oof_predictions[f"predicted_detailed_category_{safe_name}"] = [s[1] for s in split]

    # Build the deliverable starting from the REAL unlabeled file
    unlabeled = pd.read_csv(UNLABELED_PATH)
    out = unlabeled[["transaction_id", "date", "description", "amount", "type"]].copy()
    out = out.merge(oof_predictions, on="transaction_id", how="left")

    # Attach ground truth (from the labeled file) so match_<model> can be
    # computed -- this is for YOUR verification, not something the model saw.
    truth = labeled[["transaction_id", "merchant", "category"]].rename(
        columns={"merchant": "actual_merchant", "category": "actual_category"}
    )
    out = out.merge(truth, on="transaction_id", how="left")

    for name in get_models():
        if name == "Majority Class Baseline":
            continue
        safe_name = name.lower().replace(" ", "_")
        out[f"match_{safe_name}"] = (
            out[f"predicted_category_{safe_name}"] == out["actual_category"]
        ).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved to {OUT_PATH}")
    print("\nAccuracy per model (honest, out-of-fold):")
    for col in out.columns:
        if col.startswith("match_"):
            print(f"  {col.replace('match_', '')}: {out[col].mean():.4f}")


if __name__ == "__main__":
    generate()