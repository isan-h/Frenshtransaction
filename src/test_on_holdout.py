import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from feature_engineering import TEXT_COL

DATA_PATH = Path("data/processed/cleaned_transactions.csv")
MODELS_DIR = Path("models")
REPORT_PATH = Path("reports/holdout_test_report.md")

AVAILABLE_MODELS = {
    "Linear SVM": "linear_svm",
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def run_test():
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    # EXACT same split call as train.py's load_features() -- same
    # random_state, same test_size, same stratify column -> same 20%.
    y_raw = df["category"]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"Held-out test set: {len(X_test_raw)} rows (20% of {len(df)} total)")
    print("These rows were NOT used to fit models/*.joblib -- confirmed by "
          "reproducing train.py's exact split.\n")

    pipeline = joblib.load(MODELS_DIR / "feature_pipeline.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    X_test = pipeline.transform(X_test_raw[TEXT_COL])
    y_test = label_encoder.transform(y_test_raw)

    lines = [
        "# Holdout Test Report\n",
        f"Standalone test of the SAVED models (models/*.joblib) on the "
        f"{len(X_test_raw)}-row (20%) split they were never trained on.\n",
        "## Results\n",
    ]

    rows = []
    for label, key in AVAILABLE_MODELS.items():
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        rows.append({"Model": label, "Accuracy": acc, "Precision (macro)": prec,
                      "Recall (macro)": rec, "F1 (macro)": f1})
        print(f"[{label}] accuracy={acc:.4f}  F1(macro)={f1:.4f}  "
              f"(tested on {len(y_test)} held-out rows, NOT trained on them)")

    table = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    lines.append(table.round(4).to_markdown(index=False))

    # Confusion matrix + classification report for the best model
    best_label = table.iloc[0]["Model"]
    best_model = joblib.load(MODELS_DIR / f"{AVAILABLE_MODELS[best_label]}.joblib")
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    report = classification_report(y_test, y_pred_best, target_names=label_encoder.classes_, zero_division=0)

    lines.append(f"\n## Best on holdout: {best_label}\n")
    lines.append("### Confusion Matrix\n```")
    lines.append(cm_df.to_string())
    lines.append("```\n### Per-Category Report\n```")
    lines.append(report)
    lines.append("```")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_test()
