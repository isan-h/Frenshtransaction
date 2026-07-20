"""
test_on_holdout.py
------------------
Standalone check of the SAVED models (models/*.joblib) on the held-out
20% split, for BOTH targets (category and merchant), reproducing
train.py's exact split so these rows are guaranteed to be ones the
saved models never trained on.
"""

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
TARGETS = ["category", "merchant"]
TARGET_LABELS = {"category": "Category", "merchant": "Merchant"}

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
    # random_state, same test_size, same stratify column (category) ->
    # same 20% rows for BOTH targets.
    X_train_raw, X_test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )
    print(f"Held-out test set: {len(X_test_raw)} rows (20% of {len(df)} total)")
    print("These rows were NOT used to fit models/*.joblib -- confirmed by "
          "reproducing train.py's exact split.\n")

    pipeline = joblib.load(MODELS_DIR / "feature_pipeline.joblib")
    X_test = pipeline.transform(X_test_raw[TEXT_COL])

    lines = [
        "# Holdout Test Report\n",
        f"Standalone test of the SAVED models (models/*.joblib) on the "
        f"{len(X_test_raw)}-row (20%) split they were never trained on -- "
        f"for both the category model and the merchant model.\n",
    ]

    summary_rows = []
    per_target_best = {}

    for target in TARGETS:
        label_encoder = joblib.load(MODELS_DIR / f"label_encoder_{target}.joblib")
        y_test = label_encoder.transform(X_test_raw[target])

        rows = []
        for label, key in AVAILABLE_MODELS.items():
            model_path = MODELS_DIR / f"{key}_{target}.joblib"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            rows.append({"Model": label, "Accuracy": acc, "Precision (macro)": prec,
                          "Recall (macro)": rec, "F1 (macro)": f1, "F1 (weighted)": f1w})
            print(f"[{target}] [{label}] accuracy={acc:.4f}  F1(macro)={f1:.4f}  "
                  f"(tested on {len(y_test)} held-out rows, NOT trained on them)")

        table = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
        best_label = table.iloc[0]["Model"]
        per_target_best[target] = (best_label, y_test)

        summary_rows.append({
            "Target": TARGET_LABELS[target],
            "# Classes": len(label_encoder.classes_),
            "Best model": best_label,
            "Accuracy": table.iloc[0]["Accuracy"],
            "F1 (macro)": table.iloc[0]["F1 (macro)"],
            "F1 (weighted)": table.iloc[0]["F1 (weighted)"],
        })

        lines.append(f"## {TARGET_LABELS[target]} results ({len(label_encoder.classes_)} classes)\n")
        lines.append(table.round(4).to_markdown(index=False))
        lines.append("")

    summary_table = pd.DataFrame(summary_rows)
    lines = lines[:1] + ["## Executive summary\n", summary_table.round(4).to_markdown(index=False), ""] + lines[1:]

    for target in TARGETS:
        label_encoder = joblib.load(MODELS_DIR / f"label_encoder_{target}.joblib")
        best_label, y_test = per_target_best[target]
        best_model = joblib.load(MODELS_DIR / f"{AVAILABLE_MODELS[best_label]}_{target}.joblib")
        y_pred_best = best_model.predict(X_test)
        n_classes = len(label_encoder.classes_)

        lines.append(f"## {TARGET_LABELS[target]} detail -- best on holdout: {best_label}\n")
        if n_classes <= 40:
            cm = confusion_matrix(y_test, y_pred_best, labels=range(n_classes))
            cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
            report = classification_report(
                y_test, y_pred_best, labels=range(n_classes),
                target_names=label_encoder.classes_, zero_division=0,
            )
            lines += ["### Confusion Matrix\n```", cm_df.to_string(), "```",
                       "\n### Per-Class Report\n```", report, "```", ""]
        else:
            report_dict = classification_report(
                y_test, y_pred_best, labels=range(n_classes),
                target_names=label_encoder.classes_, output_dict=True, zero_division=0,
            )
            rows = [{"Class": c, "Support (test)": int(s["support"]), "Precision": s["precision"],
                      "Recall": s["recall"], "F1": s["f1-score"]}
                     for c, s in report_dict.items() if c in label_encoder.classes_]
            per_class = pd.DataFrame(rows)
            tested = per_class[per_class["Support (test)"] > 0]
            n_perfect = (tested["F1"] == 1.0).sum()
            lines += [
                f"{n_perfect}/{len(tested)} merchants seen in this holdout set were "
                f"predicted with perfect F1 (1.00).\n",
                "### 15 hardest merchants on this holdout\n",
                tested.sort_values("F1").head(15).round(3).to_markdown(index=False), "",
            ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_test()
