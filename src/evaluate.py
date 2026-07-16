import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report

REPORT_PATH = Path("reports/model_comparison.md")


def build_comparison_table(results, y_test):
    rows = []
    for name, r in results.items():
        y_pred = r["y_pred"]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Train time (s)": r["train_time"],
            "Predict time (s)": r["predict_time"],
        })
    table = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return table


def evaluate_all(results, y_test, label_encoder):
    table = build_comparison_table(results, y_test)

    real_models = table[table["Model"] != "Majority Class Baseline"]
    best_name = real_models.iloc[0]["Model"]
    y_pred = results[best_name]["y_pred"]

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )

    lines = [
        "# Model Comparison Report\n",
        "## Comparison Table\n",
        table.round(4).to_markdown(index=False),
        "",
        f"## Best model: {best_name}\n",
        "### Confusion Matrix\n",
        "```",
        cm_df.to_string(),
        "```",
        "\n### Per-Category Report\n",
        "```",
        report,
        "```",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBest model: {best_name}")
    print(table.round(4).to_string(index=False))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    from train import train_all_models
    results, X_test, y_test, label_encoder, X_test_raw = train_all_models()
    evaluate_all(results, y_test, label_encoder)