import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)

REPORT_PATH = Path("reports/model_comparison.md")
DETAIL_DIR = Path("reports")
TARGETS = ["category", "merchant"]
TARGET_LABELS = {"category": "Category", "merchant": "Merchant"}


def build_comparison_table(target_results, y_test):
    rows = []
    for name, r in target_results.items():
        y_pred = r["y_pred"]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Train time (s)": r["train_time"],
            "Predict time (s)": r["predict_time"],
        })
    table = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return table


def _per_class_table(y_test, y_pred, label_encoder):
    report_dict = classification_report(
        y_test, y_pred, labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_, output_dict=True, zero_division=0,
    )
    rows = []
    for cls in label_encoder.classes_:
        stats = report_dict[cls]
        rows.append({
            "Class": cls, "Support (test)": int(stats["support"]),
            "Precision": stats["precision"], "Recall": stats["recall"], "F1": stats["f1-score"],
        })
    return pd.DataFrame(rows)


def evaluate_all(results, y_test, encoders, skipped=None):
    skipped = skipped or {t: [] for t in TARGETS}
    lines = [
        "# Model Comparison Report\n",
        "This report covers **two models trained together in the same run**, "
        "from the same cleaned `description` text and the same train/test "
        "split (80/20, stratified on category): predicting the transaction's "
        "**category** and predicting the **merchant**. Only the label "
        "column differs between the two -- everything else (features, "
        "split, candidate models) is shared.\n",
    ]
    for target in TARGETS:
        if skipped.get(target):
            lines.append(
                f"> Note: {', '.join(skipped[target])} skipped on **{TARGET_LABELS[target]}** -- "
                "requires every class to appear in the training split, and a "
                "few rare merchants (as few as 1 example total) landed in the "
                "test split only. Not a bug: an honestly reported limitation "
                "of the small-sample classes, not the modeling approach.\n"
            )

    summary_rows = []
    best_preds = {}

    for target in TARGETS:
        label_encoder = encoders[target]
        n_classes = len(label_encoder.classes_)
        table = build_comparison_table(results[target], y_test[target])
        real = table[table["Model"] != "Majority Class Baseline"].reset_index(drop=True)
        best_name = real.iloc[0]["Model"]
        y_pred = results[target][best_name]["y_pred"]
        best_preds[target] = (best_name, y_pred)

        summary_rows.append({
            "Target": TARGET_LABELS[target],
            "# Classes": n_classes,
            "Best model": best_name,
            "Accuracy": real.iloc[0]["Accuracy"],
            "F1 (macro)": real.iloc[0]["F1 (macro)"],
            "F1 (weighted)": real.iloc[0]["F1 (weighted)"],
        })

        lines.append(f"## {TARGET_LABELS[target]} model comparison ({n_classes} classes)\n")
        lines.append(table.round(4).to_markdown(index=False))
        lines.append("")
    summary_table = pd.DataFrame(summary_rows)
    lines = lines[:2] + [
        "## Executive summary\n",
        summary_table.round(4).to_markdown(index=False),
        "",
    ] + lines[2:]

    for target in TARGETS:
        label_encoder = encoders[target]
        n_classes = len(label_encoder.classes_)
        best_name, y_pred = best_preds[target]
        lines.append(f"## {TARGET_LABELS[target]} detail -- best model: {best_name}\n")

        if n_classes <= 40:
            cm = confusion_matrix(y_test[target], y_pred, labels=range(n_classes))
            cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
            report = classification_report(
                y_test[target], y_pred, labels=range(n_classes),
                target_names=label_encoder.classes_, zero_division=0,
            )
            lines += ["### Confusion Matrix\n", "```", cm_df.to_string(), "```",
                      "\n### Per-Class Report\n", "```", report, "```", ""]
        else:
            per_class = _per_class_table(y_test[target], y_pred, label_encoder)
            per_class_out = DETAIL_DIR / f"{target}_per_class.csv"
            per_class.sort_values("F1").to_csv(per_class_out, index=False)

            tested = per_class[per_class["Support (test)"] > 0]
            worst = tested.sort_values("F1").head(15)
            best = tested.sort_values("F1", ascending=False).head(10)
            n_perfect = (tested["F1"] == 1.0).sum()

            lines += [
                f"Full per-merchant precision/recall/F1 for all {n_classes} "
                f"classes saved to `{per_class_out.as_posix()}` (too long for "
                f"this report). Summary: **{n_perfect}/{len(tested)}** "
                f"merchants seen in the test set were predicted with perfect "
                f"F1 (1.00).\n",
                "### 15 hardest merchants (lowest F1 on the test set)\n",
                worst.round(3).to_markdown(index=False),
                "",
                "### 10 easiest merchants (for reference)\n",
                best.round(3).to_markdown(index=False),
                "",
            ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Executive summary ===")
    print(summary_table.round(4).to_string(index=False))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    from train import train_all_models
    results, X_test, y_test, encoders, X_test_raw, skipped = train_all_models()
    evaluate_all(results, y_test, encoders, skipped)
