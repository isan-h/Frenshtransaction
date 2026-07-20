"""
cross_validate.py
------------------
Phase 5b: 5-fold CV robustness check for BOTH targets, run in the same
pass. `category` (37 classes, 60 rows each) is stratified-K-fold as
before. `merchant` (292 classes, a handful with only 1-2 rows) CANNOT
be stratified -- sklearn requires every class to have >= n_splits
members -- so it uses a plain shuffled KFold instead. This is a real
difference in evaluation rigor between the two targets and is called
out explicitly in the printed output/table, not hidden.
"""

import warnings
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from feature_engineering import build_feature_pipeline, TEXT_COL
from models import get_models

DATA_PATH = "data/processed/cleaned_transactions.csv"
REPORT_PATH = Path("reports/cross_validation_report.md")
TARGET_LABELS = {"category": "Category", "merchant": "Merchant"}


def run_cross_validation(n_splits: int = 5):
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    cv_by_target = {
        "category": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        "merchant": KFold(n_splits=n_splits, shuffle=True, random_state=42),
    }

    rows = []
    for target, cv in cv_by_target.items():
        y = LabelEncoder().fit_transform(df[target])
        print(f"\n=== 5-fold CV on target: {target} "
              f"({'stratified' if target == 'category' else 'plain KFold -- too many singleton classes to stratify'}) ===")
        for name, model in get_models().items():
            pipe = Pipeline([
                ("features", build_feature_pipeline()),
                ("clf", model),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    pipe, df[TEXT_COL], y, cv=cv, scoring="f1_macro", error_score=float("nan")
                )
            n_failed = int(pd.isna(scores).sum())
            note = f"{n_failed}/{n_splits} folds failed" if n_failed else ""
            rows.append({
                "Target": TARGET_LABELS[target], "Model": name,
                "F1 macro (mean)": pd.Series(scores).mean(skipna=True),
                "F1 macro (std)": pd.Series(scores).std(skipna=True),
                "Note": note,
            })
            print(f"  [{name}] F1 macro = {scores.mean():.4f} +/- {scores.std():.4f}  "
                  f"(folds: {[round(float(s), 3) if pd.notna(s) else 'FAILED' for s in scores]})")

    table = pd.DataFrame(rows).sort_values(
        ["Target", "F1 macro (mean)"], ascending=[True, False]
    ).reset_index(drop=True)

    lines = [
        "# 5-Fold Cross-Validation Report\n",
        "Robustness check on top of the single 80/20 holdout split: same "
        "models, same features, but scored across 5 different folds so a "
        "lucky/unlucky single split can't skew the numbers. `category` "
        "uses stratified folds (safe: 60 rows/class). `merchant` uses "
        "plain (non-stratified) folds because dozens of merchants have "
        "too few examples to stratify on -- some rare merchants may be "
        "entirely absent from a training fold, which is why a few models "
        "(esp. XGBoost) occasionally fail a fold on that target; failed "
        "folds are excluded from the mean/std and flagged in the Note column.\n",
        table.round(4).to_markdown(index=False),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    return table


if __name__ == "__main__":
    table = run_cross_validation()
    print("\n" + table.to_string(index=False))
