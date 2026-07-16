import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from feature_engineering import build_feature_pipeline, TEXT_COL
from models import get_models

DATA_PATH = "data/processed/cleaned_transactions.csv"


def run_cross_validation(n_splits: int = 5):
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    y = LabelEncoder().fit_transform(df["category"])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([
            ("features", build_feature_pipeline()),
            ("clf", model),
        ])
        scores = cross_val_score(pipe, df[TEXT_COL], y, cv=cv, scoring="f1_macro")
        rows.append({"Model": name, "F1 macro (mean)": scores.mean(), "F1 macro (std)": scores.std()})
        print(f"[{name}] F1 macro = {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"(folds: {[round(float(s), 3) for s in scores]})")

    table = pd.DataFrame(rows).sort_values("F1 macro (mean)", ascending=False).reset_index(drop=True)
    return table


if __name__ == "__main__":
    table = run_cross_validation()
    print("\n" + table.to_string(index=False))