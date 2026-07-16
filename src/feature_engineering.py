"""
feature_engineering.py
------------------------
Text-ONLY feature pipeline: description_clean -> TF-IDF. No
primary_category feature this time, per instruction to predict using
only the description. Target is now the full `category` field
(37 classes, e.g. "Food & Drink / Groceries") -- predicting this one
field gives us both primary_category and detailed_category for free by
splitting the string on " / ".
"""

from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COL = "description_clean"


def build_feature_pipeline():
    return TfidfVectorizer(max_features=1000, ngram_range=(1, 2))


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/cleaned_transactions.csv")
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df[TEXT_COL].fillna(""))
    print("Feature matrix shape:", X.shape)