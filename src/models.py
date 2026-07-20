from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


def get_models() -> dict:
    models = {
        "Majority Class Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=5000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)
    except ImportError:
        print("[models.py] xgboost not installed, skipping. Install with: pip install xgboost")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
    except ImportError:
        print("[models.py] lightgbm not installed, skipping. Install with: pip install lightgbm")

    return models