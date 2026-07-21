from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


def get_models() -> dict:
    models = {
        "Majority Class Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    return models