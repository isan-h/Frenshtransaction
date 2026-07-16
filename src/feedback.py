"""
feedback.py
-----------
Captures human corrections when a prediction is wrong (or confirms it
was right), so they can be reviewed and folded back into training data
later via incorporate_feedback.py.

IMPORTANT: saving a correction here does NOT retrain any model. It just
safely logs the corrected label to a CSV. Retraining is a separate,
deliberate step -- see incorporate_feedback.py.
"""

import csv
from pathlib import Path
from datetime import datetime

import pandas as pd

FEEDBACK_PATH = Path("data/feedback/corrections.csv")
FEEDBACK_COLUMNS = [
    "timestamp", "description", "model_used",
    "predicted_category", "correct_category",
]


def save_correction(description: str, model_used: str, predicted_category: str, correct_category: str):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not FEEDBACK_PATH.exists()

    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "description": description,
            "model_used": model_used,
            "predicted_category": predicted_category,
            "correct_category": correct_category,
        })


def load_corrections() -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(FEEDBACK_PATH)


def pending_count() -> int:
    return len(load_corrections())
