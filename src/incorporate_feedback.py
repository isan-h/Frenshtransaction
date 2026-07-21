import pandas as pd
from pathlib import Path
from datetime import datetime

from preprocessing import clean_text
from feedback import load_corrections, FEEDBACK_PATH

CLEANED_PATH = Path("data/processed/cleaned_transactions.csv")
ARCHIVE_PATH = Path("data/feedback/incorporated_log.csv")


def split_category(cat: str):
    if " / " in cat:
        primary, detailed = cat.split(" / ", 1)
        return primary, detailed
    return cat, cat


def incorporate():
    corrections = load_corrections()
    if corrections.empty:
        print("No pending corrections logged yet -- nothing to incorporate.")
        return

    cleaned = pd.read_csv(CLEANED_PATH)

    splits = [split_category(c) for c in corrections["correct_category"]]
    new_rows = pd.DataFrame({
        "transaction_id": "feedback",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": "00:00:00",
        "merchant": "", 
        "description": corrections["description"],
        "description_clean": corrections["description"].apply(clean_text),
        "amount": 0.0,
        "type": "UNKNOWN",
        "balance_after": 0.0,
        "primary_category": [s[0] for s in splits],
        "detailed_category": [s[1] for s in splits],
        "category": corrections["correct_category"],
    })

    combined = pd.concat([cleaned, new_rows], ignore_index=True)
    combined.to_csv(CLEANED_PATH, index=False)
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not ARCHIVE_PATH.exists()
    corrections.to_csv(ARCHIVE_PATH, mode="a", header=header_needed, index=False)
    FEEDBACK_PATH.unlink()

    print(f"Folded in {len(new_rows)} corrected transaction(s).")
    print(f"cleaned_transactions.csv: {len(cleaned)} -> {len(combined)} rows.")
    print("Now run `python src/evaluate.py` to retrain on the updated data.")


if __name__ == "__main__":
    incorporate()
