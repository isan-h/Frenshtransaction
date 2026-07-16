"""
preprocessing.py
-----------------
Cleaning for the REAL Effyis French transaction data.

IMPORTANT DIFFERENCE from the mock-data project: there, the "PAYMENT"
prefix was uninformative noise (appeared at the same rate across every
category) and was stripped like a stopword. Here, EDA showed the
opposite: prefix words are highly informative.

  - RETRAIT            -> ALWAYS "Cash"        (100% of 74 rows)
  - FRAIS/COMMISSION/
    AGIOS/COTISATION    -> ALWAYS "Fees"        (100% of 40 rows)
  - VIR                 -> ONLY "Transfers" or "Income" (never anything else)
  - PRLV                -> ONLY "Utilities"/"Subscriptions"/"Insurance"
  - CB / PAIEMENT / CARTE -> spread evenly across the 6 merchant-driven
                             categories (Shopping, Restaurants, Transport,
                             Groceries, Entertainment, Health) -- these
                             three ARE uninformative for distinguishing
                             among those 6, but still informative for
                             separating "this is a card purchase" from
                             the bank-operation categories above.

So: DO NOT strip these prefixes. Only remove things that are genuinely
noise -- punctuation/separators and numeric reference codes.
"""

import re
import pandas as pd


def remove_special_characters(text: str) -> str:
    """Replace punctuation/slashes with a space (e.g. '28/02' -> '28 02'),
    so digit-removal afterward doesn't glue neighboring words together."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_numbers(text: str) -> str:
    """Strip digits -- these are transaction reference codes (e.g. the
    '4658' in 'CB LIDL 4658 CERGY') or embedded dates (e.g. '28 02' from
    'CARTE 28/02 ...'). Neither carries category signal; both would just
    bloat the TF-IDF vocabulary with near-random one-off tokens."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\d+", " ", text)


def normalize_spaces(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Master cleaning function. Deliberately NO stopword-removal step here
    (unlike the mock-data project) -- EDA confirmed the prefix words
    (RETRAIT, FRAIS, VIR, PRLV, etc.) are strong category signal, not noise.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_special_characters(text)
    text = remove_numbers(text)
    text = normalize_spaces(text)
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the raw transactions dataframe."""
    df = df.copy()
    df["description_clean"] = df["description"].apply(clean_text)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # currency is constant (100% "EUR") -- zero predictive value, drop it
    constant_cols = [c for c in ["currency"] if c in df.columns and df[c].nunique() == 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    target_cols = [c for c in ["category", "detailed_category", "primary_category"] if c in df.columns]
    if target_cols:
        before = len(df)
        is_missing = pd.Series(False, index=df.index)
        for c in target_cols:
            is_missing = is_missing | df[c].isnull() | (df[c].astype(str).str.strip() == "")
        bad_rows = df[is_missing]
        if len(bad_rows) > 0:
            print(f"WARNING: dropping {len(bad_rows)} row(s) with a missing/empty category label:")
            print(bad_rows[["transaction_id", "description"] + target_cols].to_string())
        df = df[~is_missing]
        after = len(df)
        if before != after:
            print(f"Rows: {before} -> {after} after dropping incomplete labels.")

    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/transactions_fr_balanced.csv")
    cleaned = clean_dataframe(raw)
    out_path = "data/processed/cleaned_transactions.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Cleaned {len(cleaned)} rows -> saved to {out_path}")
    print(cleaned[["description", "description_clean"]].sample(10, random_state=1))