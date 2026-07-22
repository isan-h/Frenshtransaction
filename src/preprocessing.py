import re
import pandas as pd


def remove_special_characters(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_numbers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\d+", " ", text)


def normalize_spaces(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
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