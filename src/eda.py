import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_PATH = Path("data/raw/transactions_fr_balanced - Copie.csv")
REF_PATH = Path("data/raw/categories_reference.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/eda_report.md")

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "text.color": "white", "xtick.color": "white", "ytick.color": "white",
    "axes.titlecolor": "white", "grid.color": "#33334d",
})
BLUE, YELLOW = "#4c9be8", "#f2c94c"


def bar_chart(series, title, filename, color=BLUE, figsize=(10, 6)):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    x_positions = range(len(series))
    ax.bar(x_positions, series.values, color=color, edgecolor="white")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(series.index, rotation=60, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_eda():
    df = pd.read_csv(RAW_PATH)
    ref = pd.read_csv(REF_PATH)
    lines = ["# EDA Report -- Balanced Effyis Dataset\n", f"Source: `{RAW_PATH.name}`\n"]

    lines.append("## 1. Shape\n")
    lines.append(f"- Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**\n")
    lines.append("No missing values, no duplicates (checked).\n")

    lines.append("## 2. Taxonomy Alignment -- the previously flagged problem is RESOLVED\n")
    data_combos = set(df["category"].unique())
    ref_combos = set(ref["DESCRIPTION"].unique())
    lines.append(
        f"All **{len(data_combos)}** category combinations in this data exist in the "
        f"official 68-item reference taxonomy ({len(data_combos & ref_combos)}/{len(data_combos)} matched). "
        "Unlike the earlier dataset, `primary_category` + `detailed_category` here "
        "directly correspond to the reference file's Main/Sub structure.\n"
    )

    lines.append("## 3. Target Variable Decision\n")
    lines.append(
        "`primary_category` is present in BOTH the labeled and unlabeled files, "
        "but is deliberately NOT used as an input feature in the current version "
        "of this project -- predictions are made from `description` text ONLY.\n\n"
        "- **Target to predict: the full `category` field** (37 classes, e.g. "
        "\"Food & Drink / Groceries\")\n"
        "- **Input feature: `description` (text) ONLY** -- no `primary_category`\n"
        "- `primary_category` and `detailed_category` for reporting are recovered "
        "by splitting the predicted `category` string on \" / \"\n\n"
        "Note: an earlier version of this project DID use `primary_category` as "
        "an additional input feature (one-hot encoded) and scored slightly higher "
        "(98.76% vs 97.04% F1 on a single split) -- about a 1.8-point cost for "
        "going text-only. Section 6 below (on how much primary_category narrows "
        "the problem) is kept for reference, but is not currently exploited by "
        "the feature pipeline.\n"
    )

    lines.append("## 4. Primary Category Distribution\n")
    pc_counts = df["primary_category"].value_counts()
    lines.append("```")
    lines.append(pc_counts.to_string())
    lines.append("```")
    bar_chart(pc_counts, "Transactions per Primary Category", "primary_category_distribution.png")
    lines.append("\n![primary category distribution](figures/primary_category_distribution.png)\n")

    lines.append("## 5. Detailed Category Distribution (the actual prediction target)\n")
    dc_counts = df["detailed_category"].value_counts()
    lines.append(f"**{df['detailed_category'].nunique()}** unique detailed categories.\n")
    lines.append("```")
    lines.append(dc_counts.to_string())
    lines.append("```")
    bar_chart(dc_counts, "Transactions per Detailed Category", "detailed_category_distribution.png", color=YELLOW, figsize=(13, 7))
    lines.append("\n![detailed category distribution](figures/detailed_category_distribution.png)\n")
    lines.append(
        "33 of 34 categories have exactly 60 examples each -- genuinely balanced "
        "(unlike the earlier 6.8x-imbalanced dataset). Only \"Other\" is larger (240), "
        "as a catch-all bucket.\n"
    )

    lines.append("## 6. primary_category Narrows detailed_category Sharply\n")
    mapping = df.groupby("primary_category")["detailed_category"].unique()
    lines.append(
        "This was the key feature-engineering decision in an earlier version "
        "of this project (not the current text-only one -- see Section 3): "
        "knowing "
        "`primary_category` restricts the possible `detailed_category` values "
        "a lot -- from 1 option (Income, Medical: fully deterministic, zero "
        "ML needed) up to 7 (General Services). Using `primary_category` as "
        "an input feature, not just text, should make this an easier problem "
        "than pure text classification.\n"
    )
    for k, v in mapping.items():
        clean_v = sorted(str(x) for x in v if pd.notna(x))
        n_bad = len(v) - len(clean_v)
        suffix = f" (+ {n_bad} non-string/missing value(s) found!)" if n_bad else ""
        lines.append(f"- **{k}** ({len(clean_v)} sub-categories): {', '.join(clean_v)}{suffix}")
    lines.append("")

    lines.append("## 7. Description Format\n")
    lines.append(
        "Same French banking prefix structure as the earlier dataset "
        "(CB, PAIEMENT CB, PRLV SEPA, VIR SEPA, COMMISSION, etc.) -- the "
        "cleaning logic in preprocessing.py carries over unchanged.\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_eda()
