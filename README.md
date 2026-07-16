# Effyis Category Classifier

Predicts a transaction's category **from `description` text ONLY** --
no `primary_category` input, per current scope. Uses Effyis' balanced,
officially-taxonomized dataset.

**Scope:** `description` -> full `category` field (37 classes, e.g.
"Food & Drink / Groceries"), then split into `primary_category` and
`detailed_category`. Merchant extraction is still a later phase.

## The taxonomy problem is resolved
The earlier dataset's 13 simple category labels didn't match the 68-item
official reference taxonomy. This dataset fixes that directly:
`primary_category` + `detailed_category` together match the reference
file's Main/Sub structure exactly (37/37 combinations found in the data
exist in `category_reference.csv`).

## Results (text-only, no primary_category feature)
**Linear SVM: 98.84% F1 (macro), confirmed by 5-fold cross-validation**
(single-split: 97.04%, cross-validated: 98.84% +/- 0.36%) -- vs a 0.14%
F1 majority-class baseline.

| Model | F1 macro (5-fold CV) |
|---|---|
| Linear SVM | 98.84% |
| Logistic Regression | 98.25% |
| Naive Bayes | 96.95% |
| Random Forest | 96.85% |
| Baseline | 0.14% |

Note: an earlier version of this project also used `primary_category`
as an input feature and got slightly higher accuracy (98.76% single-split).
Current version deliberately drops it to test text-only performance --
about a 1.8-point cost, which is a reasonable, quantified trade-off if
description-only prediction is the actual requirement.

## Structure
```
effyis-category-classifier/
├── data/
│   ├── raw/
│   │   ├── transactions_fr_balanced.csv            <- labeled (has merchant/category)
│   │   ├── transactions_fr_balanced_unlabeled.csv   <- same transactions, those fields blanked
│   │   └── category_reference.csv                  <- official 68-item taxonomy
│   └── processed/
│       ├── cleaned_transactions.csv                 <- output of preprocessing.py
│       └── predictions_comparison.csv               <- THE deliverable, see below
├── src/
│   ├── preprocessing.py         <- cleaning
│   ├── eda.py                   <- Phase 1 report + charts
│   ├── feature_engineering.py   <- TF-IDF on description ONLY
│   ├── models.py                <- model registry incl. majority-class baseline
│   ├── train.py                  <- trains all models, target = full category field
│   ├── evaluate.py               <- comparison table + confusion matrix
│   ├── cross_validate.py         <- 5-fold CV robustness check
│   ├── predict.py                <- predict ONE new description (used by app.py)
│   └── generate_predictions.py   <- produces predictions_comparison.csv (see methodology note)
├── app.py                        <- Streamlit interface for testing new descriptions live
├── models/     <- generated: saved .joblib files (pipeline, label encoder, each model)
└── reports/    <- generated: eda_report.md, model_comparison.md, figures/
```

## Usage
```bash
pip install -r requirements.txt

python src/preprocessing.py         # Phase 2: clean the raw data
python src/eda.py                   # Phase 1: EDA report + charts
python src/evaluate.py              # Phase 4-5: train + compare all models
python src/cross_validate.py        # Phase 5b: 5-fold CV robustness check
python src/generate_predictions.py  # produces the comparison CSV
streamlit run app.py                # test new descriptions live in the browser
```
Run in this order -- each script depends on the output of the one before it.

## `predictions_comparison.csv` -- how it's actually built (read this)
This is built starting from the REAL unlabeled file
(`transactions_fr_balanced_unlabeled.csv`) -- not the labeled one directly.
BUT: `transactions_fr_balanced_unlabeled.csv` and `transactions_fr_balanced.csv`
contain the exact same 2,220 transactions (the unlabeled file is just
those same rows with `merchant`/`detailed_category`/`category` blanked
out). This matters a lot for methodology:

**There is no way to "train on the labeled file, then predict on the
unlabeled file" and get an honest match-rate** -- because they're
literally the same underlying rows. A model trained on all 2,220 labeled
rows would have directly memorized many of them, so predicting on the
"unlabeled" copy of those same rows would show artificially inflated,
near-100% accuracy for every model -- useless for actually choosing one.

So under the hood, this script uses `cross_val_predict` (5-fold: each
row predicted by a model that never saw it during training) on the
labeled data, then attaches those honest predictions to the real
unlabeled file's structure by `transaction_id`. The `match_<model>`
columns are trustworthy specifically because of this -- that's what
makes them usable for comparing and choosing a model.

## Choosing a model
Open `predictions_comparison.csv`, look at the `match_<model>` columns.
Linear SVM has the highest honest match rate (98.8%) and is also the
fastest to train (0.18s) -- a well-justified default unless you have a
specific reason to prefer another model.

## Next Steps
1. Merchant extraction (hybrid rules + ML fallback) -- still not started
2. Once merchant extraction works, apply the FINAL pipeline (retrained
   on all labeled rows) to `transactions_fr_balanced_unlabeled.csv` for
   the actual delivery predictions
3. Tests + interpretability, mirroring the PFA mock project, once
   merchant extraction is also validated