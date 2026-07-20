# Effyis Category and Merchant Classifier

Predicts BOTH a transaction's **category** and its **merchant** from
`description` text ONLY -- trained together, in the same run, from the
same features and the same train/test split. Uses Effyis' balanced,
officially-taxonomized dataset.

**Scope:** `description` -> full `category` field (37 classes, e.g.
"Food & Drink / Groceries", split into `primary_category` +
`detailed_category`) **and** `merchant` (292 classes, e.g. "Zodio"),
predicted side by side by every candidate model.

## The taxonomy problem is resolved
The earlier dataset's 13 simple category labels didn't match the 68-item
official reference taxonomy. This dataset fixes that directly:
`primary_category` + `detailed_category` together match the reference
file's Main/Sub structure exactly (37/37 combinations found in the data
exist in `categories_reference.csv`).

## Results -- both targets, same run, same split
Every candidate model is trained twice per run -- once per target --
on the exact same cleaned text, TF-IDF features, and 80/20 split
(stratified on category), so category and merchant results are directly
comparable model-for-model. Full detail in `reports/model_comparison.md`;
robustness check across 5 different splits in `reports/cross_validation_report.md`.

| Target | Classes | Best model | Accuracy (holdout) | F1 macro (holdout) | F1 macro (5-fold CV) |
|---|---|---|---|---|---|
| Category | 37 | Linear SVM | 97.07% | 97.04% | 98.84% +/- 0.36% |
| Merchant | 292 | Random Forest | 96.17% | 94.95% | 96.01% +/- 1.03% |

vs a majority-class baseline of 0.14% (category) / ~0% (merchant).

**Why merchant is a harder, noisier target than category:** 292 classes
vs 37, and dozens of merchants have only 1-4 examples in the whole
dataset. A few of those (e.g. "H&M" -> punctuation-stripped to "h m",
where both tokens are too short for TF-IDF to keep) lose their strongest
signal during cleaning and get misclassified; XGBoost specifically
requires every class to appear in every training fold, so it's skipped
on the merchant target only (reported, not hidden -- see the report's
note). None of this affects the category model, which is unaffected by
merchant-specific issues since it's the exact same pipeline evaluated
against a different, better-behaved label.

## Structure
```
effyis-category-classifier/
├── data/
│   ├── raw/
│   │   ├── transactions_fr_balanced.csv            <- labeled (has merchant/category)
│   │   ├── transactions_fr_balanced - Copie.csv     <- same transactions, used as the "unlabeled" delivery file
│   │   └── categories_reference.csv                  <- official 68-item taxonomy
│   └── processed/
│       ├── cleaned_transactions.csv                 <- output of preprocessing.py
│       └── predictions_comparison.csv               <- THE deliverable: honest OOF predictions, category + merchant
├── src/
│   ├── preprocessing.py         <- cleaning
│   ├── eda.py                   <- Phase 1 report + charts
│   ├── feature_engineering.py   <- TF-IDF on description ONLY (shared by both targets)
│   ├── models.py                <- model registry incl. majority-class baseline
│   ├── train.py                  <- trains every model on BOTH targets (category + merchant) in one run
│   ├── evaluate.py               <- ONE combined report: comparison tables + detail for both targets
│   ├── cross_validate.py         <- 5-fold CV robustness check, both targets
│   ├── test_on_holdout.py        <- standalone re-test of saved models on the 20% holdout, both targets
│   ├── predict.py                <- predict_transaction(): category AND merchant for one new description
│   ├── generate_predictions.py   <- produces predictions_comparison.csv (category + merchant), see methodology note
│   ├── feedback.py                <- logs human corrections (category) from the app
│   ├── incorporate_feedback.py   <- folds corrections back into training data
│   ├── label_generator.py        <- rule-based merchant->category lookup, used for the original labeling pass
│   └── run_app.py                <- launches app.py, auto-retrains on exit
├── app.py                        <- Streamlit interface: predicts + shows category AND merchant, live
├── models/     <- generated: shared feature_pipeline.joblib, label_encoder_{category,merchant}.joblib,
│                  and one {model}_{category,merchant}.joblib pair per candidate model
└── reports/    <- generated: eda_report.md, model_comparison.md, cross_validation_report.md,
                   holdout_test_report.md, merchant_per_class.csv, figures/
```

## Usage
```bash
pip install -r requirements.txt

python src/preprocessing.py         # Phase 2: clean the raw data
python src/eda.py                   # Phase 1: EDA report + charts
python src/evaluate.py              # Phase 4-5: train + compare all models, BOTH targets, one report
python src/cross_validate.py        # Phase 5b: 5-fold CV robustness check, both targets
python src/test_on_holdout.py       # standalone re-check of the saved models, both targets
python src/generate_predictions.py  # produces the comparison CSV, both targets
streamlit run app.py                # test new descriptions live in the browser (shows category + merchant)
```
Run in this order -- each script depends on the output of the one before it.

## One split, two targets -- why this matters
`train.py` splits the data ONCE (80/20, stratified on `category`, which
is perfectly balanced at 60 rows/class). Both the category model and the
merchant model for a given algorithm are then fit on that exact same
train rows and scored on that exact same test rows -- only the label
column differs. This is deliberate: it's what makes it fair to say
"Linear SVM got 97.1% on category and 96.6% on merchant" as one
comparable statement, instead of two disconnected experiments that
happened to reuse the same code.

`merchant` itself can't be stratified on directly (some merchants have
just 1 example in the whole dataset), so a handful of rare merchants may
end up test-only, meaning the model never had a chance to learn them.
That's a real, honestly reported limitation -- see the "hardest
merchants" section of `reports/model_comparison.md` -- not a hidden flaw.

## `predictions_comparison.csv` -- how it's actually built (read this)
This is built starting from the REAL unlabeled file
(`transactions_fr_balanced - Copie.csv`) -- not the labeled one directly.
BUT: that file and `transactions_fr_balanced.csv` contain the exact same
2,220 transactions. This matters a lot for methodology:

**There is no way to "train on the labeled file, then predict on the
unlabeled file" and get an honest match-rate** -- because they're
literally the same underlying rows. A model trained on all 2,220 labeled
rows would have directly memorized many of them, so predicting on the
"unlabeled" copy of those same rows would show artificially inflated,
near-100% accuracy for every model -- useless for actually choosing one.

So under the hood, this script uses `cross_val_predict` (5-fold: each
row predicted by a model that never saw it during training) on the
labeled data -- for BOTH category and merchant -- then attaches those
honest predictions to the real unlabeled file's structure by
`transaction_id`. The `match_category_<model>` and `match_merchant_<model>`
columns are trustworthy specifically because of this.

## Choosing a model
Open `predictions_comparison.csv`, look at the `match_category_<model>`
and `match_merchant_<model>` columns, or just read the executive summary
at the top of `reports/model_comparison.md`. Linear SVM is the strongest,
fastest all-round choice for category; Linear SVM and Random Forest are
essentially tied for merchant (Random Forest edges it out slightly on
this split, Linear SVM is ~5x faster to train) -- either is a
well-justified default.

## Next Steps
1. Tests + interpretability, mirroring the PFA mock project
2. Extend the feedback/correction loop (`feedback.py`) to also capture
   merchant corrections, not just category
3. Apply the FINAL pipeline (retrained on all labeled rows) to
   `transactions_fr_balanced - Copie.csv` for the actual delivery predictions
