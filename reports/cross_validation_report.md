# 5-Fold Cross-Validation Report

Robustness check on top of the single 80/20 holdout split: same models, same features, but scored across 5 different folds so a lucky/unlucky single split can't skew the numbers. `category` uses stratified folds (safe: 60 rows/class). `merchant` uses plain (non-stratified) folds because dozens of merchants have too few examples to stratify on -- some rare merchants may be entirely absent from a training fold, which is why a few models (esp. XGBoost) occasionally fail a fold on that target; failed folds are excluded from the mean/std and flagged in the Note column.

| Target   | Model                   |   F1 macro (mean) |   F1 macro (std) | Note   |
|:---------|:------------------------|------------------:|-----------------:|:-------|
| Category | Logistic Regression     |            0.9799 |           0.0097 |        |
| Category | Majority Class Baseline |            0.0014 |           0      |        |
| Merchant | Logistic Regression     |            0.9492 |           0.0164 |        |
| Merchant | Majority Class Baseline |            0      |           0      |        |