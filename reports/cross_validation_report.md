# 5-Fold Cross-Validation Report

Robustness check on top of the single 80/20 holdout split: same models, same features, but scored across 5 different folds so a lucky/unlucky single split can't skew the numbers. `category` uses stratified folds (safe: 60 rows/class). `merchant` uses plain (non-stratified) folds because dozens of merchants have too few examples to stratify on -- some rare merchants may be entirely absent from a training fold, which is why a few models (esp. XGBoost) occasionally fail a fold on that target; failed folds are excluded from the mean/std and flagged in the Note column.

| Target   | Model                   |   F1 macro (mean) |   F1 macro (std) | Note             |
|:---------|:------------------------|------------------:|-----------------:|:-----------------|
| Category | Linear SVM              |            0.988  |           0.0011 |                  |
| Category | Logistic Regression     |            0.9825 |           0.0052 |                  |
| Category | Naive Bayes             |            0.9691 |           0.0082 |                  |
| Category | Random Forest           |            0.9678 |           0.0059 |                  |
| Category | XGBoost                 |            0.927  |           0.0154 |                  |
| Category | LightGBM                |            0.4906 |           0.0317 |                  |
| Category | Majority Class Baseline |            0.0014 |           0      |                  |
| Merchant | Random Forest           |            0.9619 |           0.0119 |                  |
| Merchant | Linear SVM              |            0.9589 |           0.0075 |                  |
| Merchant | Logistic Regression     |            0.9465 |           0.0111 |                  |
| Merchant | XGBoost                 |            0.8123 |           0.0231 | 3/5 folds failed |
| Merchant | Naive Bayes             |            0.6611 |           0.025  |                  |
| Merchant | LightGBM                |            0.3144 |           0.0254 |                  |
| Merchant | Majority Class Baseline |            0      |           0      |                  |