# Baseline vs. Syntax Feature Evaluation

## Overview
This report compares baseline stylometry features (function words, character n-grams, and sentence length stats) against precomputed syntactic features and their combination using simple classifiers.

## Dataset
```json
{
  "data_path": "data/processed/gutenberg_christie_lovecraft_syntax.parquet",
  "samples": 34085,
  "labels": [
    "label"
  ],
  "syntax_features": 21
}
```

## Target: label
| label   | feature_set   | model               |   accuracy |   f1_macro |
|:--------|:--------------|:--------------------|-----------:|-----------:|
| label   | baseline      | linear_svm          |      0.969 |      0.930 |
| label   | combined      | linear_svm          |      1.000 |      1.000 |
| label   | syntax        | linear_svm          |      1.000 |      1.000 |
| label   | baseline      | logistic_regression |      0.953 |      0.888 |
| label   | combined      | logistic_regression |      0.999 |      0.997 |
| label   | syntax        | logistic_regression |      0.999 |      0.997 |
| label   | baseline      | random_forest       |      0.947 |      0.874 |
| label   | combined      | random_forest       |      0.995 |      0.988 |
| label   | syntax        | random_forest       |      0.997 |      0.993 |

## Plots
![label_accuracy](reports/figures/label_accuracy.png)
![label_f1_macro](reports/figures/label_f1_macro.png)
