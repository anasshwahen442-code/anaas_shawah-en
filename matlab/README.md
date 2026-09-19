# MATLAB Pilot Implementation

This folder contains a basic MATLAB implementation of the clinical-only logistic regression baseline, replicating the core methodology of the main Python pipeline in this repository.

## Files
- Copd.m — Loads copd_multimodal_dataset.csv, fits a logistic regression model on 9 clinical-only features, runs 5-fold cross-validation, and computes ROC/AUC per fold.
- roc_curve_matlab.png — ROC curves for all 5 folds.

## Results (synthetic pilot, n=50)
Mean AUC: 0.742 | Std: 0.133
Fold AUCs: 0.542, 0.833, 0.833, 0.667, 0.833

Consistent with the Python clinical-only Logistic Regression result (0.700). Small differences expected due to different cross-validation fold assignment between MATLAB and Python.

Same caveats as the main pipeline apply: synthetic data (n=50), pilot-sample only, not a clinical finding.

## Reproducing
Requires MATLAB with Statistics and Machine Learning Toolbox.
run('Copd.m')
