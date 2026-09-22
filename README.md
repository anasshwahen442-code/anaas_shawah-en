# COPD Exacerbation Prediction — Multimodal ML Framework
Anas Shawah'en | Clinician-Researcher | Respiratory & Cardiac Sciences | Applied Clinical Machine Learning
📧 anasshwahen442@gmail.com | LinkedIn

![Tests](https://img.shields.io/badge/tests-passing-brightgreen) ![Data Status](https://img.shields.io/badge/data-synthetic-yellow) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

**In brief:** A methodological pilot for predicting COPD exacerbations from clinical variables, biomarkers, and cough audio, built entirely on synthetic data (n=50) to validate the analysis pipeline before real-patient data collection (IRB protocol drafted).
**What it demonstrates:** leakage-free cross-validation, honest uncertainty reporting (bootstrap CIs), regression tests for past errors, and a documented self-correction.
**What it does not show:** clinical performance. Do not cite the AUC figures as findings.

## Status: Pilot Sample Only — Not a Clinical Study — Read This First

This repository is a methodological pilot sample, nothing more. It exists to demonstrate technical and analytical readiness — that the code, statistics, and documentation practices are sound — before any real patient is ever involved. It is not a clinical study, not a validated tool, and not evidence of real-world model performance.

Every result in this repository is computed on synthetic data structured to resemble the planned real-world variables. No real patient data has been used, collected, or accessed at any point. This phase exists to validate the analysis pipeline — feature extraction, cross-validation design, and leakage controls — before IRB-approved real-world data collection begins (see docs/). Do not cite AUC figures below as clinical findings; cite them as pilot-sample, pipeline-validation evidence only.

## 1. Clinical Rationale

Acute exacerbations of chronic obstructive pulmonary disease (AECOPD) are events of acute worsening of respiratory symptoms requiring additional therapy, per the GOLD 2023 definition, and remain a leading cause of disease-related hospitalization and mortality worldwide. Current detection is reactive: clinicians act once symptoms have already escalated, by which point the therapeutic window has narrowed. A growing body of work on remote respiratory monitoring — acoustic cough analysis, wearable-derived activity/dyspnea signals, and inflammatory biomarker trends — motivates the hypothesis that a multimodal signal may precede overt symptom escalation by several days.

This project is a methodological pilot toward testing that hypothesis, integrating:

- Audiomics — smartphone-recorded cough/exhalation acoustics (MFCCs, spectral centroid, zero-crossing rate, spectral bandwidth, RMS energy), extracted with librosa
- Inflammatory biomarkers — CRP, peripheral eosinophil count
- Patient-reported & utilization data — CAT score, mMRC dyspnea scale, medication adherence, activity level, ER visit history

### 1.1 Beyond Model Accuracy: Interpretation and Trust

A risk score is only useful if clinicians interpret it correctly and act on it appropriately. This pilot reports discrimination (AUC) and states explicitly that calibration is not yet assessed — a clinician reading "65% risk" will treat it as a literal probability, and a discriminative-but-uncalibrated model can mislead. A planned next step is calibration assessment on real data, alongside qualitative work on how clinicians interpret and act on AI-generated risk output in time-pressured settings.

## 2. Results (synthetic pilot, n=50)

Single source of truth: `src/model_comparison.py` — every number below is its direct, reproducible output (`results/model_comparison.csv`).

| Model | Features | Mean CV AUC | Fold Std | 95% CI (bootstrap) |
|---|---|---|---|---|
| Logistic Regression (clinical only) | 9 | 0.700 | 0.133 | [0.524, 0.834] |
| Random Forest (clinical only) | 9 | 0.654 | 0.160 | [0.458, 0.781] |
| SVM – RBF (clinical only) | 9 | 0.800 | 0.049 | [0.615, 0.885] |
| Logistic Regression + Audio PCA(k=8), leak-free | 41 | 0.775 | 0.068 | [0.608, 0.875] |

Clarification on the headline AUC figure: The highest AUC in this table (0.800) belongs to the clinical-only SVM model — it does not include audio features. The multimodal configuration that actually integrates audiomics (Logistic Regression + Audio PCA) scored 0.775. If you have seen this project's AUC cited elsewhere as a "multimodal" result, that 0.80 figure should be read as the clinical-only baseline, not the multimodal pipeline. See the CI overlap discussion immediately below — at n=50, neither figure is statistically distinguishable from the other.

Reading this table like a computer scientist, not just a clinician: SVM has both the highest mean AUC and the lowest fold-to-fold variance (0.049 vs. 0.133 for LogReg). But the 95% bootstrap confidence intervals above overlap substantially across every model in this table (e.g. LogReg's upper bound of 0.834 sits inside SVM's interval) — at n=50, the apparent ranking between models is not statistically distinguishable from noise. SVM is the best point estimate and the most stable across folds, which is a reasonable basis for choosing it as the working model going forward, but "SVM beats LogReg" is not a claim this pilot can support with statistical confidence. That confirmation is exactly what the planned real-data validation study (`docs/Clinical_Validation_Plan.docx`) exists to provide.

Calibration is not yet assessed. The AUC numbers above describe discrimination (can the model rank higher-risk patients above lower-risk ones) — they say nothing about calibration (whether a patient assigned "65% risk" by `src/add_risk_scores.py` actually experiences the outcome roughly 65% of the time). A discriminative-but-uncalibrated model can still mislead a clinician reading a percentage at face value. Calibration assessment (reliability diagrams, Brier score) is deferred to the real-data validation phase, where sample size can support it, but is flagged here explicitly rather than left implicit.

### 2.1 A documented methodological failure (kept deliberately visible)

An earlier iteration of this pipeline selected the top-8 audio features using `SelectKBest` fit on the entire dataset before cross-validation. That produced an AUC of 0.850 for the same clinical+audio configuration that honestly scores 0.708 once feature selection is correctly nested inside each CV fold (see `src/dimensionality_reduction_leakfree.py` and `tests/test_pipeline.py::test_feature_selection_leakage_regression`). This is left in the repository history and documentation deliberately: it is more useful to a reviewer as evidence of methodological self-correction than it would be if scrubbed out.

The originally reported 0.813/0.838 AUC figures (an earlier project stage, before this repository's current history) were never reproducible: the source script had a corrupted newline encoding (would not execute) and a label-generation threshold placing every simulated patient in one class.

### 2.2 GOLD ABE risk stratification (exploratory, `src/add_risk_scores.py`)

Out-of-fold predicted exacerbation risk and a GOLD 2023 ABE proxy group are computed per synthetic patient. Known limitation, stated plainly: the current synthetic CAT-score distribution skews the ABE grouping heavily toward group B (35/50 patients), because the generator's CAT-score mean (~18) sits well above the GOLD B threshold (>=10). This is a synthetic-data generation artifact, not a clinical finding, and is asserted against by `tests/test_pipeline.py::test_gold_group_uses_all_three_categories` so it can't silently worsen unnoticed. It is not urgent to fix before real-data collection (the real cohort will have its own, real distribution), but recalibrating `src/generate_dataset.py`'s CAT-score generator to better approximate published GOLD-group prevalence would make this synthetic pilot a more informative dry run of the real analysis.

## 3. Engineering Practices Applied

- No un-nested feature selection. Any transform that "looks at labels" (PCA on a label-correlated subset, SelectKBest, etc.) is fit inside a Pipeline/ColumnTransformer, never on the full dataset before `cross_val_score`.
- Fixed, shared CV folds (`StratifiedKFold(..., random_state=42)`) across every model in `model_comparison.py`, so AUC differences reflect the model, not different train/test splits.
- Explicit `scoring='roc_auc'` everywhere — `cross_val_score`'s default (accuracy) silently misrepresented results in the original code.
- Regression tests (`tests/test_pipeline.py`) codify the three failure modes already encountered in this project (degenerate labels, CV leakage, silent data corruption) so they can't reappear unnoticed.
- Derived data never overwrites source data — `add_risk_scores.py` reads `copd_multimodal_dataset.csv` and writes a new file, never overwriting the input in place.

## 4. Repository Structure

```
README.md
requirements.txt
.gitignore
src/
  copd_exacerbation_analysis.py
  audiomics_features.py
  generate_dataset.py
  generate_synthetic_audio.py
  build_multimodal_dataset.py
  compare_multimodal.py
  dimensionality_reduction_leakfree.py
  model_comparison.py
  add_risk_scores.py
tests/
  test_pipeline.py
data/
  copd_synthetic_dataset.csv
  copd_synthetic_dataset.xlsx
  copd_multimodal_dataset.csv
  copd_multimodal_dataset_with_risk.csv
  cough_audio_manifest.csv
docs/
  COPD_Study_Protocol_IRB_Draft.docx
results/
  confusion_matrices.png
  model_comparison.csv
  model_comparison.md
```

## 5. Reproducing These Results

```bash
pip install -r requirements.txt
python src/generate_dataset.py
python src/generate_synthetic_audio.py
python src/build_multimodal_dataset.py
python src/model_comparison.py
python src/add_risk_scores.py
pytest tests/
```

## 6. Limitations (stated explicitly, not buried)

- n=50, single synthetic cohort. No real-world generalization claim is made or implied.
- Audio is synthetic, generated from a simple noise-burst model loosely tied to FEV1/mMRC — it is a pipeline stand-in, not evidence that real cough acoustics carry this signal.
- GOLD ABE grouping uses a proxy (ER visits as a stand-in for "moderate exacerbations," which GOLD defines more specifically) — flagged in code and tests, not just in prose.
- No external validation cohort. All numbers are in-sample CV on one synthetic dataset.

## 7. Roadmap

- [x] Fix corrupted analysis script (encoding, threshold, AUC scoring)
- [x] Build and verify audiomics feature-extraction module
- [x] Multimodal pipeline with leakage-free cross-validation
- [x] Consolidate all model comparisons into one reproducible script
- [x] Add regression tests for known failure modes
- [x] Draft IRB protocol for real-data collection
- [ ] Calibration assessment (reliability diagrams, Brier score) on real data
- [ ] Qualitative work on clinician interpretation of AI risk output (planned)
- [ ] Biostatistician review of sample-size calculation
- [ ] IRB submission and approval
- [ ] Real patient data collection (~150–175 patients, target)
- [ ] Re-run pipeline on real data; report validated AUC with 95% CI

## About

COPD, Respiratory, Machine Learning, Clinical Decision Support, Audiomics

License: MIT
