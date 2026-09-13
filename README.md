# COPD Exacerbation Prediction — Multimodal ML Framework

**Anas Shawah'en** | Clinician-Researcher | Respiratory Sciences
Emergency Medicine & Critical Care | Applied Data Science
📧 anasshwahen442@gmail.com | [LinkedIn](https://www.linkedin.com/in/anas-shawah-en-b74168165)

[![Tests](https://img.shields.io/badge/tests-5%20passing-brightgreen)]()
[![Data](https://img.shields.io/badge/data-synthetic--only-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ⚠️ Status: Methodological Pilot on Synthetic Data — Read This First

**Every result in this repository is computed on synthetic data structured to resemble the planned real-world variables. No real patient data has been used or accessed.** This phase exists to validate the analysis pipeline — feature extraction, cross-validation design, and leakage controls — *before* IRB-approved real-world data collection begins (see `docs/`). Do not cite AUC figures below as clinical findings; cite them as pipeline-validation evidence.

## 1. Clinical Rationale

Acute exacerbations of chronic obstructive pulmonary disease (AECOPD) are events of acute worsening of respiratory symptoms requiring additional therapy, per the GOLD 2023 definition, and remain a leading cause of disease-related hospitalization and mortality worldwide. Current detection is reactive: clinicians act once symptoms have already escalated, by which point the therapeutic window has narrowed. A growing body of work on remote respiratory monitoring — acoustic cough analysis, wearable-derived activity/dyspnea signals, and inflammatory biomarker trends — motivates the hypothesis that a multimodal signal may precede overt symptom escalation by several days.

This project is a methodological pilot toward testing that hypothesis, integrating:
- 🎙️ **Audiomics** — smartphone-recorded cough/exhalation acoustics (MFCCs, spectral centroid, zero-crossing rate, spectral bandwidth, RMS energy), extracted with `librosa`
- 🧪 **Inflammatory biomarkers** — CRP, peripheral eosinophil count
- 📋 **Patient-reported & utilization data** — CAT score, mMRC dyspnea scale, medication adherence, activity level, ER visit history

## 2. Results (synthetic pilot, n=50)

Single source of truth: `src/model_comparison.py` — every number below is its direct, reproducible output (`results/model_comparison.csv`).

| Model | Features | Mean CV AUC | Fold Std | 95% CI (bootstrap) |
|---|---|---|---|---|
| Logistic Regression (clinical only) | 9 | 0.700 | 0.133 | [0.524, 0.834] |
| Random Forest (clinical only) | 9 | 0.654 | 0.160 | [0.458, 0.781] |
| **SVM – RBF (clinical only)** | 9 | **0.800** | **0.049** | [0.615, 0.885] |
| Logistic Regression + Audio PCA(k=8), leak-free | 41 | 0.775 | 0.068 | [0.608, 0.875] |

**Reading this table like a computer scientist, not just a clinician:** SVM has both the highest mean AUC *and* the lowest fold-to-fold variance (0.049 vs. 0.133 for LogReg). But the 95% bootstrap confidence intervals above overlap substantially across every model in this table (e.g. LogReg's upper bound of 0.834 sits inside SVM's interval) — at n=50, **the apparent ranking between models is not statistically distinguishable from noise**. SVM is the best point estimate and the most stable across folds, which is a reasonable basis for choosing it as the working model going forward, but "SVM beats LogReg" is not a claim this pilot can support with statistical confidence. That confirmation is exactly what the planned real-data validation study (`docs/Clinical_Validation_Plan.docx`) exists to provide.

**Calibration is not yet assessed.** The AUC numbers above describe *discrimination* (can the model rank higher-risk patients above lower-risk ones) — they say nothing about *calibration* (whether a patient assigned "65% risk" by `src/add_risk_scores.py` actually experiences the outcome roughly 65% of the time). A discriminative-but-uncalibrated model can still mislead a clinician reading a percentage at face value. Calibration assessment (reliability diagrams, Brier score) is deferred to the real-data validation phase, where sample size can support it, but is flagged here explicitly rather than left implicit.

### 2.1 A documented methodological failure (kept deliberately visible)

An earlier iteration of this pipeline selected the top-8 audio features using `SelectKBest` fit on the **entire dataset before cross-validation**. That produced an AUC of **0.850** for the same clinical+audio configuration that honestly scores **0.708** once feature selection is correctly nested inside each CV fold (see `src/dimensionality_reduction_leakfree.py` and `tests/test_pipeline.py::test_feature_selection_leakage_regression`). This is left in the repository history and documentation deliberately: it is more useful to a reviewer as evidence of methodological self-correction than it would be if scrubbed out.

The originally reported 0.813/0.838 AUC figures (an earlier project stage, before this repository's current history) were never reproducible: the source script had a corrupted newline encoding (would not execute) and a label-generation threshold placing every simulated patient in one class.

### 2.2 GOLD ABE risk stratification (exploratory, `src/add_risk_scores.py`)

Out-of-fold predicted exacerbation risk and a GOLD 2023 ABE proxy group are computed per synthetic patient. **Known limitation, stated plainly:** the current synthetic CAT-score distribution skews the ABE grouping heavily toward group B (35/50 patients), because the generator's CAT-score mean (~18) sits well above the GOLD B threshold (≥10). This is a synthetic-data generation artifact, not a clinical finding, and is asserted against by `tests/test_pipeline.py::test_gold_group_uses_all_three_categories` so it can't silently worsen unnoticed. It is not urgent to fix before real-data collection (the real cohort will have its own, real distribution), but recalibrating `src/generate_dataset.py`'s CAT-score generator to better approximate published GOLD-group prevalence would make this synthetic pilot a more informative dry run of the real analysis.

## 3. Engineering Practices Applied

- **No un-nested feature selection.** Any transform that "looks at labels" (PCA on a label-correlated subset, `SelectKBest`, etc.) is fit inside a `Pipeline`/`ColumnTransformer`, never on the full dataset before `cross_val_score`.
- **Fixed, shared CV folds** (`StratifiedKFold(..., random_state=42)`) across every model in `model_comparison.py`, so AUC differences reflect the model, not different train/test splits.
- **Explicit `scoring='roc_auc'`** everywhere — `cross_val_score`'s default (accuracy) silently misrepresented results in the original code.
- **Regression tests** (`tests/test_pipeline.py`) codify the three failure modes already encountered in this project (degenerate labels, CV leakage, silent data corruption) so they can't reappear unnoticed.
- **Derived data never overwrites source data** — `add_risk_scores.py` reads `copd_multimodal_dataset.csv` and writes a new file, never overwriting the input in place.

## 4. Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore                              # blocks real patient data / audio / PHI from ever being committed
├── src/
│   ├── copd_exacerbation_analysis.py       # Core clinical-only pipeline (fixed)
│   ├── audiomics_features.py               # librosa-based acoustic feature extraction
│   ├── generate_dataset.py                 # Synthetic 50-patient clinical dataset generator
│   ├── generate_synthetic_audio.py         # Synthetic cough-clip generator (stand-in for real recordings)
│   ├── build_multimodal_dataset.py         # Merges audio features into the clinical dataset
│   ├── compare_multimodal.py               # Early clinical-vs-multimodal comparison (superseded by model_comparison.py)
│   ├── dimensionality_reduction_leakfree.py   # PCA / SelectKBest, correctly nested in CV
│   ├── model_comparison.py                 # ⭐ Single source of truth for all reported AUC numbers
│   └── add_risk_scores.py                  # Out-of-fold risk %, GOLD ABE group (writes a new file)
├── tests/
│   └── test_pipeline.py                    # Regression tests for known past failure modes
├── data/
│   ├── copd_synthetic_dataset.csv / .xlsx  # Clinical + biomarker synthetic data
│   ├── copd_multimodal_dataset.csv         # Clinical + audiomics merged
│   ├── copd_multimodal_dataset_with_risk.csv  # + predicted risk % and GOLD group (generated)
│   └── cough_audio_manifest.csv            # Per-clip file index (synthetic clips, not included)
├── docs/
│   └── COPD_Study_Protocol_IRB_Draft.docx  # Draft IRB protocol for real-world data collection
└── results/
    ├── confusion_matrices.png
    ├── model_comparison.csv                # Machine-readable results (regenerate: python src/model_comparison.py)
    └── model_comparison.md
```

## 5. Reproducing These Results

```bash
pip install -r requirements.txt
python src/generate_dataset.py              # regenerate synthetic clinical data
python src/generate_synthetic_audio.py      # regenerate synthetic cough clips
python src/build_multimodal_dataset.py      # merge audio features into clinical data
python src/model_comparison.py              # regenerate every AUC number in this README
python src/add_risk_scores.py               # add risk %/ GOLD group columns
pytest tests/                               # verify all methodological safeguards hold
```

## 6. Limitations (stated explicitly, not buried)

- **n=50, single synthetic cohort.** No real-world generalization claim is made or implied.
- **Audio is synthetic**, generated from a simple noise-burst model loosely tied to FEV1/mMRC — it is a pipeline stand-in, not evidence that real cough acoustics carry this signal.
- **GOLD ABE grouping uses a proxy** (ER visits as a stand-in for "moderate exacerbations," which GOLD defines more specifically) — flagged in code and tests, not just in prose.
- **No external validation cohort.** All numbers are in-sample CV on one synthetic dataset.

## 7. Roadmap

- [x] Fix corrupted analysis script (encoding, threshold, AUC scoring)
- [x] Build and verify audiomics feature-extraction module
- [x] Multimodal pipeline with leakage-free cross-validation
- [x] Consolidate all model comparisons into one reproducible script
- [x] Add regression tests for known failure modes
- [x] Draft IRB protocol for real-data collection
- [ ] Biostatistician review of sample-size calculation
- [ ] IRB submission and approval
- [ ] Real patient data collection (~150–175 patients, target)
- [ ] Re-run pipeline on real data; report validated AUC with 95% CI
