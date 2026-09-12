# COPD Exacerbation Prediction — Multimodal ML Framework

**Anas Shawah'en** | Clinician-Researcher | Respiratory Sciences
Emergency Medicine & Critical Care | Applied Data Science
📧 anasshwahen442@gmail.com | [LinkedIn](https://www.linkedin.com/in/anas-shawah-en-b74168165)

---

## Overview

Acute exacerbations of COPD (AECOPD) are the leading cause of disease-related hospitalization and mortality. Current clinical detection relies on symptom escalation — by which point the therapeutic window has often narrowed significantly.

This project develops a **multimodal machine learning framework** for **pre-symptomatic AECOPD detection**. The target is an early-warning window of approximately 3–5 days before symptom escalation — this window is a research objective to be validated through prospective lead-time analysis on real patient data, not yet an established result.

The framework integrates:
- 🎙️ **Audiomics** — smartphone-based cough and exhalation acoustics (MFCCs, spectral centroid, zero-crossing rate, spectral bandwidth, RMS energy)
- 🧪 **Inflammatory biomarkers** — CRP, eosinophil counts
- 📋 **Patient-reported parameters** — CAT score, mMRC dyspnea, medication adherence, activity level, ER visit history

## ⚠️ Status: Methodological Pilot on Synthetic Data

**All results below use synthetic/simulated data structured to match the real variables**, not real patient records. This phase exists to validate the analysis pipeline (feature extraction, cross-validation design, leakage controls) before real-world data collection begins under IRB approval (see `/docs`). Do not cite these numbers as clinical findings.

## Current Results (synthetic pilot, n=50)

| Model configuration | Mean CV AUC (5-fold, stratified) |
|---|---|
| Clinical + biomarkers only | 0.700 (LogReg) / 0.654 (RF) |
| + Audiomics, raw 32 features | 0.708 (LogReg) / 0.662 (RF) |
| + Audiomics, PCA (k=8), **leak-free** | **0.775 (LogReg)** / 0.562 (RF) |

**A methodological lesson worth stating plainly:** an earlier run selected top audio features using the *entire* dataset before cross-validation, producing an inflated AUC of 0.850. Once feature selection was correctly nested inside each CV fold, the honest number for that same configuration dropped to 0.708. The corrected, leak-free pipeline is what's in this repo (`src/dimensionality_reduction_leakfree.py`).

The previously reported 0.813/0.838 AUC figures were **not reproducible** — the original script had a corrupted newline encoding (wouldn't run at all) and a label-generation threshold that made every simulated patient fall in the same class. Both are fixed here.

## Repository Structure

```
├── README.md
├── src/
│   ├── copd_exacerbation_analysis.py      # Core clinical-only pipeline (fixed)
│   ├── audiomics_features.py              # librosa-based acoustic feature extraction
│   ├── generate_dataset.py                # Synthetic 50-patient clinical dataset generator
│   ├── generate_synthetic_audio.py        # Synthetic cough-clip generator (stand-in for real recordings)
│   ├── build_multimodal_dataset.py        # Merges audio features into the clinical dataset
│   ├── compare_multimodal.py              # Clinical-only vs multimodal comparison (unreduced)
│   └── dimensionality_reduction_leakfree.py  # PCA / SelectKBest, correctly nested in CV
├── data/
│   ├── copd_synthetic_dataset.csv / .xlsx # Clinical + biomarker synthetic data
│   ├── copd_multimodal_dataset.csv        # Clinical + audiomics merged
│   └── cough_audio_manifest.csv           # Per-clip file index
├── docs/
│   └── COPD_Study_Protocol_IRB_Draft.docx # Draft IRB protocol for real-world data collection
└── results/
    └── confusion_matrices.png
```

## Roadmap

- [x] Fix corrupted analysis script (encoding, threshold, AUC scoring)
- [x] Build and verify audiomics feature-extraction module
- [x] Multimodal pipeline with leakage-free cross-validation
- [x] Draft IRB protocol for real-data collection
- [ ] Biostatistician review of sample-size calculation
- [ ] IRB submission and approval
- [ ] Real patient data collection (~150–175 patients, target)
- [ ] Re-run pipeline on real data; report validated AUC with 95% CI
