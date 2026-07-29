# COPD Exacerbation Prediction — Multimodal ML Framework
> **Anas Shawah'en** | Clinician-Researcher | Respiratory Sciences  
> Emergency Medicine & Critical Care | Applied Data Science  
> 📧 anasshwahen442@gmail.com | [LinkedIn](https://www.linkedin.com/in/anas-shawah-en-b74168165)

---

## Overview

Acute exacerbations of COPD (AECOPD) are the leading cause of disease-related hospitalization and mortality. Current clinical detection relies on symptom escalation — by which point the therapeutic window has often narrowed significantly.

This project develops a **multimodal machine learning framework** for **pre-symptomatic AECOPD detection**. The target is an early-warning window of approximately 3–5 days before symptom escalation — this window is a research objective to be validated through prospective lead-time analysis, not yet an established result of the current models. The framework integrates:

- 🎙️ **Audiomics** — smartphone-based cough and exhalation acoustics (MFCCs, spectral centroid, zero-crossing rate)
- 🧪 **Inflammatory biomarkers** — CRP, eosinophil counts
- 📋 **Patient-reported parameters** — dyspnea scores, medication use, activity level

The core computational challenge is extracting separable, clinically meaningful signatures from overlapping, multivariate time-series data — audio, biomarker, and patient-reported signals that individually offer limited specificity but may carry complementary information when combined.

---

## Current Results

Preliminary results below reflect models trained on structured clinical variables only; audiomics and biomarker integration are in progress (see Roadmap).

| Model               | Mean AUC (CV) | Test AUC |
|---------------------|---------------|----------|
| Logistic Regression | 0.813         | **0.838** |
| Random Forest       | —             | in progress |

> Evaluated using 5-fold cross-validation on structured clinical variables.  
> Audio feature integration: next phase (see Roadmap).

---

## Project Structure
