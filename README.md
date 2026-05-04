ش# COPD Exacerbation Prediction — Multimodal ML Framework

> **Anas Shawah'en** | Clinician-Researcher | Respiratory Sciences  
> Emergency Medicine & Critical Care | Applied Data Science  
> 📧 anasshwahen442@gmail.com | [LinkedIn](https://www.linkedin.com/in/anas-shawah-en-b74168165)

---

## Overview

Acute exacerbations of COPD (AECOPD) are the leading cause of disease-related hospitalization and mortality. Current clinical detection relies on symptom escalation — by which point the therapeutic window has often narrowed significantly.

This project develops a **multimodal machine learning framework** for **pre-symptomatic AECOPD detection**, targeting a 3–5 day early-warning window before symptom onset. The framework integrates:

- 🎙️ **Audiomics** — smartphone-based cough and exhalation acoustics (MFCCs, spectral centroid, zero-crossing rate)
- 🧪 **Inflammatory biomarkers** — CRP, eosinophil counts
- 📋 **Patient-reported parameters** — dyspnea scores, medication use, activity level

The core computational challenge mirrors gas sensor array cross-sensitivity analysis: extracting separable biological signatures from overlapping, multivariate time-series signals.

---

## Current Results

| Model               | Mean AUC (CV) | Test AUC |
|---------------------|---------------|----------|
| Logistic Regression | 0.813         | **0.838** |
| Random Forest       | —             | in progress |

> Evaluated using 5-fold cross-validation on structured clinical variables.  
> Audio feature integration: next phase (see Roadmap).

---

## Project Structure
