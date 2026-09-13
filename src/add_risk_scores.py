"""
Adds out-of-fold predicted risk and GOLD ABE group to the multimodal dataset.
Reads the existing dataset, writes a NEW file -- never overwrites the source
dataset in place, so raw data stays reproducible/diffable in git history.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SOURCE = "data/copd_multimodal_dataset.csv"
OUTPUT = "data/copd_multimodal_dataset_with_risk.csv"  # new file, source untouched

df = pd.read_csv(SOURCE)
Y = df["Exacerbation_90d"]

clinical_cols = [
    'Age', 'FEV1_pct_predicted', 'CRP_mg_L', 'Eosinophils_cells_uL',
    'CAT_Score', 'mMRC_Dyspnea', 'Medication_Adherence_pct',
    'Activity_Steps_per_day', 'ER_Visits_Last_6mo'
]
X_clin = df[clinical_cols]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
])

auc_scores = cross_val_score(pipe, X_clin, Y, cv=cv, scoring='roc_auc')
print(f"SVM CV AUC: {auc_scores.mean():.3f} (folds: {np.round(auc_scores,2)})")

# Out-of-fold predictions: each patient's probability comes from a fold that
# never trained on that patient -- same leak-free principle as the rest of
# this repo's pipeline (see src/dimensionality_reduction_leakfree.py).
oof_proba = cross_val_predict(pipe, X_clin, Y, cv=cv, method="predict_proba")[:, 1]
df['Predicted_Exacerbation_Risk_pct'] = (oof_proba * 100).round(1)

def gold_group(row):
    """GOLD 2023 ABE risk stratification (proxy, see README caveat).
    NOTE: with this synthetic CAT_Score distribution (mean~18), the mMRC/CAT
    threshold puts ~70% of patients in group B. This is a known artifact of
    the synthetic data generator, not a clinical finding -- flagged here so
    it isn't mistaken for a real stratification result."""
    if row['ER_Visits_Last_6mo'] >= 2:
        return 'E'
    if row['CAT_Score'] >= 10 or row['mMRC_Dyspnea'] >= 2:
        return 'B'
    return 'A'

df['GOLD_Group'] = df.apply(gold_group, axis=1)

print("\nGOLD group distribution (see skew caveat in code comments above):")
print(df['GOLD_Group'].value_counts().to_string())

df.to_csv(OUTPUT, index=False)
print(f"\nSaved {OUTPUT} (source file {SOURCE} left unmodified)")
