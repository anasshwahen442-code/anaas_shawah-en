import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("/home/claude/copd_multimodal_dataset.csv")
Y = df["Exacerbation_90d"]

clinical_cols = [
    'Age', 'FEV1_pct_predicted', 'CRP_mg_L', 'Eosinophils_cells_uL',
    'CAT_Score', 'mMRC_Dyspnea', 'Medication_Adherence_pct',
    'Activity_Steps_per_day', 'ER_Visits_Last_6mo'
]
audio_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["mfcc", "spectral_", "zcr", "rms"]
)]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_feature_set(name, cols):
    X = df[cols].fillna(df[cols].mean())
    log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    rf = RandomForestClassifier(random_state=42, n_estimators=300)
    lr_auc = cross_val_score(log_reg, X, Y, cv=cv, scoring='roc_auc')
    rf_auc = cross_val_score(rf, X, Y, cv=cv, scoring='roc_auc')
    print(f"\n[{name}]  n_features={len(cols)}")
    print(f"  LogReg CV AUC: mean={lr_auc.mean():.3f}  folds={np.round(lr_auc,2)}")
    print(f"  RF     CV AUC: mean={rf_auc.mean():.3f}  folds={np.round(rf_auc,2)}")
    return lr_auc.mean(), rf_auc.mean()

clinical_lr, clinical_rf = eval_feature_set("Clinical + biomarkers only", clinical_cols)
audio_lr, audio_rf = eval_feature_set("Audiomics only", audio_cols)
combo_lr, combo_rf = eval_feature_set("Multimodal (clinical + audiomics)", clinical_cols + audio_cols)

print("\n--- Summary: does adding audio help? ---")
print(f"RF   clinical-only: {clinical_rf:.3f}  ->  multimodal: {combo_rf:.3f}  (delta {combo_rf-clinical_rf:+.3f})")
print(f"LogReg clinical-only: {clinical_lr:.3f}  ->  multimodal: {combo_lr:.3f}  (delta {combo_lr-clinical_lr:+.3f})")
