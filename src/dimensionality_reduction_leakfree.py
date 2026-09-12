import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer

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
X = df[clinical_cols + audio_cols].fillna(df[clinical_cols + audio_cols].mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_pipeline(reducer):
    """reducer runs on audio_cols only; clinical_cols pass through untouched.
    Everything happens inside the pipeline -> fit only sees training folds."""
    pre = ColumnTransformer([
        ("clinical", StandardScaler(), clinical_cols),
        ("audio", Pipeline([("scale", StandardScaler()), ("reduce", reducer)]), audio_cols),
    ])
    return pre


def eval_leak_free(name, reducer):
    pre = build_pipeline(reducer)
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=42))])
    rf_pre = ColumnTransformer([
        ("clinical", "passthrough", clinical_cols),
        ("audio", reducer, audio_cols),
    ])
    rf = Pipeline([("pre", rf_pre), ("clf", RandomForestClassifier(random_state=42, n_estimators=300))])
    lr_auc = cross_val_score(lr, X, Y, cv=cv, scoring='roc_auc')
    rf_auc = cross_val_score(rf, X, Y, cv=cv, scoring='roc_auc')
    print(f"[{name}]  LogReg AUC={lr_auc.mean():.3f} (folds {np.round(lr_auc,2)})  "
          f"RF AUC={rf_auc.mean():.3f} (folds {np.round(rf_auc,2)})")


print("=== Leak-free re-evaluation (feature selection nested inside CV) ===")
eval_leak_free("PCA k=5",  PCA(n_components=5, random_state=42))
eval_leak_free("PCA k=8",  PCA(n_components=8, random_state=42))
eval_leak_free("SelectKBest k=5", SelectKBest(f_classif, k=5))
eval_leak_free("SelectKBest k=8", SelectKBest(f_classif, k=8))

print("\n=== Reference: clinical-only (no audio) ===")
clin_only = X[clinical_cols]
lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))])
rf = RandomForestClassifier(random_state=42, n_estimators=300)
print(f"LogReg AUC={cross_val_score(lr, clin_only, Y, cv=cv, scoring='roc_auc').mean():.3f}  "
      f"RF AUC={cross_val_score(rf, clin_only, Y, cv=cv, scoring='roc_auc').mean():.3f}")
