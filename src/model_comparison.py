"""
model_comparison.py
--------------------
Single source of truth for model performance in this project.

Rationale: results were previously scattered across several ad-hoc scripts
(compare_multimodal.py, dimensionality_reduction_leakfree.py, add_risk_scores.py),
each re-implementing its own CV loop. That made it easy for numbers to drift
out of sync with the README (which is exactly how the original 0.813/0.838
AUC figures ended up unverifiable). This script is the single, reproducible
entry point for every reported number: run it, and every value in the
README's results table should match its output exactly.

Usage:
    python src/model_comparison.py

Outputs:
    results/model_comparison.csv   (machine-readable)
    results/model_comparison.md    (paste directly into README)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5

CLINICAL_COLS = [
    "Age", "FEV1_pct_predicted", "CRP_mg_L", "Eosinophils_cells_uL",
    "CAT_Score", "mMRC_Dyspnea", "Medication_Adherence_pct",
    "Activity_Steps_per_day", "ER_Visits_Last_6mo",
]


def load_data(path: str = "data/copd_multimodal_dataset.csv") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    audio_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ("mfcc", "spectral_", "zcr", "rms")
    )]
    return df, df["Exacerbation_90d"], audio_cols


def cv() -> StratifiedKFold:
    # Fixed fold assignment across every model in this file -- required for
    # a fair head-to-head comparison (different fold splits would make small
    # AUC differences meaningless with n=50).
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def evaluate(name: str, estimator, X, y) -> dict:
    scores = cross_val_score(estimator, X, y, cv=cv(), scoring="roc_auc")
    return {
        "model": name,
        "n_features": X.shape[1] if hasattr(X, "shape") else None,
        "mean_auc": round(float(scores.mean()), 3),
        "std_auc": round(float(scores.std()), 3),
        "fold_aucs": [round(float(s), 2) for s in scores],
    }


def build_models(clinical_cols: list[str], audio_cols: list[str]) -> dict:
    """Every model configuration reported in the README, defined once."""
    models = {}

    models["Logistic Regression (clinical only)"] = (
        Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        clinical_cols,
    )
    models["Random Forest (clinical only)"] = (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        clinical_cols,
    )
    models["SVM - RBF (clinical only)"] = (
        Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE))]),
        clinical_cols,
    )

    if audio_cols:
        # PCA fit only inside each CV fold via ColumnTransformer -- prevents
        # the feature-selection leakage documented in the project history
        # (an earlier, un-nested run of this same configuration reported an
        # inflated AUC of 0.850; the leak-free number is 0.775).
        pre = ColumnTransformer([
            ("clinical", StandardScaler(), clinical_cols),
            ("audio_pca", Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=8, random_state=RANDOM_STATE))]), audio_cols),
        ])
        models["Logistic Regression + Audio PCA(k=8), leak-free"] = (
            Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
            clinical_cols + audio_cols,
        )

    return models


def main():
    df, y, audio_cols = load_data()
    models = build_models(CLINICAL_COLS, audio_cols)

    rows = []
    for name, (estimator, cols) in models.items():
        X = df[cols].fillna(df[cols].mean())
        result = evaluate(name, estimator, X, y)
        rows.append(result)
        print(f"{name:50s}  AUC = {result['mean_auc']:.3f} +/- {result['std_auc']:.3f}"
              f"   folds={result['fold_aucs']}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results/model_comparison.csv", index=False)

    with open("results/model_comparison.md", "w") as f:
        f.write("| Model | Features | Mean CV AUC | Std |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['n_features']} | {r['mean_auc']:.3f} | {r['std_auc']:.3f} |\n")

    print("\nSaved results/model_comparison.csv and results/model_comparison.md")
    print("\nReminder: n=50, synthetic data. Std AUC columns show real fold-to-fold\n"
          "variance -- treat differences smaller than ~0.05 as noise, not a winner.")


if __name__ == "__main__":
    main()
