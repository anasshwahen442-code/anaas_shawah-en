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

Confidence intervals: mean/std across 5 CV folds describes fold-to-fold
*variance*, not the *precision* of the AUC estimate itself. With n=50 that
distinction matters -- two models can look "different" by mean AUC while
their bootstrap CIs overlap heavily, meaning the apparent ranking is not
statistically distinguishable from noise. We report both.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000  # resamples for the AUC confidence interval

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


def bootstrap_auc_ci(y_true, y_proba, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Nonparametric 95% CI for AUC via patient-level bootstrap resampling
    of out-of-fold predicted probabilities. This bounds the precision of the
    AUC point estimate itself -- distinct from (and complementary to) the
    fold-to-fold std, which only describes variance across the 5 CV splits.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y_true[idx], y_proba[idx]
        if len(np.unique(y_b)) < 2:
            continue  # skip degenerate resamples (both classes required for AUC)
        boot_scores.append(roc_auc_score(y_b, p_b))
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def evaluate(name: str, estimator, X, y) -> dict:
    scores = cross_val_score(estimator, X, y, cv=cv(), scoring="roc_auc")

    # Out-of-fold probabilities (leak-free: each patient's probability comes
    # from a fold where that patient was held out) feed the bootstrap CI.
    oof_proba = cross_val_predict(estimator, X, y, cv=cv(), method="predict_proba")[:, 1]
    ci_lo, ci_hi = bootstrap_auc_ci(y, oof_proba)

    return {
        "model": name,
        "n_features": X.shape[1] if hasattr(X, "shape") else None,
        "mean_auc": round(float(scores.mean()), 3),
        "std_auc": round(float(scores.std()), 3),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
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
              f"   95% CI [{result['ci95_lo']:.3f}, {result['ci95_hi']:.3f}]"
              f"   folds={result['fold_aucs']}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results/model_comparison.csv", index=False)

    with open("results/model_comparison.md", "w") as f:
        f.write("| Model | Features | Mean CV AUC | Fold Std | 95% CI (bootstrap) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['n_features']} | {r['mean_auc']:.3f} | {r['std_auc']:.3f} "
                    f"| [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] |\n")

    print("\nSaved results/model_comparison.csv and results/model_comparison.md")
    print("\nReminder: n=50, synthetic data. The 95% CI bounds the precision of each")
    print("AUC estimate; where CIs overlap substantially across models, the apparent")
    print("ranking is not statistically distinguishable from noise at this sample size.")


if __name__ == "__main__":
    main()
