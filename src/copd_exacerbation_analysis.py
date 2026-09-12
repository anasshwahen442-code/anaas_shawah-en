import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the real (currently synthetic-placeholder) patient dataset instead of
# generating a random toy sample inline. Replace this path with the real
# de-identified export once available -- the column names below already
# match what was agreed on for the 50-patient pilot.
dataset = pd.read_csv('copd_synthetic_dataset.csv')
print(dataset.head())

feature_cols = [
    'Age', 'Gender', 'Smoking_Status', 'FEV1_pct_predicted',
    'CRP_mg_L', 'Eosinophils_cells_uL', 'CAT_Score', 'mMRC_Dyspnea',
    'Medication_Adherence_pct', 'Activity_Steps_per_day', 'ER_Visits_Last_6mo'
]
X = dataset[feature_cols]
X = pd.get_dummies(X, columns=['Gender', 'Smoking_Status'], drop_first=True)
Y = dataset['Exacerbation_90d']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Logistic regression needs scaled inputs -- Activity_Steps_per_day (~thousands)
# otherwise dominates CRP/CAT/mMRC (single digits) and the solver fails to converge.
log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
rf_clf = RandomForestClassifier(random_state=42)  # tree splits are scale-invariant, no scaler needed

# --- Cross-validated AUC (explicit scoring, was defaulting to accuracy) ---
log_reg_cv_auc = cross_val_score(log_reg, X_train, Y_train, cv=5, scoring='roc_auc')
rf_cv_auc = cross_val_score(rf_clf, X_train, Y_train, cv=5, scoring='roc_auc')

print(f"Logistic Regression CV AUC: {log_reg_cv_auc}")
print(f"Mean CV AUC (LogReg): {np.mean(log_reg_cv_auc):.3f}")
print(f"Random Forest CV AUC: {rf_cv_auc}")
print(f"Mean CV AUC (RF): {np.mean(rf_cv_auc):.3f}")

# --- Fit on full training set ---
log_reg.fit(X_train, Y_train)
rf_clf.fit(X_train, Y_train)

# --- Test-set AUC (uses predict_proba, not predict, which classification_report needs) ---
log_reg_test_auc = roc_auc_score(Y_test, log_reg.predict_proba(X_test)[:, 1])
rf_test_auc = roc_auc_score(Y_test, rf_clf.predict_proba(X_test)[:, 1])
print(f"\nLogistic Regression Test AUC: {log_reg_test_auc:.3f}")
print(f"Random Forest Test AUC: {rf_test_auc:.3f}")

Y_pred_log_reg = log_reg.predict(X_test)
Y_pred_rf = rf_clf.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(Y_test, Y_pred_log_reg, zero_division=0))
print("\nRandom Forest Classification Report:")
print(classification_report(Y_test, Y_pred_rf, zero_division=0))

log_reg_cm = confusion_matrix(Y_test, Y_pred_log_reg)
rf_cm = confusion_matrix(Y_test, Y_pred_rf)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Logistic Regression Confusion Matrix')
plt.imshow(log_reg_cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(2), ['No Exacerbation', 'Exacerbation'])
plt.yticks(np.arange(2), ['No Exacerbation', 'Exacerbation'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.subplot(1, 2, 2)
plt.title('Random Forest Confusion Matrix')
plt.imshow(rf_cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(2), ['No Exacerbation', 'Exacerbation'])
plt.yticks(np.arange(2), ['No Exacerbation', 'Exacerbation'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
print("\nSaved confusion_matrices.png")
