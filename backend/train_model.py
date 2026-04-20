"""
train_model.py - Train and save the fraud detection model
Run this script once to generate 'model.joblib' and 'scaler.joblib'
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os, json

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'creditcard.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'metrics.json')
STATS_PATH = os.path.join(os.path.dirname(__file__), 'stats.json')

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Data cleaning
df = df.drop_duplicates()

# ── Save dataset stats ─────────────────────────────────────────────────
total = len(df)
fraud_count = int(df['Class'].sum())
normal_count = int(total - fraud_count)
fraud_pct = round(fraud_count / total * 100, 4)

stats = {
    "total_transactions": total,
    "fraud_count": fraud_count,
    "normal_count": normal_count,
    "fraud_percentage": fraud_pct,
    "normal_percentage": round(100 - fraud_pct, 4),
    "num_features": 30,
    "amount_mean": round(float(df['Amount'].mean()), 2),
    "amount_max": round(float(df['Amount'].max()), 2),
    "amount_median": round(float(df['Amount'].median()), 2),
    "fraud_amount_mean": round(float(df[df['Class'] == 1]['Amount'].mean()), 2),
    "normal_amount_mean": round(float(df[df['Class'] == 0]['Amount'].mean()), 2),
}
with open(STATS_PATH, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Stats saved → {STATS_PATH}")

# ── Feature Engineering ────────────────────────────────────────────────
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE oversampling
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ── Train Models ───────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred, y_prob=None):
    metrics = {
        "name": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = None
    return metrics

all_metrics = []

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
all_metrics.append(evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr))

# Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train_res, y_train_res)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
all_metrics.append(evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt))

# Random Forest (tuned)
print("Training Random Forest (tuned)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train_res, y_train_res)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf > 0.4).astype(int)  # threshold tuning
all_metrics.append(evaluate_model("Random Forest (Tuned)", y_test, y_pred_rf, y_prob_rf))

# Isolation Forest (anomaly detection)
print("Training Isolation Forest...")
iso = IsolationForest(contamination=0.0017, random_state=42)
iso.fit(X_train_res)
y_pred_iso_raw = iso.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso_raw]
all_metrics.append(evaluate_model("Isolation Forest", y_test, y_pred_iso))

# ── Save the best model (RF) ───────────────────────────────────────────
# We also save a scaler trained specifically on Amount/Time for inference
scaler_final = StandardScaler()
# Fit on original (pre-SMOTE) train data for inference usage
scaler_final.fit(X_train[['Amount', 'Time']])
joblib.dump(rf, MODEL_PATH)
joblib.dump(scaler_final, SCALER_PATH)
print(f"Model saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")

# ── Save metrics ───────────────────────────────────────────────────────
with open(METRICS_PATH, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"Metrics saved → {METRICS_PATH}")

print("\n✅ Training complete!")
for m in all_metrics:
    print(f"  {m['name']}: Precision={m['precision']}, Recall={m['recall']}, F1={m['f1_score']}")
