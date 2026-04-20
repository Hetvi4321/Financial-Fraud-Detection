"""
app.py - Flask REST API for Financial Fraud Detection
Endpoints:
  GET  /api/stats          → Dataset overview statistics
  GET  /api/model-metrics  → Model performance comparison
  GET  /api/eda            → EDA chart data
  POST /api/predict        → Real-time fraud prediction
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

BASE_DIR = os.path.dirname(__file__)

# ── Load pre-computed artifacts (saved by train_model.py) ──────────────
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_model():
    path = os.path.join(BASE_DIR, 'model.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_scaler():
    path = os.path.join(BASE_DIR, 'scaler.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()
scaler = load_scaler()
stats_data = load_json(os.path.join(BASE_DIR, 'stats.json'))
metrics_data = load_json(os.path.join(BASE_DIR, 'metrics.json'))

# ── Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "Fraud Detection API is running"})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return dataset overview statistics."""
    if stats_data:
        return jsonify(stats_data)
    # Fallback if stats.json not found (notebook outputs)
    return jsonify({
        "total_transactions": 284807,
        "fraud_count": 492,
        "normal_count": 284315,
        "fraud_percentage": 0.1727,
        "normal_percentage": 99.8273,
        "num_features": 30,
        "amount_mean": 88.35,
        "amount_max": 25691.16,
        "amount_median": 22.0,
        "fraud_amount_mean": 122.21,
        "normal_amount_mean": 88.29,
    })


@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    """Return model performance metrics for all trained models."""
    if metrics_data:
        return jsonify(metrics_data)
    # Fallback hardcoded from notebook outputs
    return jsonify([
        {
            "name": "Logistic Regression",
            "precision": 0.08,
            "recall": 0.91,
            "f1_score": 0.15,
            "accuracy": 0.97,
            "roc_auc": 0.97
        },
        {
            "name": "Decision Tree",
            "precision": 0.12,
            "recall": 0.82,
            "f1_score": 0.21,
            "accuracy": 0.98,
            "roc_auc": 0.90
        },
        {
            "name": "Random Forest (Tuned)",
            "precision": 0.35,
            "recall": 0.83,
            "f1_score": 0.49,
            "accuracy": 1.00,
            "roc_auc": 0.99
        },
        {
            "name": "Isolation Forest",
            "precision": 0.04,
            "recall": 0.27,
            "f1_score": 0.07,
            "accuracy": 0.95,
            "roc_auc": None
        },
    ])


@app.route('/api/eda', methods=['GET'])
def get_eda():
    """Return EDA data for chart rendering."""
    return jsonify({
        "class_distribution": {
            "labels": ["Normal (0)", "Fraud (1)"],
            "values": [284315, 492],
            "percentages": [99.83, 0.17]
        },
        "amount_by_class": {
            "fraud": {
                "mean": 122.21,
                "median": 9.25,
                "max": 2125.87,
                "bins": [0, 50, 100, 200, 500, 1000, 2200],
                "counts": [196, 74, 87, 73, 42, 20]
            },
            "normal": {
                "mean": 88.29,
                "median": 22.0,
                "max": 25691.16,
                "bins": [0, 50, 100, 200, 500, 1000, 25700],
                "counts": [120000, 52000, 45000, 38000, 22000, 7315]
            }
        },
        "sampling_comparison": {
            "before_smote": {"normal": 226602, "fraud": 378},
            "after_smote": {"normal": 226602, "fraud": 226602},
            "after_undersample": {"normal": 378, "fraud": 378}
        },
        "pipeline_steps": [
            {"step": 1, "name": "Data Loading", "desc": "Load creditcard.csv (284K rows)"},
            {"step": 2, "name": "Data Cleaning", "desc": "Remove 1,081 duplicate rows"},
            {"step": 3, "name": "EDA", "desc": "Analyze class imbalance & distributions"},
            {"step": 4, "name": "Feature Scaling", "desc": "StandardScaler on Amount & Time"},
            {"step": 5, "name": "SMOTE", "desc": "Balance dataset 226K → 226K each class"},
            {"step": 6, "name": "Model Training", "desc": "LR, DT, Random Forest trained"},
            {"step": 7, "name": "Hyperparameter Tuning", "desc": "GridSearchCV on RF (max_depth=10, n=100)"},
            {"step": 8, "name": "Threshold Tuning", "desc": "Threshold 0.4 boosts recall to 83%"},
            {"step": 9, "name": "Anomaly Detection", "desc": "Isolation Forest & LOF applied"},
            {"step": 10, "name": "Evaluation", "desc": "Precision, Recall, F1, ROC-AUC compared"},
        ]
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accept transaction features and return fraud probability.
    Expected JSON body:
    {
        "Amount": 149.62,
        "Time": 0.0,
        "V1": -1.36, "V2": -0.07, ..., "V28": -0.02
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # Build feature vector: Time, V1-V28, Amount (30 features, same order as training)
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        features = []
        for fname in feature_names:
            val = float(data.get(fname, 0.0))
            features.append(val)

        X = np.array(features).reshape(1, -1)

        # Scale Amount and Time (columns 0 and 29)
        if scaler is not None:
            amount_time = np.array([[X[0][29], X[0][0]]])
            scaled_at = scaler.transform(amount_time)
            X[0][29] = scaled_at[0][0]  # Amount scaled
            X[0][0] = scaled_at[0][1]   # Time scaled

        # Predict probability
        prob = model.predict_proba(X)[0][1]
        threshold = 0.4
        prediction = int(prob >= threshold)

        risk_level = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW")

        return jsonify({
            "fraud_probability": round(float(prob), 4),
            "prediction": prediction,
            "is_fraud": bool(prediction),
            "risk_level": risk_level,
            "threshold_used": threshold,
            "message": "⚠️ FRAUD DETECTED" if prediction else "✅ Transaction appears legitimate"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if model is None:
        print("⚠️  WARNING: model.joblib not found.")
        print("   Run 'python train_model.py' first to train the model.")
    else:
        print("✅ Model loaded successfully.")
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
