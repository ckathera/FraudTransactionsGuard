"""
ML Engine - Bank Transaction Fraud Detection
Models: XGBoost (primary) vs RandomForest (baseline comparison)
Handles class imbalance via SMOTE + scale_pos_weight
"""
import joblib
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.joblib")
ISO_PATH = os.path.join(DATA_DIR, "isolation_forest.joblib")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.joblib")
TRAIN_PATH = os.path.join(DATA_DIR, "transactions_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "transactions_test.csv")

# Feature config
CATEGORICAL = ["merchant_category"]
NUMERIC = [
    "amount",
    "hour",
    "day_of_week",
    "is_online",
    "distance_from_home_km",
    "num_transactions_24h",
    "account_age_days",
    "avg_amount_30d",
    "amount_vs_avg_ratio",
]
FEATURES = NUMERIC + ["merchant_category_encoded"]
TARGET = "fraud"
PRIMARY_MODEL_NAME = "XGBoost"


def _encode(df: pd.DataFrame, le: LabelEncoder | None = None):
    """Label-encode merchant_category. Fit if le is None."""
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"])
    else:
        df["merchant_category_encoded"] = le.transform(df["merchant_category"])
    return df, le


def train(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH) -> dict:
    """Train benchmark models, report metrics, and save XGBoost as the primary model."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train, le = _encode(df_train)
    df_test, _ = _encode(df_test, le)

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test, y_test = df_test[FEATURES], df_test[TARGET]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}
    trained_models = {}
    best_name, best_f1 = "", 0.0

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        results[name] = {"f1": round(f1, 4), "roc_auc": round(auc, 4)}
        print(f"{name}: F1={f1:.4f}  ROC-AUC={auc:.4f}")
        if f1 > best_f1:
            best_f1, best_name = f1, name

    # Choose the actual best model found by benchmarking (fall back to PRIMARY_MODEL_NAME)
    selected_name = best_name if best_name else PRIMARY_MODEL_NAME
    selected_model = trained_models[selected_name]
    print(f"\nBenchmark winner: {best_name}  (F1={best_f1:.4f})")
    print(f"Selected primary model for deployment: {selected_name}")
    print(
        classification_report(
            y_test,
            selected_model.predict(X_test),
            target_names=["Legit", "Fraud"],
        )
    )

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train[FEATURES])

    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    joblib.dump(iso, ISO_PATH)
    joblib.dump(
        {
            "merchant_category": le,
            "best_model_name": best_name,
            "selected_model_name": selected_name,
        },
        ENCODER_PATH,
    )

    print(f"\nModels saved -> {DATA_DIR}")
    return {
        "best_model": best_name,
        "selected_model": PRIMARY_MODEL_NAME,
        "metrics": results,
    }


def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    iso = joblib.load(ISO_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, iso, encoders


def score_transaction(tx: dict) -> dict:
    """
    Score a single transaction dict.
    Returns: fraud_probability, anomaly_score, risk_level, top_risk_factors.
    """
    model, iso, encoders = _load_artifacts()
    le = encoders["merchant_category"]

    row = pd.DataFrame([tx])

    known = list(le.classes_)
    if tx.get("merchant_category") not in known:
        row["merchant_category"] = known[0]
    row, _ = _encode(row, le)

    X = row[FEATURES]

    prob = float(model.predict_proba(X)[0][1])
    anomaly_raw = float(iso.score_samples(X)[0])
    anomaly_score = round(1 / (1 + np.exp(anomaly_raw * 5)), 4)

    if prob >= 0.70:
        risk_level = "HIGH"
    elif prob >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    top_factors = {}
    if hasattr(model, "feature_importances_"):
        importances = {
            f: round(float(v), 4)
            for f, v in zip(FEATURES, model.feature_importances_)
        }
        top_factors = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]
        )

    return {
        "fraud_probability": round(prob, 4),
        "anomaly_score": anomaly_score,
        "risk_level": risk_level,
        "top_risk_factors": top_factors,
    }


if __name__ == "__main__":
    train()
