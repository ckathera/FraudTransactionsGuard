"""
Kaggle Dataset Adapter — kartik2112/fraud-detection
Loads fraudTrain.csv + fraudTest.csv, engineers features to match
our standard transaction schema.

Outputs:
  data/transactions_train.csv  — processed training set
  data/transactions_test.csv   — processed test set
  data/transactions.csv        — combined (used by the app UI)

Run: python src/prepare_kaggle_data.py
"""
import os
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
KAGGLE_TRAIN   = os.path.join(BASE_DIR, "data", "fraudTrain.csv")
KAGGLE_TEST    = os.path.join(BASE_DIR, "data", "fraudTest.csv")
TRAIN_OUT_PATH = os.path.join(BASE_DIR, "data", "transactions_train.csv")
TEST_OUT_PATH  = os.path.join(BASE_DIR, "data", "transactions_test.csv")
OUT_PATH       = os.path.join(BASE_DIR, "data", "transactions.csv")

# Keep this many legit transactions per split (fraud rows are kept in full)
LEGIT_SAMPLE = 49_000
SEED = 42


# ── Haversine distance ─────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def _engineer(df: pd.DataFrame, label: str, legit_sample: int, seed: int) -> pd.DataFrame:
    """Feature-engineer a single raw Kaggle dataframe."""
    df = df.drop(columns=["loaded"], errors="ignore")

    print(f"[{label}] rows: {len(df):,}  |  fraud: {df['is_fraud'].sum():,}  ({df['is_fraud'].mean():.2%})")

    # ── Sample: keep all fraud + sample of legit ──────────────────────────────
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0].sample(
        n=min(legit_sample, len(df[df["is_fraud"] == 0])), random_state=seed
    )
    df = pd.concat([fraud_df, legit_df], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    print(f"[{label}] after sampling: {len(df):,} rows")

    # ── Datetime features ─────────────────────────────────────────────────────
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # ── Online flag: categories ending in _net are card-not-present ──────────
    df["is_online"] = df["category"].str.endswith("_net").astype(int)

    # ── Distance: cardholder home → merchant ─────────────────────────────────
    print(f"[{label}] computing distances...")
    df["distance_from_home_km"] = np.vectorize(haversine_km)(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # ── Per-account rolling features ──────────────────────────────────────────
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    df["avg_amount_30d"] = (
        df.groupby("cc_num")["amt"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        .fillna(df["amt"])
        .round(2)
    )

    df["_date"] = df["trans_date_trans_time"].dt.date
    df["num_transactions_24h"] = df.groupby(["cc_num", "_date"])["amt"].transform("count")
    df = df.drop(columns=["_date"])

    first_tx = df.groupby("cc_num")["trans_date_trans_time"].transform("min")
    df["account_age_days"] = (df["trans_date_trans_time"] - first_tx).dt.days + 1

    df["amount_vs_avg_ratio"] = (df["amt"] / (df["avg_amount_30d"] + 1e-6)).round(4)

    # ── Rename to standard schema ─────────────────────────────────────────────
    df["merchant_name"] = df["merchant"].str.replace("fraud_", "", regex=False)
    df["account_id"]    = "ACC" + df["cc_num"].astype(str).str[-4:]

    df = df.rename(columns={
        "trans_num": "transaction_id",
        "amt":       "amount",
        "category":  "merchant_category",
        "is_fraud":  "fraud",
    })

    keep = [
        "transaction_id", "account_id", "amount", "merchant_category",
        "merchant_name", "hour", "day_of_week", "is_online",
        "distance_from_home_km", "num_transactions_24h", "account_age_days",
        "avg_amount_30d", "amount_vs_avg_ratio", "fraud",
    ]
    return df[keep].reset_index(drop=True)


def prepare(legit_sample: int = LEGIT_SAMPLE, seed: int = SEED):
    print("Loading Kaggle datasets...")
    df_train = _engineer(pd.read_csv(KAGGLE_TRAIN), "train", legit_sample, seed)
    df_test  = _engineer(pd.read_csv(KAGGLE_TEST),  "test",  legit_sample, seed)

    df_train.to_csv(TRAIN_OUT_PATH, index=False)
    print(f"Saved → {TRAIN_OUT_PATH}  ({len(df_train):,} rows)")

    df_test.to_csv(TEST_OUT_PATH, index=False)
    print(f"Saved → {TEST_OUT_PATH}  ({len(df_test):,} rows)")

    # Combined file for the app UI
    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}  ({len(combined):,} rows, combined for app UI)")

    return df_train, df_test


if __name__ == "__main__":
    prepare()
