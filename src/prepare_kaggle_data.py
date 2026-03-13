"""
Kaggle Dataset Adapter — kartik2112/fraud-detection
Loads fraudTrain.csv + fraudTest.csv, engineers features to match
our standard transaction schema, saves to data/transactions.csv.

Run: python src/prepare_kaggle_data.py
"""
import os
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "fraudTrain.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "fraudTest.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "transactions.csv")

# Keep this many legit transactions (fraud rows are kept in full)
LEGIT_SAMPLE = 49_000
SEED = 42


# ── Haversine distance ─────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def prepare(legit_sample: int = LEGIT_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    print("Loading Kaggle dataset...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop unnamed index column that Kaggle adds
    df = df.drop(columns=["loaded"], errors="ignore")

    print(f"Total rows: {len(df):,}  |  Fraud: {df['is_fraud'].sum():,}  ({df['is_fraud'].mean():.2%})")

    # ── Sample: keep all fraud + sample of legit ──────────────────────────────
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0].sample(n=min(legit_sample, len(df[df["is_fraud"] == 0])), random_state=seed)
    df = pd.concat([fraud_df, legit_df], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"After sampling: {len(df):,} rows  |  Fraud: {df['is_fraud'].sum():,}  ({df['is_fraud'].mean():.2%})")

    # ── Datetime features ─────────────────────────────────────────────────────
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # ── Online flag: categories ending in _net are card-not-present ──────────
    df["is_online"] = df["category"].str.endswith("_net").astype(int)

    # ── Distance: cardholder home → merchant ─────────────────────────────────
    print("Computing distances (this may take ~30 seconds)...")
    df["distance_from_home_km"] = np.vectorize(haversine_km)(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # ── Per-account rolling features ──────────────────────────────────────────
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    # avg amount over last 30 transactions (proxy for 30-day average)
    df["avg_amount_30d"] = (
        df.groupby("cc_num")["amt"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        .fillna(df["amt"])
        .round(2)
    )

    # transactions in last 24h: count per account per calendar date
    df["_date"] = df["trans_date_trans_time"].dt.date
    df["num_transactions_24h"] = df.groupby(["cc_num", "_date"])["amt"].transform("count")
    df = df.drop(columns=["_date"])

    # account age: days since first recorded transaction for this card
    first_tx = df.groupby("cc_num")["trans_date_trans_time"].transform("min")
    df["account_age_days"] = (df["trans_date_trans_time"] - first_tx).dt.days + 1

    # amount vs average ratio
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

    # ── Final column selection ────────────────────────────────────────────────
    keep = [
        "transaction_id", "account_id", "amount", "merchant_category",
        "merchant_name", "hour", "day_of_week", "is_online",
        "distance_from_home_km", "num_transactions_24h", "account_age_days",
        "avg_amount_30d", "amount_vs_avg_ratio", "fraud",
    ]
    df = df[keep].reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Final: {len(df):,} rows  |  {df['fraud'].sum():,} fraud  ({df['fraud'].mean():.2%} rate)")
    print(df.head(3))
    return df


if __name__ == "__main__":
    prepare()
