"""
Synthetic bank transaction data generator for fraud detection.
Generates realistic transaction data with ~1% fraud rate.
Run: python src/generate_data.py
"""
import pandas as pd
import numpy as np
import os

SEED = 42
N_TRANSACTIONS = 5000
FRAUD_RATE = 0.01

MERCHANT_CATEGORIES = ["grocery", "online_retail", "travel", "restaurant", "gas_station", "entertainment", "atm", "pharmacy"]
MERCHANT_NAMES = {
    "grocery": ["FreshMart", "SuperSave", "ValueGrocer", "DailyBasket"],
    "online_retail": ["ShopNow", "QuickBuy", "NetStore", "EasyShop"],
    "travel": ["SkyFly", "RailGo", "HotelStay", "TravelEase"],
    "restaurant": ["TasteBite", "QuickEats", "FoodHub", "DineFine"],
    "gas_station": ["FuelStop", "QuickGas", "PetroPoint", "EasyFuel"],
    "entertainment": ["CinePlus", "GameZone", "EventHub", "FunPark"],
    "atm": ["BankATM", "QuickCash", "MoneyStop", "CashPoint"],
    "pharmacy": ["HealthPlus", "MedStore", "PharmaCare", "WellnessRx"],
}


def generate_data(n: int = N_TRANSACTIONS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    def make_transactions(n_rows: int, is_fraud: bool) -> dict:
        categories = rng.choice(MERCHANT_CATEGORIES, n_rows)
        names = [rng.choice(MERCHANT_NAMES[c]) for c in categories]

        if is_fraud:
            amounts = rng.uniform(500, 5000, n_rows)
            hours = rng.choice(list(range(0, 5)) + list(range(22, 24)), n_rows)
            distance_km = rng.uniform(50, 2000, n_rows)
            is_online = rng.choice([0, 1], n_rows, p=[0.3, 0.7])
            num_tx_24h = rng.integers(5, 20, n_rows)
            account_age_days = rng.integers(1, 90, n_rows)
            avg_amount_30d = rng.uniform(20, 150, n_rows)
        else:
            amounts = rng.exponential(scale=80, size=n_rows).clip(5, 800)
            hours = rng.integers(7, 22, n_rows)
            distance_km = rng.exponential(scale=5, size=n_rows).clip(0, 30)
            is_online = rng.choice([0, 1], n_rows, p=[0.6, 0.4])
            num_tx_24h = rng.integers(1, 6, n_rows)
            account_age_days = rng.integers(90, 3650, n_rows)
            avg_amount_30d = rng.uniform(40, 300, n_rows)

        amount_vs_avg = np.round(amounts / (avg_amount_30d + 1e-6), 4)

        return {
            "amount": np.round(amounts, 2),
            "merchant_category": categories,
            "merchant_name": names,
            "hour": hours,
            "day_of_week": rng.integers(0, 7, n_rows),
            "is_online": is_online,
            "distance_from_home_km": np.round(distance_km, 2),
            "num_transactions_24h": num_tx_24h,
            "account_age_days": account_age_days,
            "avg_amount_30d": np.round(avg_amount_30d, 2),
            "amount_vs_avg_ratio": amount_vs_avg,
            "fraud": int(is_fraud),
        }

    legit = make_transactions(n_legit, is_fraud=False)
    fraud = make_transactions(n_fraud, is_fraud=True)

    frames = []
    for d in [legit, fraud]:
        frames.append(pd.DataFrame(d))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add IDs
    df.insert(0, "transaction_id", [f"TXN{str(i).zfill(5)}" for i in range(len(df))])
    df.insert(1, "account_id", [f"ACC{str(rng.integers(1, 500)).zfill(4)}" for _ in range(len(df))])

    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    df = generate_data()
    out_path = os.path.join(out_dir, "transactions.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} transactions → {out_path}")
    print(f"Fraud rate: {df['fraud'].mean():.2%}  ({df['fraud'].sum()} fraud cases)")
    print(df[df["fraud"] == 1].head(3))
