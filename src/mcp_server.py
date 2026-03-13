"""
MCP Server — FraudGuard Bank Tools
Exposes 5 fraud investigation tools via the Model Context Protocol (FastMCP).

Run standalone:  python src/mcp_server.py
The agent imports these same functions directly for in-process calls.
"""
import json
import os
import sys

import pandas as pd

# Allow imports from src/ when run as standalone server
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from ml_engine import score_transaction

mcp = FastMCP("fraudguard-bank-tools")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")


def _load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# ── Tool 1 ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_transaction_details(transaction_id: str) -> str:
    """
    Retrieve full details of a bank transaction by its ID.
    Returns merchant, amount, timestamp info, account ID, and all features.
    """
    df = _load_data()
    row = df[df["transaction_id"] == transaction_id]
    if row.empty:
        return json.dumps({"error": f"Transaction {transaction_id} not found."})
    record = row.iloc[0].to_dict()
    record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
    return json.dumps(record)


# ── Tool 2 ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def score_fraud(transaction_id: str) -> str:
    """
    Run the ML fraud scoring model (XGBoost + Isolation Forest) on a transaction.
    Returns: fraud_probability (0-1), anomaly_score, risk_level (HIGH/MEDIUM/LOW),
    and top risk factors with their importance weights.
    """
    df = _load_data()
    row = df[df["transaction_id"] == transaction_id]
    if row.empty:
        return json.dumps({"error": f"Transaction {transaction_id} not found."})
    tx = row.iloc[0].to_dict()
    result = score_transaction(tx)
    return json.dumps(result)


# ── Tool 3 ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_account_history(account_id: str, limit: int = 10) -> str:
    """
    Retrieve the most recent transactions for a given account.
    Useful for spotting velocity patterns, repeated fraud attempts, or unusual behavior.
    """
    df = _load_data()
    account_txns = df[df["account_id"] == account_id].tail(limit)
    if account_txns.empty:
        return json.dumps({"error": f"No transactions found for account {account_id}."})
    records = account_txns[
        ["transaction_id", "amount", "merchant_category", "merchant_name", "hour", "is_online", "fraud"]
    ].to_dict(orient="records")
    summary = {
        "account_id": account_id,
        "total_transactions_shown": len(records),
        "fraud_count_in_history": int(account_txns["fraud"].sum()),
        "transactions": records,
    }
    return json.dumps(summary)


# ── Tool 4 ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def check_velocity(account_id: str) -> str:
    """
    Check transaction velocity for an account — how many transactions occurred
    and the total amount spent. Flags accounts with high transaction frequency
    which is a key indicator of card testing or automated fraud.
    """
    df = _load_data()
    account_txns = df[df["account_id"] == account_id]
    if account_txns.empty:
        return json.dumps({"error": f"Account {account_id} not found."})

    recent = account_txns[account_txns["num_transactions_24h"] > 0]
    max_velocity = int(account_txns["num_transactions_24h"].max())
    avg_velocity = float(account_txns["num_transactions_24h"].mean())
    total_amount = float(account_txns["amount"].sum())
    online_ratio = float(account_txns["is_online"].mean())

    flags = []
    if max_velocity >= 10:
        flags.append(f"HIGH velocity detected: {max_velocity} transactions in a 24h window")
    if online_ratio > 0.8:
        flags.append(f"High online transaction ratio: {online_ratio:.0%}")
    if total_amount > 10000:
        flags.append(f"High cumulative spend: ${total_amount:,.2f}")

    return json.dumps({
        "account_id": account_id,
        "max_transactions_24h": max_velocity,
        "avg_transactions_24h": round(avg_velocity, 2),
        "total_amount_all_time": round(total_amount, 2),
        "online_ratio": round(online_ratio, 4),
        "velocity_flags": flags,
    })


# ── Tool 5 ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_merchant_risk(merchant_category: str) -> str:
    """
    Return the risk profile for a merchant category.
    Online retail, travel, and entertainment are high-risk; grocery and pharmacy are low-risk.
    """
    risk_map = {
        "online_retail":   {"risk": "HIGH",   "reason": "CNP (card not present) fraud is common; no physical verification."},
        "travel":          {"risk": "HIGH",   "reason": "Large ticket amounts and international transactions increase exposure."},
        "entertainment":   {"risk": "HIGH",   "reason": "Frequent target for card testing with small charges."},
        "atm":             {"risk": "HIGH",   "reason": "Physical skimming and cash fraud risk."},
        "gas_station":     {"risk": "MEDIUM", "reason": "Skimming devices occasionally installed on pumps."},
        "restaurant":      {"risk": "LOW",    "reason": "Card-present transactions with staff supervision."},
        "grocery":         {"risk": "LOW",    "reason": "High-frequency, low-value, low-risk category."},
        "pharmacy":        {"risk": "LOW",    "reason": "Regulated purchases, low fraud incidence."},
    }
    profile = risk_map.get(
        merchant_category.lower(),
        {"risk": "UNKNOWN", "reason": "Merchant category not in risk database."},
    )
    profile["merchant_category"] = merchant_category
    return json.dumps(profile)


if __name__ == "__main__":
    mcp.run(transport="stdio")
