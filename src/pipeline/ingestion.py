"""
ingestion.py — Data loading, currency normalization, and account enrichment.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Approximate USD conversion rates (as of dataset period)
CURRENCY_TO_USD = {
    "US Dollar": 1.0,
    "Euro": 1.08,
    "UK Pound": 1.27,
    "Swiss Franc": 1.12,
    "Australian Dollar": 0.65,
    "Canadian Dollar": 0.74,
    "Yuan": 0.14,
    "Yen": 0.0067,
    "Rupee": 0.012,
    "Ruble": 0.011,
    "Saudi Riyal": 0.267,
    "Shekel": 0.27,
    "Mexican Peso": 0.058,
    "Bitcoin": 26000.0,  # approximate Sept 2022
    "Turkish Lira": 0.054,
    "Brazilian Real": 0.19,
}

ENTITY_RISK_TIER = {
    "Individual": "HIGH",
    "Sole Proprietorship": "MEDIUM",
    "Partnership": "MEDIUM",
    "Corporation": "LOW",
    "Country": "LOW",
    "Direct": "MEDIUM",
}


def load_transactions(filepath: str, nrows: int = None) -> pd.DataFrame:
    """Load and normalize transaction data."""
    logger.info(f"Loading transactions from {filepath} ...")
    df = pd.read_csv(filepath, nrows=nrows)
    df.columns = [
        "timestamp", "from_bank", "from_account",
        "to_bank", "to_account",
        "amount_received", "receiving_currency",
        "amount_paid", "payment_currency",
        "payment_format", "is_laundering"
    ]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")

    # Normalize amounts to USD — vectorized map (100x faster than apply)
    recv_rates = df["receiving_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    pay_rates = df["payment_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    df["amount_received_usd"] = df["amount_received"] * recv_rates
    df["amount_paid_usd"] = df["amount_paid"] * pay_rates

    # Derived booleans
    df["is_self_transaction"] = (df["from_account"] == df["to_account"]).astype(int)
    df["same_bank"] = (df["from_bank"] == df["to_bank"]).astype(int)
    df["currency_changed"] = (df["receiving_currency"] != df["payment_currency"]).astype(int)
    df["amount_ratio"] = df["amount_received_usd"] / (df["amount_paid_usd"] + 1e-9)

    # Temporal features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Encode payment format
    payment_fmt_map = {
        "ACH": 0, "Cheque": 1, "Credit Card": 2, "Cash": 3,
        "Reinvestment": 4, "Wire": 5, "Bitcoin": 6
    }
    df["payment_format_enc"] = df["payment_format"].map(payment_fmt_map).fillna(-1).astype(int)

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} transactions. Laundering rate: {df['is_laundering'].mean()*100:.3f}%")
    return df


def load_accounts(filepath: str) -> pd.DataFrame:
    """Load account data and derive entity type features."""
    logger.info(f"Loading accounts from {filepath} ...")
    df = pd.read_csv(filepath)
    df.columns = ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"]

    # Extract entity type
    import re
    df["entity_type"] = df["entity_name"].str.extract(r"^([A-Za-z ]+)#").iloc[:, 0].str.strip()
    df["entity_type"] = df["entity_type"].fillna("Unknown")

    # Risk tier
    df["risk_tier"] = df["entity_type"].map(ENTITY_RISK_TIER).fillna("MEDIUM")

    # Entity type encoding
    entity_type_map = {
        "Individual": 0, "Sole Proprietorship": 1, "Partnership": 2,
        "Corporation": 3, "Country": 4, "Direct": 5, "Unknown": 6
    }
    df["entity_type_enc"] = df["entity_type"].map(entity_type_map).fillna(6).astype(int)

    logger.info(f"Loaded {len(df):,} accounts. Entity types: {df['entity_type'].value_counts().to_dict()}")
    return df


def enrich_transactions(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Join account info onto transactions for both sender and receiver."""
    logger.info("Enriching transactions with account data ...")

    acct_cols = ["account_number", "entity_type", "entity_type_enc", "risk_tier", "bank_name", "entity_id"]
    acct = accounts[acct_cols].drop_duplicates("account_number")

    # Enrich sender
    df = transactions.merge(
        acct.rename(columns={c: f"from_{c}" for c in acct_cols if c != "account_number"}),
        left_on="from_account", right_on="account_number", how="left"
    ).drop(columns=["account_number"], errors="ignore")

    # Enrich receiver
    df = df.merge(
        acct.rename(columns={c: f"to_{c}" for c in acct_cols if c != "account_number"}),
        left_on="to_account", right_on="account_number", how="left"
    ).drop(columns=["account_number"], errors="ignore")

    # Fill missing
    for col in ["from_entity_type_enc", "to_entity_type_enc"]:
        df[col] = df[col].fillna(6)
    for col in ["from_risk_tier", "to_risk_tier"]:
        df[col] = df[col].fillna("MEDIUM")

    risk_tier_enc = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df["from_risk_tier_enc"] = df["from_risk_tier"].map(risk_tier_enc).fillna(1)
    df["to_risk_tier_enc"] = df["to_risk_tier"].map(risk_tier_enc).fillna(1)

    logger.info("Enrichment complete.")
    return df


def load_all_data(data_dir: str, nrows: int = None) -> tuple:
    """Convenience: load and enrich everything."""
    base = Path(data_dir)
    transactions = load_transactions(str(base / "HI-Small_Trans.csv"), nrows=nrows)
    accounts = load_accounts(str(base / "HI-Small_accounts.csv"))
    enriched = enrich_transactions(transactions, accounts)
    return enriched, accounts
