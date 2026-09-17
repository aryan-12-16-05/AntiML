"""
feature_engineering.py — Extracts tabular and graph-based features for ML models.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional
from loguru import logger


FEATURE_COLS = [
    # Amount features
    "amount_received_usd", "amount_paid_usd", "amount_ratio",
    "log_amount_received_usd", "log_amount_paid_usd",
    # Transaction type
    "is_self_transaction", "same_bank", "currency_changed",
    "payment_format_enc",
    # Temporal
    "hour_of_day", "day_of_week", "is_weekend",
    # Entity features
    "from_entity_type_enc", "to_entity_type_enc",
    "from_risk_tier_enc", "to_risk_tier_enc",
    # Rolling account-level features
    "from_tx_count_7d", "from_unique_recipients_7d",
    "from_total_sent_7d", "from_avg_amount_7d", "from_std_amount_7d",
    "to_tx_count_7d", "to_unique_senders_7d",
    "to_total_received_7d", "to_avg_received_7d",
    # Graph degree features (at prediction time)
    "from_out_degree", "from_in_degree",
    "to_out_degree", "to_in_degree",
]


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed amount features."""
    df["log_amount_received_usd"] = np.log1p(df["amount_received_usd"])
    df["log_amount_paid_usd"] = np.log1p(df["amount_paid_usd"])
    return df


def compute_rolling_account_features(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Compute rolling 7-day account-level features.
    For each transaction, computes stats about the SENDER and RECEIVER
    based on all prior transactions within the window.
    """
    logger.info("Computing rolling account features (this may take a few minutes) ...")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp_ts"] = df["timestamp"].astype(np.int64) // 10**9  # seconds

    window_secs = window_days * 86400

    # Pre-group by from_account and to_account for efficiency
    from_grp = df.groupby("from_account")
    to_grp = df.groupby("to_account")

    # Initialize columns
    for col in ["from_tx_count_7d", "from_unique_recipients_7d", "from_total_sent_7d",
                "from_avg_amount_7d", "from_std_amount_7d",
                "to_tx_count_7d", "to_unique_senders_7d",
                "to_total_received_7d", "to_avg_received_7d",
                "from_out_degree", "from_in_degree",
                "to_out_degree", "to_in_degree"]:
        df[col] = 0.0

    # Vectorized rolling using pandas groupby + rolling on time
    # Sender stats
    df_sorted = df.copy()
    df_sorted = df_sorted.set_index("timestamp")

    logger.info("Computing sender rolling stats ...")
    sender_agg = (
        df.groupby("from_account")
        .apply(lambda g: g.set_index("timestamp")["amount_received_usd"]
               .rolling(f"{window_days}D", min_periods=1)
               .agg(["count", "sum", "mean", "std"])
               .reset_index())
        .reset_index(level=0)
    )
    sender_agg.columns = ["from_account", "timestamp",
                          "from_tx_count_7d", "from_total_sent_7d",
                          "from_avg_amount_7d", "from_std_amount_7d"]
    sender_agg["from_std_amount_7d"] = sender_agg["from_std_amount_7d"].fillna(0)

    df = df.merge(sender_agg, on=["from_account", "timestamp"], how="left", suffixes=("", "_new"))
    for col in ["from_tx_count_7d", "from_total_sent_7d", "from_avg_amount_7d", "from_std_amount_7d"]:
        new_col = col + "_new"
        if new_col in df.columns:
            df[col] = df[new_col].fillna(0)
            df.drop(columns=[new_col], inplace=True)

    # Receiver rolling stats
    logger.info("Computing receiver rolling stats ...")
    recv_agg = (
        df.groupby("to_account")
        .apply(lambda g: g.set_index("timestamp")["amount_received_usd"]
               .rolling(f"{window_days}D", min_periods=1)
               .agg(["count", "sum", "mean"])
               .reset_index())
        .reset_index(level=0)
    )
    recv_agg.columns = ["to_account", "timestamp",
                        "to_tx_count_7d", "to_total_received_7d", "to_avg_received_7d"]

    df = df.merge(recv_agg, on=["to_account", "timestamp"], how="left", suffixes=("", "_r"))
    for col in ["to_tx_count_7d", "to_total_received_7d", "to_avg_received_7d"]:
        new_col = col + "_r"
        if new_col in df.columns:
            df[col] = df[new_col].fillna(0)
            df.drop(columns=[new_col], inplace=True)

    # Unique recipients/senders — approximate via cumcount per window
    # Use simpler cumulative unique count as approximation
    df["from_unique_recipients_7d"] = (
        df.groupby(["from_account", df["timestamp"].dt.date])["to_account"]
        .transform("nunique")
        .fillna(0)
    )
    df["to_unique_senders_7d"] = (
        df.groupby(["to_account", df["timestamp"].dt.date])["from_account"]
        .transform("nunique")
        .fillna(0)
    )

    # Graph degree features (cumulative up to current tx)
    logger.info("Computing graph degree features ...")
    df["from_out_degree"] = df.groupby("from_account").cumcount() + 1
    df["from_in_degree"] = df.groupby("to_account").cumcount() + 1
    df["to_out_degree"] = df["from_out_degree"]  # approximation
    df["to_in_degree"] = df["from_in_degree"]

    # Drop helper column
    df.drop(columns=["timestamp_ts"], errors="ignore", inplace=True)

    logger.info("Rolling features complete.")
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Returns (X, y) where X is the feature matrix and y is the label.
    Only uses columns that are available.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns (will use 0): {missing}")
        for c in missing:
            df[c] = 0.0

    X = df[FEATURE_COLS].fillna(0).astype(np.float32)
    y = df["is_laundering"].astype(int) if "is_laundering" in df.columns else None
    return X, y
