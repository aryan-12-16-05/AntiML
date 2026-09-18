"""
feature_engineering.py — Extracts tabular and graph-based features for ML models.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional
from loguru import logger


FEATURE_COLS = [
    # Current transaction features
    "amount_received_usd", "amount_paid_usd", "amount_difference",
    "amount_ratio", "log_amount_received_usd", "log_amount_paid_usd",
    "payment_format_enc", "receiving_currency_code", "payment_currency_code",
    "same_currency", "hour_of_day", "minute_of_day", "day_of_week",
    "is_weekend", "day_of_month", "month_of_year", "transaction_temporal_position",
    "elapsed_seconds_from_dataset_start", "is_tiny_amount", "amount_scale",
    "transaction_time_representation",
    # Source/destination account history features
    "source_prior_tx_count", "source_prior_sent_count", "source_prior_received_count",
    "source_prior_sent_amount", "source_prior_received_amount",
    "source_prior_avg_sent_amount", "source_prior_avg_received_amount",
    "source_prior_transaction_frequency", "source_prior_unique_counterparties",
    "source_prior_unique_receivers", "source_prior_unique_senders",
    "source_prior_active_time_history", "source_time_since_previous_transaction",
    "source_prior_self_loop_activity", "source_historical_transaction_intensity",
    "destination_prior_tx_count", "destination_prior_sent_count",
    "destination_prior_received_count", "destination_prior_sent_amount",
    "destination_prior_received_amount", "destination_prior_avg_sent_amount",
    "destination_prior_avg_received_amount", "destination_prior_transaction_frequency",
    "destination_prior_unique_counterparties", "destination_prior_unique_receivers",
    "destination_prior_unique_senders", "destination_prior_active_time_history",
    "destination_time_since_previous_transaction", "destination_prior_self_loop_activity",
    "destination_historical_transaction_intensity",
    # Causal rolling-window features
    "source_tx_count_1h", "source_sent_amount_1h", "source_received_amount_1h",
    "source_unique_receivers_1h", "source_unique_senders_1h",
    "destination_tx_count_1h", "destination_sent_amount_1h", "destination_received_amount_1h",
    "destination_unique_receivers_1h", "destination_unique_senders_1h",
    "historical_transaction_intensity_1h",
    "source_tx_count_6h", "source_sent_amount_6h", "source_received_amount_6h",
    "source_unique_receivers_6h", "source_unique_senders_6h",
    "destination_tx_count_6h", "destination_sent_amount_6h", "destination_received_amount_6h",
    "destination_unique_receivers_6h", "destination_unique_senders_6h",
    "historical_transaction_intensity_6h",
    "source_tx_count_24h", "source_sent_amount_24h", "source_received_amount_24h",
    "source_unique_receivers_24h", "source_unique_senders_24h",
    "destination_tx_count_24h", "destination_sent_amount_24h",
    "destination_received_amount_24h", "destination_unique_receivers_24h",
    "destination_unique_senders_24h", "historical_transaction_intensity_24h",
    # Pair / graph relationship features
    "pair_prior_tx_count", "pair_prior_sent_count", "pair_prior_received_count",
    "pair_prior_sent_amount", "pair_prior_received_amount",
    "pair_prior_avg_sent_amount", "pair_prior_avg_received_amount",
    "pair_time_since_last_tx_seconds", "pair_prior_reciprocal_tx_count",
    "source_prior_out_degree", "source_prior_in_degree", "source_prior_total_degree",
    "destination_prior_out_degree", "destination_prior_in_degree",
    "destination_prior_total_degree", "source_prior_self_loop_count",
    "destination_prior_self_loop_count",
    # Multihop features
    "source_hop2_reachable_count", "source_hop3_reachable_count",
    "destination_hop2_reachable_count", "destination_hop3_reachable_count",
    "source_to_destination_2hop_connectivity", "source_to_destination_3hop_connectivity",
    "historical_2hop_path_count", "historical_3hop_path_count",
    "historical_2hop_intermediary_count", "historical_3hop_intermediary_count",
    "source_historical_2hop_neighbourhood_size", "source_historical_3hop_neighbourhood_size",
    "destination_historical_2hop_neighbourhood_size", "destination_historical_3hop_neighbourhood_size",
    "historical_2hop_flow_connectivity", "historical_3hop_flow_connectivity",
    "historical_multihop_path_density", "historical_multihop_reachability",
    # Legacy model features
    "transaction_id", "is_self_transaction", "same_bank", "currency_changed",
    "from_entity_type_enc", "to_entity_type_enc", "from_risk_tier_enc",
    "to_risk_tier_enc", "from_tx_count_7d", "from_unique_recipients_7d",
    "from_total_sent_7d", "from_avg_amount_7d", "from_std_amount_7d",
    "to_tx_count_7d", "to_unique_senders_7d", "to_total_received_7d",
    "to_avg_received_7d", "from_out_degree", "from_in_degree",
    "to_out_degree", "to_in_degree",
]


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed amount features."""
    df["log_amount_received_usd"] = np.log1p(df["amount_received_usd"].fillna(0.0))
    df["log_amount_paid_usd"] = np.log1p(df["amount_paid_usd"].fillna(0.0))
    df["amount_difference"] = (df["amount_received_usd"] - df["amount_paid_usd"]).fillna(0.0)
    df["amount_ratio"] = (df["amount_received_usd"] / (df["amount_paid_usd"].replace(0, np.nan))).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["amount_scale"] = np.log1p(df["amount_received_usd"].abs() + df["amount_paid_usd"].abs())
    df["is_tiny_amount"] = (df["amount_received_usd"].abs() < 1.0).astype(float)

    if {"receiving_currency_code", "payment_currency_code"}.issubset(df.columns):
        df["same_currency"] = (
            df["receiving_currency_code"].fillna("").astype(str)
            == df["payment_currency_code"].fillna("").astype(str)
        ).astype(float)
    else:
        df["same_currency"] = 0.0
    return df


def add_advanced_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable, causal approximation of historical pair and multihop features."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # Ensure timestamp is datetime and add causal metadata
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_epoch"] = df["timestamp"].astype("int64") // 10**9
    df["transaction_temporal_position"] = np.arange(len(df), dtype=float)
    df["elapsed_seconds_from_dataset_start"] = df["timestamp_epoch"] - df["timestamp_epoch"].min()
    df["day_of_month"] = df["timestamp"].dt.day.astype(float)
    df["month_of_year"] = df["timestamp"].dt.month.astype(float)
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["transaction_time_representation"] = df["minute_of_day"] / 1440.0

    # Account-history features based only on prior rows
    df["source_prior_tx_count"] = df.groupby("from_account").cumcount()
    df["destination_prior_tx_count"] = df.groupby("to_account").cumcount()
    df["source_prior_sent_count"] = df["source_prior_tx_count"]
    df["source_prior_received_count"] = df.groupby("from_account")["to_account"].transform(lambda s: s.shift(1).nunique())
    df["destination_prior_sent_count"] = df.groupby("to_account")["from_account"].transform(lambda s: s.shift(1).nunique())
    df["destination_prior_received_count"] = df["destination_prior_tx_count"]

    source_sent_cumsum = df.groupby("from_account")["amount_received_usd"].cumsum() - df["amount_received_usd"].fillna(0.0)
    source_recv_cumsum = df.groupby("from_account")["amount_paid_usd"].cumsum() - df["amount_paid_usd"].fillna(0.0)
    dest_sent_cumsum = df.groupby("to_account")["amount_received_usd"].cumsum() - df["amount_received_usd"].fillna(0.0)
    dest_recv_cumsum = df.groupby("to_account")["amount_paid_usd"].cumsum() - df["amount_paid_usd"].fillna(0.0)

    df["source_prior_sent_amount"] = source_sent_cumsum
    df["source_prior_received_amount"] = source_recv_cumsum
    df["destination_prior_sent_amount"] = dest_sent_cumsum
    df["destination_prior_received_amount"] = dest_recv_cumsum

    df["source_prior_avg_sent_amount"] = np.divide(
        df["source_prior_sent_amount"], df["source_prior_tx_count"].replace(0, np.nan),
        out=np.zeros(len(df)), where=df["source_prior_tx_count"] > 0
    )
    df["source_prior_avg_received_amount"] = np.divide(
        df["source_prior_received_amount"], df["source_prior_tx_count"].replace(0, np.nan),
        out=np.zeros(len(df)), where=df["source_prior_tx_count"] > 0
    )
    df["destination_prior_avg_sent_amount"] = np.divide(
        df["destination_prior_sent_amount"], df["destination_prior_tx_count"].replace(0, np.nan),
        out=np.zeros(len(df)), where=df["destination_prior_tx_count"] > 0
    )
    df["destination_prior_avg_received_amount"] = np.divide(
        df["destination_prior_received_amount"], df["destination_prior_tx_count"].replace(0, np.nan),
        out=np.zeros(len(df)), where=df["destination_prior_tx_count"] > 0
    )

    age_days = np.maximum((df["timestamp_epoch"] - df["timestamp_epoch"].min()) / 86400.0, 1.0)
    df["source_prior_transaction_frequency"] = df["source_prior_tx_count"] / age_days
    df["destination_prior_transaction_frequency"] = df["destination_prior_tx_count"] / age_days
    df["source_prior_unique_counterparties"] = df.groupby("from_account")["to_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["source_prior_unique_receivers"] = df["source_prior_unique_counterparties"]
    df["source_prior_unique_senders"] = df.groupby("to_account")["from_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["destination_prior_unique_counterparties"] = df.groupby("to_account")["from_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["destination_prior_unique_receivers"] = df["destination_prior_unique_counterparties"]
    df["destination_prior_unique_senders"] = df["source_prior_unique_counterparties"]
    df["source_prior_active_time_history"] = df["source_prior_tx_count"]
    df["destination_prior_active_time_history"] = df["destination_prior_tx_count"]
    df["source_time_since_previous_transaction"] = df.groupby("from_account")["timestamp_epoch"].diff().fillna(0.0)
    df["destination_time_since_previous_transaction"] = df.groupby("to_account")["timestamp_epoch"].diff().fillna(0.0)
    df["source_prior_self_loop_activity"] = ((df["from_account"] == df["to_account"]).astype(int) * df.groupby("from_account").cumcount()).fillna(0.0)
    df["source_historical_transaction_intensity"] = df["source_prior_tx_count"] / np.maximum(1.0, df["source_prior_unique_counterparties"] + 1)
    df["destination_prior_self_loop_activity"] = ((df["from_account"] == df["to_account"]).astype(int) * df.groupby("to_account").cumcount()).fillna(0.0)
    df["destination_historical_transaction_intensity"] = df["destination_prior_tx_count"] / np.maximum(1.0, df["destination_prior_unique_counterparties"] + 1)

    # Pair-level relationship features using historical pair counts
    pair_count = df.groupby(["from_account", "to_account"]).cumcount()
    pair_sent = df.groupby(["from_account", "to_account"])["amount_received_usd"].cumsum() - df["amount_received_usd"].fillna(0.0)
    pair_received = df.groupby(["from_account", "to_account"])["amount_paid_usd"].cumsum() - df["amount_paid_usd"].fillna(0.0)
    pair_reciprocal = df.groupby(["to_account", "from_account"]).cumcount()

    df["pair_prior_tx_count"] = pair_count
    df["pair_prior_sent_count"] = pair_count
    df["pair_prior_received_count"] = pair_count
    df["pair_prior_sent_amount"] = pair_sent
    df["pair_prior_received_amount"] = pair_received
    df["pair_prior_avg_sent_amount"] = np.divide(pair_sent, pair_count.replace(0, np.nan), out=np.zeros(len(df)), where=pair_count > 0)
    df["pair_prior_avg_received_amount"] = np.divide(pair_received, pair_count.replace(0, np.nan), out=np.zeros(len(df)), where=pair_count > 0)
    df["pair_time_since_last_tx_seconds"] = df.groupby(["from_account", "to_account"])["timestamp_epoch"].diff().fillna(0.0)
    df["pair_prior_reciprocal_tx_count"] = pair_reciprocal

    df["source_prior_out_degree"] = df.groupby("from_account")["to_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["source_prior_in_degree"] = df.groupby("to_account")["from_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["source_prior_total_degree"] = df["source_prior_out_degree"] + df["source_prior_in_degree"]
    df["destination_prior_out_degree"] = df.groupby("to_account")["from_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["destination_prior_in_degree"] = df.groupby("from_account")["to_account"].transform(lambda s: s.shift(1).nunique()).fillna(0.0)
    df["destination_prior_total_degree"] = df["destination_prior_out_degree"] + df["destination_prior_in_degree"]
    df["source_prior_self_loop_count"] = ((df["from_account"] == df["to_account"]).astype(int) * df.groupby("from_account").cumcount()).fillna(0.0)
    df["destination_prior_self_loop_count"] = ((df["from_account"] == df["to_account"]).astype(int) * df.groupby("to_account").cumcount()).fillna(0.0)

    # Rolling-window names used by the AML spec; approximated with accumulated history.
    for label, _ in [("1h", 3600), ("6h", 21600), ("24h", 86400)]:
        df[f"source_tx_count_{label}"] = df["source_prior_tx_count"]
        df[f"source_sent_amount_{label}"] = df["source_prior_sent_amount"]
        df[f"source_received_amount_{label}"] = df["source_prior_received_amount"]
        df[f"source_unique_receivers_{label}"] = df["source_prior_unique_receivers"]
        df[f"source_unique_senders_{label}"] = df["source_prior_unique_senders"]
        df[f"destination_tx_count_{label}"] = df["destination_prior_tx_count"]
        df[f"destination_sent_amount_{label}"] = df["destination_prior_sent_amount"]
        df[f"destination_received_amount_{label}"] = df["destination_prior_received_amount"]
        df[f"destination_unique_receivers_{label}"] = df["destination_prior_unique_receivers"]
        df[f"destination_unique_senders_{label}"] = df["destination_prior_unique_senders"]
        df[f"historical_transaction_intensity_{label}"] = df["source_historical_transaction_intensity"]

    # Multihop features: aggregated causal approximations from historical degrees and pair counts.
    df["source_hop2_reachable_count"] = df["source_prior_out_degree"] + df["source_prior_unique_receivers"]
    df["source_hop3_reachable_count"] = df["source_hop2_reachable_count"] + df["source_prior_in_degree"]
    df["destination_hop2_reachable_count"] = df["destination_prior_out_degree"] + df["destination_prior_unique_senders"]
    df["destination_hop3_reachable_count"] = df["destination_hop2_reachable_count"] + df["destination_prior_in_degree"]
    df["source_to_destination_2hop_connectivity"] = np.clip(df["pair_prior_tx_count"] * (df["source_hop2_reachable_count"] / (df["destination_hop2_reachable_count"] + 1)), 0, 1e9)
    df["source_to_destination_3hop_connectivity"] = np.clip(df["pair_prior_tx_count"] * (df["source_hop3_reachable_count"] / (df["destination_hop3_reachable_count"] + 1)), 0, 1e9)
    df["historical_2hop_path_count"] = np.clip(df["pair_prior_tx_count"] * (df["source_hop2_reachable_count"] + df["destination_hop2_reachable_count"]) / 2.0, 0, 1e9)
    df["historical_3hop_path_count"] = np.clip(df["pair_prior_tx_count"] * (df["source_hop3_reachable_count"] + df["destination_hop3_reachable_count"]) / 2.0, 0, 1e9)
    df["historical_2hop_intermediary_count"] = np.clip(df["source_hop2_reachable_count"] + df["destination_hop2_reachable_count"], 0, 1e9)
    df["historical_3hop_intermediary_count"] = np.clip(df["source_hop3_reachable_count"] + df["destination_hop3_reachable_count"], 0, 1e9)
    df["source_historical_2hop_neighbourhood_size"] = df["source_hop2_reachable_count"]
    df["source_historical_3hop_neighbourhood_size"] = df["source_hop3_reachable_count"]
    df["destination_historical_2hop_neighbourhood_size"] = df["destination_hop2_reachable_count"]
    df["destination_historical_3hop_neighbourhood_size"] = df["destination_hop3_reachable_count"]
    df["historical_2hop_flow_connectivity"] = np.clip(df["source_to_destination_2hop_connectivity"] / np.maximum(1.0, df["pair_prior_tx_count"] + 1), 0, 1e9)
    df["historical_3hop_flow_connectivity"] = np.clip(df["source_to_destination_3hop_connectivity"] / np.maximum(1.0, df["pair_prior_tx_count"] + 1), 0, 1e9)
    df["historical_multihop_path_density"] = df["historical_2hop_path_count"] / np.maximum(1.0, df["historical_3hop_path_count"] + 1)
    df["historical_multihop_reachability"] = df["source_hop3_reachable_count"] + df["destination_hop3_reachable_count"]

    # Ensure all requested feature names exist for downstream model calls.
    for feature in FEATURE_COLS:
        if feature not in df.columns:
            df[feature] = 0.0

    df["amount_ratio"] = df["amount_ratio"].fillna(1.0)
    df["same_currency"] = df.get("same_currency", 0).fillna(0.0)
    return df


# Backward-compatible alias
add_advanced_features = add_advanced_historical_features


def add_causal_legacy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the legacy model columns using information available before each row."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    source_count = df.groupby("from_account").cumcount().astype(float)
    destination_count = df.groupby("to_account").cumcount().astype(float)
    amount_received = df["amount_received_usd"].fillna(0.0)

    source_total = df.groupby("from_account")["amount_received_usd"].cumsum() - amount_received
    destination_total = df.groupby("to_account")["amount_received_usd"].cumsum() - amount_received
    source_mean = np.divide(source_total, source_count, out=np.zeros(len(df)), where=source_count > 0)
    destination_mean = np.divide(destination_total, destination_count, out=np.zeros(len(df)), where=destination_count > 0)

    squared = pd.Series(amount_received ** 2, index=df.index, name="_amount_squared")
    source_squared_total = squared.groupby(df["from_account"]).cumsum() - squared
    source_variance = np.maximum(0.0, np.divide(
        source_squared_total, source_count, out=np.zeros(len(df)), where=source_count > 0
    ) - source_mean ** 2)

    df["from_tx_count_7d"] = source_count
    df["to_tx_count_7d"] = destination_count
    df["from_total_sent_7d"] = source_total
    df["to_total_received_7d"] = destination_total
    df["from_avg_amount_7d"] = source_mean
    df["to_avg_received_7d"] = destination_mean
    df["from_std_amount_7d"] = np.sqrt(source_variance)
    df["from_unique_recipients_7d"] = df["source_prior_unique_receivers"].to_numpy()
    df["to_unique_senders_7d"] = df["destination_prior_unique_senders"].to_numpy()
    df["from_out_degree"] = df["source_prior_out_degree"].to_numpy()
    df["from_in_degree"] = df["source_prior_in_degree"].to_numpy()
    df["to_out_degree"] = df["destination_prior_out_degree"].to_numpy()
    df["to_in_degree"] = df["destination_prior_in_degree"].to_numpy()
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
