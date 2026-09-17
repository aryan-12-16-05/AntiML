"""
profiler.py — Customer behavioral profiling for AML false positive suppression.
Builds behavioral baselines per account and entity type to reduce noise.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger
import joblib
from pathlib import Path


# Entity types that naturally receive many payments — lower risk modifier
HIGH_VOLUME_ENTITY_TYPES = {
    "Country": 0.3,           # sovereign entities
    "Corporation": 0.7,       # large corps receive many payments
    "Partnership": 0.85,
    "Sole Proprietorship": 0.95,
    "Individual": 1.1,        # individuals → slightly elevated
    "Direct": 1.0,
    "Unknown": 1.2,
}

# Keywords in entity names that reduce risk (educational, government, utilities)
BENIGN_ENTITY_KEYWORDS = [
    "university", "college", "school", "education",
    "government", "ministry", "municipal", "federal",
    "hospital", "clinic", "health", "medical",
    "utility", "water", "electric", "power",
    "insurance", "pension", "fund",
]


class CustomerProfiler:
    """
    Builds and maintains behavioral profiles for accounts.
    Used to:
    1. Compute risk modifier (0.3 = very low risk, 2.0 = very high risk)
    2. Provide profile context to dashboard (typical spend, frequency, etc.)
    3. Detect anomalies vs. account's own baseline
    """

    def __init__(self):
        self.profiles: Dict[str, Dict] = {}
        self._entity_name_lookup: Dict[str, str] = {}

    def build_profiles(self, df: pd.DataFrame, accounts_df: pd.DataFrame) -> None:
        """
        Build behavioral profiles from historical transaction data.
        """
        logger.info("Building customer profiles ...")

        # Join entity names
        name_map = accounts_df.set_index("account_number")["entity_name"].to_dict()
        self._entity_name_lookup = {str(k): str(v) for k, v in name_map.items()}

        # Aggregate per account (sender perspective)
        sender_stats = df.groupby("from_account").agg(
            tx_count=("amount_received_usd", "count"),
            total_sent=("amount_received_usd", "sum"),
            avg_sent=("amount_received_usd", "mean"),
            std_sent=("amount_received_usd", "std"),
            max_sent=("amount_received_usd", "max"),
            unique_recipients=("to_account", "nunique"),
            unique_banks=("to_bank", "nunique"),
            laundering_rate=("is_laundering", "mean"),
        ).reset_index()

        # Aggregate per account (receiver perspective)
        recv_stats = df.groupby("to_account").agg(
            recv_count=("amount_received_usd", "count"),
            total_received=("amount_received_usd", "sum"),
            avg_received=("amount_received_usd", "mean"),
            unique_senders=("from_account", "nunique"),
        ).reset_index()

        # Build profiles
        for _, row in sender_stats.iterrows():
            acc = str(row["from_account"])
            entity_name = self._entity_name_lookup.get(acc, "")
            self.profiles[acc] = {
                "account": acc,
                "entity_name": entity_name,
                "entity_type": df[df["from_account"].astype(str) == acc]["from_entity_type"].iloc[0]
                    if "from_entity_type" in df.columns and len(df[df["from_account"].astype(str) == acc]) > 0
                    else "Unknown",
                "tx_count": int(row["tx_count"]),
                "total_sent_usd": float(row["total_sent"]),
                "avg_sent_usd": float(row["avg_sent"]),
                "std_sent_usd": float(row.get("std_sent", 0) or 0),
                "max_sent_usd": float(row["max_sent"]),
                "unique_recipients": int(row["unique_recipients"]),
                "unique_banks": int(row["unique_banks"]),
                "historical_laundering_rate": float(row["laundering_rate"]),
                "is_benign_keyword": any(kw in entity_name.lower() for kw in BENIGN_ENTITY_KEYWORDS),
            }

        for _, row in recv_stats.iterrows():
            acc = str(row["to_account"])
            if acc not in self.profiles:
                self.profiles[acc] = {"account": acc, "entity_name": "", "entity_type": "Unknown"}
            self.profiles[acc].update({
                "recv_count": int(row["recv_count"]),
                "total_received_usd": float(row["total_received"]),
                "avg_received_usd": float(row["avg_received"]),
                "unique_senders": int(row["unique_senders"]),
            })

        logger.info(f"Built profiles for {len(self.profiles):,} accounts.")

    def get_profile(self, account: str) -> Dict:
        """Get profile for a single account."""
        return self.profiles.get(str(account), {
            "account": account,
            "entity_name": "Unknown",
            "entity_type": "Unknown",
        })

    def get_risk_modifier(self, tx: Dict[str, Any]) -> float:
        """
        Compute risk modifier for a transaction based on customer profiles.
        < 1.0 = lower risk (suppress false positives)
        > 1.0 = higher risk (boost detection sensitivity)
        """
        from_acc = str(tx.get("from_account", ""))
        to_acc = str(tx.get("to_account", ""))

        from_profile = self.profiles.get(from_acc, {})
        to_profile = self.profiles.get(to_acc, {})

        from_entity = from_profile.get("entity_type", "Unknown")
        to_entity = to_profile.get("entity_type", "Unknown")

        # Base modifier from entity types
        from_modifier = HIGH_VOLUME_ENTITY_TYPES.get(from_entity, 1.0)
        to_modifier = HIGH_VOLUME_ENTITY_TYPES.get(to_entity, 1.0)

        # Benign keyword suppression
        if from_profile.get("is_benign_keyword", False):
            from_modifier *= 0.5
        if to_profile.get("is_benign_keyword", False):
            to_modifier *= 0.5

        # Transaction amount vs. historical baseline anomaly
        amount = tx.get("amount_received_usd", 0.0)
        avg = from_profile.get("avg_sent_usd", 0.0)
        std = from_profile.get("std_sent_usd", 1.0)
        if avg > 0 and std > 0:
            z_score = abs(amount - avg) / (std + 1e-9)
            if z_score > 3:  # major deviation from normal
                from_modifier *= min(1.5, 1.0 + z_score * 0.1)

        # Combined modifier (average of sender and receiver)
        modifier = (from_modifier + to_modifier) / 2.0
        return float(np.clip(modifier, 0.2, 2.0))

    def get_profile_summary(self, account: str) -> Dict:
        """Rich profile summary for dashboard."""
        profile = self.get_profile(account)
        risk_tier = profile.get("entity_type", "Unknown")
        return {
            **profile,
            "risk_modifier": HIGH_VOLUME_ENTITY_TYPES.get(risk_tier, 1.0),
            "display_name": profile.get("entity_name", account[:8] + "..."),
        }

    def save(self, path: str) -> None:
        joblib.dump({
            "profiles": self.profiles,
            "entity_name_lookup": self._entity_name_lookup,
        }, path)
        logger.info(f"Profiles saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.profiles = data["profiles"]
        self._entity_name_lookup = data["entity_name_lookup"]
        logger.info(f"Profiles loaded from {path} ({len(self.profiles):,} accounts)")
