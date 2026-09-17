"""
retroactive_linker.py — When a laundering pattern is confirmed, backtrack
through the transaction graph to retroactively flag all connected transactions.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any, Set, Optional
from loguru import logger


# Per-pattern retroactive linking configuration
PATTERN_BFS_CONFIG = {
    "FAN-OUT": {"depth": 2, "direction": "both", "window_days": 7},
    "FAN-IN": {"depth": 2, "direction": "both", "window_days": 7},
    "CYCLE": {"depth": 10, "direction": "both", "window_days": 14},
    "STACK": {"depth": 4, "direction": "out", "window_days": 7},
    "RANDOM": {"depth": 8, "direction": "out", "window_days": 10},
    "BIPARTITE": {"depth": 3, "direction": "both", "window_days": 7},
    "GATHER-SCATTER": {"depth": 3, "direction": "both", "window_days": 10},
    "SCATTER-GATHER": {"depth": 3, "direction": "both", "window_days": 10},
    "UNKNOWN": {"depth": 2, "direction": "both", "window_days": 7},
}


class RetroactiveLinker:
    """
    When transaction N confirms a laundering pattern, retroactively:
    1. Traverse graph from involved accounts up to pattern-specific depth
    2. Collect all transactions in the traversal within the time window
    3. Return list of previously-undetected tx_ids to retroactively flag

    This solves the "FAN-OUT first-6-look-legit" problem:
    - Txs 1–6 seen individually → below threshold → marked CLEAN
    - Tx 7 crosses threshold → LAUNDERING detected
    - Retroactive linker BFS from source → finds txs 1–6 → retroactively flags all
    """

    def __init__(self, alert_registry: Dict[str, Any] = None):
        # Keep track of already-alerted tx_ids to avoid double-alerting
        self.alerted_tx_ids: Set[str] = set()
        self.alert_registry = alert_registry or {}

    def link_and_flag(self,
                      confirmed_tx: Dict[str, Any],
                      pattern: str,
                      graph,
                      transaction_store: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Given a confirmed laundering transaction, find all related prior transactions
        and return them as retroactive alerts.

        Args:
            confirmed_tx: The triggering transaction dict
            pattern: Detected laundering pattern
            graph: TransactionGraph instance
            transaction_store: DataFrame of all seen transactions

        Returns:
            List of retroactive alert dicts for previously-seen transactions
        """
        config = PATTERN_BFS_CONFIG.get(pattern, PATTERN_BFS_CONFIG["UNKNOWN"])
        depth = config["depth"]
        direction = config["direction"]
        window_days = config["window_days"]

        ts = confirmed_tx.get("timestamp")
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        window_start = ts - timedelta(days=window_days)

        from_acc = str(confirmed_tx.get("from_account", ""))
        to_acc = str(confirmed_tx.get("to_account", ""))
        involved_accounts = confirmed_tx.get("involved_accounts", [from_acc, to_acc])
        if not involved_accounts:
            involved_accounts = [from_acc, to_acc]

        logger.info(f"RetroactiveLinker: pattern={pattern}, depth={depth}, accounts={involved_accounts[:5]}")

        # BFS from each involved account
        linked_tx_ids: Set[str] = set()
        for seed_acc in involved_accounts:
            sub = graph.bfs_subgraph(
                seed_account=str(seed_acc),
                depth=depth,
                direction=direction,
                since=window_start
            )
            tx_ids = graph.get_tx_ids_in_subgraph(sub, since=window_start)
            linked_tx_ids.update(tx_ids)

        # Remove already-alerted tx_ids
        new_tx_ids = linked_tx_ids - self.alerted_tx_ids

        if not new_tx_ids:
            logger.info("No new retroactive transactions found.")
            return []

        logger.info(f"Retroactively flagging {len(new_tx_ids)} prior transactions.")

        # Build retroactive alerts
        retro_alerts = []
        for tx_id in new_tx_ids:
            tx_data = graph.edge_index.get(tx_id)
            if tx_data is None:
                continue
            retro_alert = {
                "tx_id": tx_id,
                "is_laundering": True,
                "retroactive": True,
                "triggering_tx_id": confirmed_tx.get("tx_id"),
                "pattern": pattern,
                "severity": "HIGH",
                "final_score": 0.85,  # conservative retroactive score
                "retroactive_reason": f"Linked via {pattern} pattern — triggered by tx {confirmed_tx.get('tx_id')}",
                **{k: tx_data.get(k) for k in [
                    "timestamp", "from_account", "to_account",
                    "amount_received_usd", "payment_format",
                    "from_bank", "to_bank",
                ]},
            }
            retro_alerts.append(retro_alert)
            self.alerted_tx_ids.add(tx_id)

        return retro_alerts

    def register_alerted(self, tx_id: str) -> None:
        """Mark a transaction as already alerted."""
        self.alerted_tx_ids.add(tx_id)

    def is_alerted(self, tx_id: str) -> bool:
        """Check if a transaction was already alerted."""
        return tx_id in self.alerted_tx_ids
