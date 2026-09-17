"""
rule_engine.py — Rule-based money laundering detector.
Implements heuristic rules for all 8 laundering pattern types.
"""
from datetime import timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger


class RuleEngine:
    """
    Applies domain-expert rules for each laundering pattern type.
    Operates on the transaction graph and recent transaction history.
    """

    def __init__(self,
                 fan_out_threshold: int = 3,
                 fan_in_threshold: int = 3,
                 cycle_max_hops: int = 10,
                 chain_min_hops: int = 3,
                 time_window_hours: int = 168,  # 7 days
                 amount_similarity_tolerance: float = 0.15):
        self.fan_out_threshold = fan_out_threshold
        self.fan_in_threshold = fan_in_threshold
        self.cycle_max_hops = cycle_max_hops
        self.chain_min_hops = chain_min_hops
        self.time_window = timedelta(hours=time_window_hours)
        self.amount_tol = amount_similarity_tolerance

    def evaluate(self, tx: Dict[str, Any], graph, recent_txs: pd.DataFrame) -> Dict:
        """
        Evaluate all rules against the incoming transaction.
        Returns confidence score [0,1] and triggered patterns.
        """
        results = {
            "rule_score": 0.0,
            "triggered_patterns": [],
            "involved_accounts": set(),
            "confidence_details": {}
        }

        from_acc = str(tx.get("from_account", ""))
        to_acc = str(tx.get("to_account", ""))
        ts = tx.get("timestamp", pd.Timestamp.now())
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        window_start = ts - self.time_window

        # Filter recent transactions to time window
        if len(recent_txs) > 0:
            recent = recent_txs[recent_txs["timestamp"] >= window_start]
        else:
            recent = pd.DataFrame()

        # Run all rule checks
        fan_out = self._check_fan_out(from_acc, to_acc, ts, window_start, graph, recent)
        fan_in = self._check_fan_in(from_acc, to_acc, ts, window_start, graph, recent)
        cycle = self._check_cycle(from_acc, ts, window_start, graph)
        stack = self._check_stack(from_acc, to_acc, ts, window_start, recent)
        random_walk = self._check_random_walk(from_acc, to_acc, ts, window_start, graph, recent)
        bipartite = self._check_bipartite(from_acc, to_acc, ts, window_start, recent)
        gather_scatter = self._check_gather_scatter(from_acc, to_acc, ts, window_start, graph, recent)
        scatter_gather = self._check_scatter_gather(from_acc, to_acc, ts, window_start, graph, recent)

        all_checks = {
            "FAN-OUT": fan_out,
            "FAN-IN": fan_in,
            "CYCLE": cycle,
            "STACK": stack,
            "RANDOM": random_walk,
            "BIPARTITE": bipartite,
            "GATHER-SCATTER": gather_scatter,
            "SCATTER-GATHER": scatter_gather,
        }

        max_conf = 0.0
        for pattern, (triggered, conf, accounts) in all_checks.items():
            if triggered:
                results["triggered_patterns"].append(pattern)
                results["involved_accounts"].update(accounts)
                max_conf = max(max_conf, conf)
                results["confidence_details"][pattern] = conf

        results["rule_score"] = max_conf
        results["involved_accounts"] = list(results["involved_accounts"])
        return results

    # ------------------------------------------------------------------ #
    #  Individual pattern checks                                           #
    # ------------------------------------------------------------------ #

    def _check_fan_out(self, from_acc, to_acc, ts, window_start, graph, recent):
        """One account sends to N+ distinct accounts in time window."""
        if len(recent) == 0:
            return False, 0.0, []
        sender_txs = recent[recent["from_account"].astype(str) == from_acc]
        unique_recipients = sender_txs["to_account"].nunique()
        if unique_recipients >= self.fan_out_threshold:
            conf = min(1.0, unique_recipients / (self.fan_out_threshold * 2))
            accounts = [from_acc] + list(sender_txs["to_account"].astype(str).unique())
            return True, conf, accounts
        return False, 0.0, []

    def _check_fan_in(self, from_acc, to_acc, ts, window_start, graph, recent):
        """Many distinct accounts send to same destination in time window."""
        if len(recent) == 0:
            return False, 0.0, []
        recv_txs = recent[recent["to_account"].astype(str) == to_acc]
        unique_senders = recv_txs["from_account"].nunique()
        if unique_senders >= self.fan_in_threshold:
            conf = min(1.0, unique_senders / (self.fan_in_threshold * 2))
            accounts = [to_acc] + list(recv_txs["from_account"].astype(str).unique())
            return True, conf, accounts
        return False, 0.0, []

    def _check_cycle(self, from_acc, ts, window_start, graph):
        """Detect if account is part of a directed cycle."""
        has_cycle = graph.detect_cycle(from_acc, max_hops=self.cycle_max_hops, since=window_start)
        if has_cycle:
            return True, 0.85, [from_acc]
        return False, 0.0, []

    def _check_stack(self, from_acc, to_acc, ts, window_start, recent):
        """Multiple parallel 2-hop chains with amount preservation."""
        if len(recent) == 0:
            return False, 0.0, []
        # Look for chains: from_acc → to_acc → X with similar amounts
        forwarded = recent[recent["from_account"].astype(str) == to_acc]
        if len(forwarded) == 0:
            return False, 0.0, []
        # Check amount similarity in the chain
        in_txs = recent[recent["to_account"].astype(str) == to_acc]
        if len(in_txs) == 0:
            return False, 0.0, []

        avg_in = in_txs["amount_received_usd"].mean()
        avg_out = forwarded["amount_received_usd"].mean()
        ratio = abs(avg_in - avg_out) / (avg_in + 1e-9)
        if ratio < self.amount_tol and len(forwarded) >= 1:
            conf = 0.6 + 0.2 * (1 - ratio)
            accounts = (list(in_txs["from_account"].astype(str).unique()) +
                       [to_acc] +
                       list(forwarded["to_account"].astype(str).unique()))
            return True, conf, accounts
        return False, 0.0, []

    def _check_random_walk(self, from_acc, to_acc, ts, window_start, graph, recent):
        """Multi-hop chain where each hop goes to a new account."""
        if len(recent) == 0:
            return False, 0.0, []
        # Check if to_acc then forwards to a new account quickly
        forwarded = recent[
            (recent["from_account"].astype(str) == to_acc) &
            (recent["to_account"].astype(str) != from_acc)
        ]
        if len(forwarded) >= self.chain_min_hops - 1:
            conf = min(0.75, 0.4 + 0.1 * len(forwarded))
            accounts = [from_acc, to_acc] + list(forwarded["to_account"].astype(str).unique())
            return True, conf, accounts
        return False, 0.0, []

    def _check_bipartite(self, from_acc, to_acc, ts, window_start, recent):
        """
        Bipartite: two groups of accounts exchange in structured pattern.
        Detect when same set of accounts repeatedly exchange with another set.
        """
        if len(recent) == 0:
            return False, 0.0, []
        # Group A sends to group B (both sets of size >=2)
        out_from = recent[recent["from_account"].astype(str) == from_acc]["to_account"].unique()
        in_to = recent[recent["to_account"].astype(str) == to_acc]["from_account"].unique()
        # Cross-check: does from_acc receive from group B senders?
        recv_accs = recent[recent["to_account"].astype(str) == from_acc]["from_account"].unique()
        overlap = set(out_from) & set(recv_accs)
        if len(overlap) >= 2:
            conf = min(0.8, 0.4 + 0.1 * len(overlap))
            accounts = [from_acc, to_acc] + list(overlap)
            return True, conf, accounts
        return False, 0.0, []

    def _check_gather_scatter(self, from_acc, to_acc, ts, window_start, graph, recent):
        """Fan-in to central node then fan-out from same node."""
        if len(recent) == 0:
            return False, 0.0, []
        # from_acc receives from multiple then sends to multiple
        received_from = recent[recent["to_account"].astype(str) == from_acc]["from_account"].nunique()
        sent_to = recent[recent["from_account"].astype(str) == from_acc]["to_account"].nunique()
        if received_from >= self.fan_in_threshold and sent_to >= self.fan_out_threshold:
            conf = min(0.9, 0.5 + 0.05 * (received_from + sent_to))
            accounts = [from_acc] + list(
                recent[recent["to_account"].astype(str) == from_acc]["from_account"].astype(str).unique()
            ) + list(
                recent[recent["from_account"].astype(str) == from_acc]["to_account"].astype(str).unique()
            )
            return True, conf, accounts
        return False, 0.0, []

    def _check_scatter_gather(self, from_acc, to_acc, ts, window_start, graph, recent):
        """Fan-out first, then all recipients send to a single collector."""
        if len(recent) == 0:
            return False, 0.0, []
        # to_acc sends to multiple
        scattered = recent[recent["from_account"].astype(str) == to_acc]["to_account"].unique()
        if len(scattered) < self.fan_out_threshold:
            return False, 0.0, []
        # Check if those accounts then send to a common destination
        all_downstream = recent[recent["from_account"].astype(str).isin([str(a) for a in scattered])]
        if len(all_downstream) == 0:
            return False, 0.0, []
        common_dest = all_downstream["to_account"].value_counts()
        if len(common_dest) > 0 and common_dest.iloc[0] >= max(2, len(scattered) // 2):
            conf = min(0.85, 0.55 + 0.05 * len(scattered))
            accounts = [from_acc, to_acc] + [str(a) for a in scattered] + [str(common_dest.index[0])]
            return True, conf, accounts
        return False, 0.0, []
