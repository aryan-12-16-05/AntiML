"""
graph_builder.py — Builds and maintains a directed transaction graph using NetworkX.
Each node = account, each edge = transaction with attributes.
"""
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional, List, Dict, Any


class TransactionGraph:
    """
    Directed multigraph of financial transactions.
    Supports incremental addition of transactions and time-windowed queries.
    """

    def __init__(self, window_days: int = 30):
        self.G = nx.MultiDiGraph()
        self.window_days = window_days
        self.edge_index: Dict[str, Dict] = {}  # tx_id -> edge metadata
        self._tx_counter = 0

    def add_transaction(self, tx: Dict[str, Any]) -> str:
        """Add a single transaction to the graph. Returns unique tx_id."""
        tx_id = f"tx_{self._tx_counter}"
        self._tx_counter += 1

        from_acc = str(tx["from_account"])
        to_acc = str(tx["to_account"])

        # Add nodes with account metadata if not present
        if not self.G.has_node(from_acc):
            self.G.add_node(from_acc,
                entity_type=tx.get("from_entity_type", "Unknown"),
                entity_type_enc=tx.get("from_entity_type_enc", 6),
                risk_tier=tx.get("from_risk_tier", "MEDIUM"),
                bank=tx.get("from_bank", ""),
            )
        if not self.G.has_node(to_acc):
            self.G.add_node(to_acc,
                entity_type=tx.get("to_entity_type", "Unknown"),
                entity_type_enc=tx.get("to_entity_type_enc", 6),
                risk_tier=tx.get("to_risk_tier", "MEDIUM"),
                bank=tx.get("to_bank", ""),
            )

        # Add edge
        self.G.add_edge(
            from_acc, to_acc,
            tx_id=tx_id,
            timestamp=tx["timestamp"],
            amount_usd=tx.get("amount_received_usd", 0.0),
            payment_format=tx.get("payment_format", ""),
            currency=tx.get("receiving_currency", "US Dollar"),
            is_laundering=tx.get("is_laundering", 0),
        )
        self.edge_index[tx_id] = {
            "from_account": from_acc,
            "to_account": to_acc,
            **tx,
        }
        return tx_id

    def add_transactions_bulk(self, df: pd.DataFrame) -> List[str]:
        """Bulk add from DataFrame. Returns list of tx_ids."""
        tx_ids = []
        for _, row in df.iterrows():
            tx_ids.append(self.add_transaction(row.to_dict()))
        return tx_ids

    # ------------------------------------------------------------------ #
    #  Graph query helpers                                                 #
    # ------------------------------------------------------------------ #

    def get_outgoing_neighbors(self, account: str, since: Optional[datetime] = None,
                               until: Optional[datetime] = None) -> List[str]:
        """All accounts that received money from `account` in time window."""
        neighbors = []
        if not self.G.has_node(account):
            return neighbors
        for _, dst, data in self.G.out_edges(account, data=True):
            ts = data["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            neighbors.append(dst)
        return list(set(neighbors))

    def get_incoming_neighbors(self, account: str, since: Optional[datetime] = None,
                               until: Optional[datetime] = None) -> List[str]:
        """All accounts that sent money to `account` in time window."""
        neighbors = []
        if not self.G.has_node(account):
            return neighbors
        for src, _, data in self.G.in_edges(account, data=True):
            ts = data["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            neighbors.append(src)
        return list(set(neighbors))

    def get_transactions_between(self, from_acc: str, to_acc: str,
                                  since: Optional[datetime] = None) -> List[Dict]:
        """All transactions between two accounts."""
        txs = []
        if not self.G.has_edge(from_acc, to_acc):
            return txs
        for _, _, data in self.G.edges(from_acc, data=True):
            if data.get("timestamp") and since:
                ts = data["timestamp"]
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                if ts < since:
                    continue
            txs.append(data)
        return txs

    def bfs_subgraph(self, seed_account: str, depth: int = 3,
                     direction: str = "both",
                     since: Optional[datetime] = None) -> nx.MultiDiGraph:
        """
        BFS from seed_account up to `depth` hops.
        direction: 'out' | 'in' | 'both'
        """
        visited = set()
        queue = [(seed_account, 0)]
        subgraph_nodes = set()

        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)
            subgraph_nodes.add(node)

            if direction in ("out", "both"):
                for nbr in self.get_outgoing_neighbors(node, since=since):
                    if nbr not in visited:
                        queue.append((nbr, d + 1))
            if direction in ("in", "both"):
                for nbr in self.get_incoming_neighbors(node, since=since):
                    if nbr not in visited:
                        queue.append((nbr, d + 1))

        return self.G.subgraph(subgraph_nodes).copy()

    def get_tx_ids_in_subgraph(self, subgraph: nx.MultiDiGraph,
                                since: Optional[datetime] = None) -> List[str]:
        """Get all tx_ids for edges in the given subgraph."""
        tx_ids = []
        for u, v, data in subgraph.edges(data=True):
            ts = data.get("timestamp")
            if since and ts:
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                if ts < since:
                    continue
            tx_id = data.get("tx_id")
            if tx_id:
                tx_ids.append(tx_id)
        return tx_ids

    def node_degree_features(self, account: str, since: Optional[datetime] = None) -> Dict:
        """Compute degree-based node features for ML."""
        out_nbrs = self.get_outgoing_neighbors(account, since=since)
        in_nbrs = self.get_incoming_neighbors(account, since=since)
        return {
            "out_degree": len(out_nbrs),
            "in_degree": len(in_nbrs),
            "total_degree": len(set(out_nbrs + in_nbrs)),
            "unique_out_neighbors": len(set(out_nbrs)),
            "unique_in_neighbors": len(set(in_nbrs)),
        }

    def detect_cycle(self, account: str, max_hops: int = 10,
                     since: Optional[datetime] = None) -> bool:
        """Check if account participates in a directed cycle within max_hops."""
        try:
            sub = self.bfs_subgraph(account, depth=max_hops, direction="out", since=since)
            if not self.G.has_node(account):
                return False
            # Account is in cycle if there's a path back to itself in subgraph
            return nx.has_path(sub, account, account) if account in sub else False
        except Exception:
            return False

    @property
    def num_nodes(self):
        return self.G.number_of_nodes()

    @property
    def num_edges(self):
        return self.G.number_of_edges()
