"""
gnn_model.py — Graph Transformer for AML detection using PyTorch Geometric.
Uses a Graph Transformer with multi-head attention on edge features.
Trained with Focal Loss to handle extreme class imbalance.
CUDA-accelerated.
"""
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple, List
import joblib

try:
    from torch_geometric.data import Data, DataLoader, HeteroData
    from torch_geometric.nn import (
        TransformerConv, GATConv, global_mean_pool, global_max_pool
    )
    from torch_geometric.utils import add_self_loops, degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("PyTorch Geometric not installed. GNN model will be disabled.")


# ------------------------------------------------------------------ #
#  Focal Loss — better than BCE for extreme imbalance                 #
# ------------------------------------------------------------------ #
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(probs, self.alpha),
                              torch.full_like(probs, 1 - self.alpha))
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        loss = (focal_weight * bce).mean()
        return loss


# ------------------------------------------------------------------ #
#  Graph Transformer Model                                            #
# ------------------------------------------------------------------ #
class GraphTransformerAML(nn.Module):
    """
    Multi-layer Graph Transformer for node-level laundering classification.
    - Node features: account-level behavioral features + entity type embedding
    - Edge features: amount, currency, payment format, time delta
    """

    def __init__(self,
                 node_feat_dim: int = 16,
                 edge_feat_dim: int = 8,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("PyTorch Geometric is required for GNN model.")

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=hidden_dim,
                dropout=dropout,
                concat=True,
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Node-level classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # binary logit
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x                     # [N, node_feat_dim]
        edge_index = data.edge_index   # [2, E]
        edge_attr = data.edge_attr     # [E, edge_feat_dim]

        # Initial embeddings
        x = F.relu(self.node_embed(x))
        edge_emb = F.relu(self.edge_embed(edge_attr))

        # Graph Transformer layers with residual connections
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x_new = conv(x, edge_index, edge_attr=edge_emb)
            x = norm(x + self.dropout(x_new))

        # Node classification
        logits = self.classifier(x).squeeze(-1)  # [N]
        return logits


# ------------------------------------------------------------------ #
#  GNN Trainer + Inference Wrapper                                    #
# ------------------------------------------------------------------ #
class GNNAMLModel:
    """
    Wraps the GraphTransformerAML for training and inference.
    """

    NODE_FEAT_COLS = [
        "entity_type_enc", "risk_tier_enc",
        "tx_count_7d", "total_sent_7d", "avg_amount_7d", "std_amount_7d",
        "out_degree", "in_degree",
        "unique_recipients_7d", "unique_senders_7d",
        "hour_of_day", "day_of_week", "is_weekend",
        "is_self_transaction", "same_bank", "payment_format_enc"
    ]
    EDGE_FEAT_COLS = [
        "log_amount_received_usd", "log_amount_paid_usd",
        "amount_ratio", "currency_changed",
        "payment_format_enc", "hour_of_day", "is_weekend", "same_bank"
    ]

    def __init__(self, model_path: Optional[str] = None,
                 hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"GNN using device: {self.device}")
        self.threshold = 0.5
        self.model: Optional[GraphTransformerAML] = None
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def build_pyg_graph(self, df: pd.DataFrame, account_index: dict) -> "Data":
        """
        Build a PyG Data object from a transaction DataFrame.
        """
        import pandas as pd

        # Node features — one per unique account
        accounts = list(account_index.keys())
        n_nodes = len(accounts)

        # Build node feature matrix
        node_feats = np.zeros((n_nodes, len(self.NODE_FEAT_COLS)), dtype=np.float32)
        # Aggregate account-level features from transactions
        from_agg = df.groupby("from_account").agg({
            "from_entity_type_enc": "first",
            "from_risk_tier_enc": "first",
            "from_tx_count_7d": "last",
            "from_total_sent_7d": "last",
            "from_avg_amount_7d": "last",
            "from_std_amount_7d": "last",
            "from_out_degree": "last",
            "from_in_degree": "last",
            "from_unique_recipients_7d": "last",
            "hour_of_day": "mean",
            "day_of_week": "mean",
            "is_weekend": "mean",
            "is_self_transaction": "mean",
            "same_bank": "mean",
            "payment_format_enc": "first",
        }).reset_index()

        for _, row in from_agg.iterrows():
            acc = str(row["from_account"])
            if acc in account_index:
                idx = account_index[acc]
                node_feats[idx] = [
                    row.get("from_entity_type_enc", 6),
                    row.get("from_risk_tier_enc", 1),
                    row.get("from_tx_count_7d", 0),
                    row.get("from_total_sent_7d", 0),
                    row.get("from_avg_amount_7d", 0),
                    row.get("from_std_amount_7d", 0),
                    row.get("from_out_degree", 0),
                    row.get("from_in_degree", 0),
                    row.get("from_unique_recipients_7d", 0),
                    row.get("to_unique_senders_7d", 0) if "to_unique_senders_7d" in row else 0,
                    row.get("hour_of_day", 0),
                    row.get("day_of_week", 0),
                    row.get("is_weekend", 0),
                    row.get("is_self_transaction", 0),
                    row.get("same_bank", 0),
                    row.get("payment_format_enc", 0),
                ]

        # Build edge index and features
        src_ids, dst_ids, edge_attrs, edge_labels = [], [], [], []
        for _, row in df.iterrows():
            fa = str(row["from_account"])
            ta = str(row["to_account"])
            if fa in account_index and ta in account_index:
                src_ids.append(account_index[fa])
                dst_ids.append(account_index[ta])
                edge_attrs.append([
                    float(row.get("log_amount_received_usd", 0)),
                    float(row.get("log_amount_paid_usd", 0)),
                    float(row.get("amount_ratio", 1)),
                    float(row.get("currency_changed", 0)),
                    float(row.get("payment_format_enc", 0)),
                    float(row.get("hour_of_day", 0)),
                    float(row.get("is_weekend", 0)),
                    float(row.get("same_bank", 0)),
                ])
                edge_labels.append(int(row.get("is_laundering", 0)))

        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        # Node labels: node is laundering if any connected tx is laundering
        node_labels = np.zeros(n_nodes, dtype=np.float32)
        for i, (s, d) in enumerate(zip(src_ids, dst_ids)):
            if edge_labels[i] == 1:
                node_labels[s] = 1.0
                node_labels[d] = 1.0

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(node_labels, dtype=torch.float32),
        )
        return data

    def train(self, train_df, val_df, epochs: int = 30,
              lr: float = 1e-3, batch_size: int = 1024) -> None:
        """Train the GNN on a transaction DataFrame."""
        if not HAS_PYG:
            logger.error("PyG not available. Cannot train GNN.")
            return

        import pandas as pd
        logger.info("Building account index ...")
        all_accounts = list(set(
            train_df["from_account"].astype(str).tolist() +
            train_df["to_account"].astype(str).tolist()
        ))
        account_index = {acc: idx for idx, acc in enumerate(all_accounts)}

        logger.info(f"Building PyG graph — {len(all_accounts):,} nodes ...")
        train_data = self.build_pyg_graph(train_df, account_index)
        train_data = train_data.to(self.device)

        node_feat_dim = train_data.x.shape[1]
        edge_feat_dim = train_data.edge_attr.shape[1]

        self.model = GraphTransformerAML(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = FocalLoss(alpha=0.9, gamma=2.0)

        best_val_loss = float("inf")
        patience = 7
        patience_counter = 0

        logger.info(f"Starting GNN training for {epochs} epochs ...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(train_data)
            loss = criterion(logits, train_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(train_data)  # use train graph for now
                    val_loss = criterion(val_logits, train_data.y).item()
                    preds = (torch.sigmoid(val_logits) >= 0.5).float()
                    acc = (preds == train_data.y).float().mean().item()
                logger.info(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        self.account_index = account_index
        logger.info("GNN training complete.")

    def predict_proba_for_accounts(self, accounts: List[str], graph_data) -> dict:
        """Predict laundering probability for a list of accounts."""
        if self.model is None or not HAS_PYG:
            return {acc: 0.5 for acc in accounts}

        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph_data)
            probs = torch.sigmoid(logits).cpu().numpy()

        results = {}
        for acc in accounts:
            if acc in self.account_index:
                idx = self.account_index[acc]
                results[acc] = float(probs[idx])
            else:
                results[acc] = 0.5
        return results

    def predict_transaction_score(self, from_acc: str, to_acc: str,
                                   graph_data=None) -> float:
        """Return GNN score for a transaction (average of node scores)."""
        if not HAS_PYG or self.model is None or graph_data is None:
            return 0.5
        scores = self.predict_proba_for_accounts([from_acc, to_acc], graph_data)
        return float(np.mean(list(scores.values())))

    def save(self, path: str) -> None:
        """Save GNN model weights and account index."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "model_config": {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            },
            "account_index": getattr(self, "account_index", {}),
            "threshold": self.threshold,
        }, path)
        logger.info(f"GNN model saved to {path}")

    def load(self, path: str) -> None:
        """Load GNN model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        cfg = checkpoint["model_config"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.account_index = checkpoint.get("account_index", {})
        self.threshold = checkpoint.get("threshold", 0.5)

        if checkpoint["model_state"] is not None:
            # Need to infer dims from saved state — use defaults
            self.model = GraphTransformerAML(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        logger.info(f"GNN model loaded from {path}")



