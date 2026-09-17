"""
detector.py — Main streaming AML detector pipeline.
Processes transactions one-by-one, orchestrating all 3 engines.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import deque
from loguru import logger

from src.pipeline.ingestion import CURRENCY_TO_USD
from src.pipeline.feature_engineering import FEATURE_COLS
from src.models.rule_engine import RuleEngine
from src.models.ensemble import EnsembleDecisionMaker
from src.detection.retroactive_linker import RetroactiveLinker


class StreamingDetector:
    """
    Main entry point for transaction-by-transaction AML detection.

    Processing flow per transaction:
    1. Normalize + feature-extract
    2. Add to graph
    3. Run Rule Engine → rule score + pattern hints
    4. Run XGBoost → ML score
    5. Run GNN (if available) → graph score
    6. Ensemble → final decision
    7. If laundering: pattern classify + retroactive linking
    8. Create alert
    """

    def __init__(self,
                 xgb_model=None,
                 gnn_model=None,
                 pattern_classifier=None,
                 graph=None,
                 whitelist=None,
                 customer_profiler=None,
                 alert_manager=None,
                 scaler=None,
                 rule_weight: float = 0.20,
                 xgb_weight: float = 0.35,
                 gnn_weight: float = 0.45,
                 threshold: float = 0.50,
                 recent_window_size: int = 50000):
        self.xgb_model = xgb_model
        self.gnn_model = gnn_model
        self.pattern_classifier = pattern_classifier
        self.graph = graph
        self.whitelist = whitelist
        self.customer_profiler = customer_profiler
        self.alert_manager = alert_manager
        self.scaler = scaler

        self.rule_engine = RuleEngine()
        self.ensemble = EnsembleDecisionMaker(
            rule_weight=rule_weight,
            xgb_weight=xgb_weight,
            gnn_weight=gnn_weight,
            threshold=threshold,
        )
        self.retroactive_linker = RetroactiveLinker()

        # Rolling window of recent transactions for rule engine
        self._recent_txs = deque(maxlen=recent_window_size)
        self._recent_df_cache: Optional[pd.DataFrame] = None
        self._cache_dirty = True

        self.stats = {
            "processed": 0,
            "alerted": 0,
            "retroactive_flagged": 0,
            "suppressed": 0,
        }

    def _get_recent_df(self) -> pd.DataFrame:
        """Get recent transactions as DataFrame (cached)."""
        if self._cache_dirty or self._recent_df_cache is None:
            if self._recent_txs:
                self._recent_df_cache = pd.DataFrame(list(self._recent_txs))
            else:
                self._recent_df_cache = pd.DataFrame()
            self._cache_dirty = False
        return self._recent_df_cache

    def _extract_features(self, tx: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from a single transaction dict."""
        recent = self._get_recent_df()
        from_acc = str(tx.get("from_account", ""))
        to_acc = str(tx.get("to_account", ""))

        # Rolling account stats (7-day window)
        tx_ts = tx.get("timestamp")
        if isinstance(tx_ts, str):
            tx_ts = pd.to_datetime(tx_ts)
        window_start = tx_ts - timedelta(days=7) if tx_ts else None

        from_txs = recent[recent["from_account"].astype(str) == from_acc] if len(recent) > 0 else pd.DataFrame()
        to_txs = recent[recent["to_account"].astype(str) == to_acc] if len(recent) > 0 else pd.DataFrame()

        if window_start and len(from_txs) > 0:
            from_txs = from_txs[from_txs["timestamp"] >= window_start]
        if window_start and len(to_txs) > 0:
            to_txs = to_txs[to_txs["timestamp"] >= window_start]

        feats = {
            "amount_received_usd": tx.get("amount_received_usd", 0.0),
            "amount_paid_usd": tx.get("amount_paid_usd", 0.0),
            "amount_ratio": tx.get("amount_ratio", 1.0),
            "log_amount_received_usd": np.log1p(tx.get("amount_received_usd", 0.0)),
            "log_amount_paid_usd": np.log1p(tx.get("amount_paid_usd", 0.0)),
            "is_self_transaction": tx.get("is_self_transaction", 0),
            "same_bank": tx.get("same_bank", 0),
            "currency_changed": tx.get("currency_changed", 0),
            "payment_format_enc": tx.get("payment_format_enc", -1),
            "hour_of_day": tx.get("hour_of_day", 0),
            "day_of_week": tx.get("day_of_week", 0),
            "is_weekend": tx.get("is_weekend", 0),
            "from_entity_type_enc": tx.get("from_entity_type_enc", 6),
            "to_entity_type_enc": tx.get("to_entity_type_enc", 6),
            "from_risk_tier_enc": tx.get("from_risk_tier_enc", 1),
            "to_risk_tier_enc": tx.get("to_risk_tier_enc", 1),
            # Rolling
            "from_tx_count_7d": len(from_txs),
            "from_unique_recipients_7d": from_txs["to_account"].nunique() if len(from_txs) > 0 else 0,
            "from_total_sent_7d": from_txs["amount_received_usd"].sum() if len(from_txs) > 0 else 0,
            "from_avg_amount_7d": from_txs["amount_received_usd"].mean() if len(from_txs) > 0 else 0,
            "from_std_amount_7d": from_txs["amount_received_usd"].std() if len(from_txs) > 0 else 0,
            "to_tx_count_7d": len(to_txs),
            "to_unique_senders_7d": to_txs["from_account"].nunique() if len(to_txs) > 0 else 0,
            "to_total_received_7d": to_txs["amount_received_usd"].sum() if len(to_txs) > 0 else 0,
            "to_avg_received_7d": to_txs["amount_received_usd"].mean() if len(to_txs) > 0 else 0,
            # Graph degree
            "from_out_degree": self.graph.node_degree_features(from_acc)["out_degree"] if self.graph else 0,
            "from_in_degree": self.graph.node_degree_features(from_acc)["in_degree"] if self.graph else 0,
            "to_out_degree": self.graph.node_degree_features(to_acc)["out_degree"] if self.graph else 0,
            "to_in_degree": self.graph.node_degree_features(to_acc)["in_degree"] if self.graph else 0,
        }

        x = np.array([feats.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0)
        return x

    async def process_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single incoming transaction.
        Returns result dict (with is_laundering, alert if triggered, etc.)
        """
        self.stats["processed"] += 1

        # Add to graph
        tx_id = None
        if self.graph:
            tx_id = self.graph.add_transaction(tx)
            tx["tx_id"] = tx_id

        # Get recent transactions DF for rule engine
        recent_df = self._get_recent_df()

        # 1. Rule Engine
        rule_result = self.rule_engine.evaluate(tx, self.graph, recent_df)

        # 2. XGBoost
        xgb_score = 0.5
        shap_vals = {}
        if self.xgb_model:
            x = self._extract_features(tx)
            x_scaled = self.scaler.transform(x.reshape(1, -1)) if self.scaler else x.reshape(1, -1)
            _, xgb_score = self.xgb_model.predict_single(x_scaled[0])
            shap_vals = self.xgb_model.explain_single(x_scaled[0])

        # 3. GNN
        gnn_score = 0.5
        if self.gnn_model:
            gnn_score = self.gnn_model.predict_transaction_score(
                str(tx.get("from_account", "")),
                str(tx.get("to_account", "")),
            )

        # 4. Customer profile check
        whitelist_suppressed = False
        customer_risk_modifier = 1.0
        if self.whitelist:
            whitelist_suppressed = await self.whitelist.is_suppressed(
                str(tx.get("from_account", "")),
                str(tx.get("to_account", "")),
            )
        if self.customer_profiler:
            modifier = self.customer_profiler.get_risk_modifier(tx)
            customer_risk_modifier = modifier

        # 5. Ensemble decision
        decision = self.ensemble.decide(
            rule_result=rule_result,
            xgb_score=xgb_score,
            gnn_score=gnn_score,
            whitelist_suppressed=whitelist_suppressed,
            customer_risk_modifier=customer_risk_modifier,
        )

        # 6. Pattern classification (if laundering detected)
        if decision["is_laundering"] and self.pattern_classifier and self.xgb_model:
            x = self._extract_features(tx)
            rule_conf = rule_result.get("confidence_details", {})
            pattern_feats = self.pattern_classifier.build_features(x, rule_conf)
            pattern_result = self.pattern_classifier.predict(pattern_feats)
            if pattern_result["probability"] > 0.4:
                decision["detected_pattern"] = pattern_result["pattern"]

        # Build full result
        result = {
            **tx,
            **decision,
            "tx_id": tx_id,
            "shap_values": shap_vals,
            "triggered_patterns": rule_result.get("triggered_patterns", []),
            "involved_accounts": decision.get("involved_accounts", []),
        }

        # 7. Add to recent buffer
        self._recent_txs.append(tx)
        self._cache_dirty = True

        # 8. Alert + retroactive linking if laundering
        if decision["is_laundering"] and self.alert_manager:
            if not whitelist_suppressed:
                alert = await self.alert_manager.create_alert(result)
                if alert:
                    self.stats["alerted"] += 1
                    self.retroactive_linker.register_alerted(tx_id)

                    # Retroactive linking
                    retro_alerts = self.retroactive_linker.link_and_flag(
                        confirmed_tx=result,
                        pattern=decision["detected_pattern"] or "UNKNOWN",
                        graph=self.graph,
                        transaction_store=recent_df,
                    )
                    for retro in retro_alerts:
                        await self.alert_manager.create_alert(retro)
                        self.stats["retroactive_flagged"] += 1
            else:
                self.stats["suppressed"] += 1

        return result

    async def process_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Process a batch of transactions."""
        results = []
        for tx in transactions:
            r = await self.process_transaction(tx)
            results.append(r)
        return results

    def update_ensemble_weights(self, rule_w: float, xgb_w: float, gnn_w: float):
        self.ensemble.update_weights(rule_w, xgb_w, gnn_w)

    def update_threshold(self, threshold: float):
        self.ensemble.update_threshold(threshold)
