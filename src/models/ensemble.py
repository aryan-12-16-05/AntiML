"""
ensemble.py — Weighted ensemble combiner for 3-model AML decision making.
Combines Rule Engine, XGBoost, and GNN scores into a final verdict.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger


PATTERN_LABEL_MAP = {
    "FAN-OUT": 0,
    "FAN-IN": 1,
    "CYCLE": 2,
    "STACK": 3,
    "RANDOM": 4,
    "BIPARTITE": 5,
    "GATHER-SCATTER": 6,
    "SCATTER-GATHER": 7,
}

PATTERN_LABEL_INV = {v: k for k, v in PATTERN_LABEL_MAP.items()}


class EnsembleDecisionMaker:
    """
    Combines scores from all three detection engines:
      - Rule Engine (heuristic confidence score)
      - XGBoost (ML probability)
      - GNN (graph-level probability)

    Final score = w1*rule + w2*xgb + w3*gnn

    Decision logic:
      1. Any single model detecting with very high confidence overrides.
      2. Otherwise, weighted ensemble must exceed threshold.
      3. Customer whitelist can suppress any alert.
    """

    def __init__(self,
                 rule_weight: float = 0.20,
                 xgb_weight: float = 0.35,
                 gnn_weight: float = 0.45,
                 threshold: float = 0.50,
                 override_threshold: float = 0.92):
        assert abs(rule_weight + xgb_weight + gnn_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.w_rule = rule_weight
        self.w_xgb = xgb_weight
        self.w_gnn = gnn_weight
        self.threshold = threshold
        self.override_threshold = override_threshold

    def decide(self,
               rule_result: Dict[str, Any],
               xgb_score: float,
               gnn_score: float,
               whitelist_suppressed: bool = False,
               customer_risk_modifier: float = 1.0) -> Dict[str, Any]:
        """
        Make final laundering decision.

        Args:
            rule_result: Output from RuleEngine.evaluate()
            xgb_score: XGBoost probability [0,1]
            gnn_score: GNN node score [0,1]
            whitelist_suppressed: Whether account is in approved whitelist
            customer_risk_modifier: Multiplier based on customer profile (0.5=low risk, 2.0=high risk)

        Returns:
            Decision dict with: is_laundering, final_score, pattern, confidence, explanation
        """
        rule_score = float(rule_result.get("rule_score", 0.0))
        patterns = rule_result.get("triggered_patterns", [])
        confidence_details = rule_result.get("confidence_details", {})

        # Weighted ensemble score
        ensemble_score = (
            self.w_rule * rule_score +
            self.w_xgb * xgb_score +
            self.w_gnn * gnn_score
        )

        # Apply customer risk modifier (clamp to [0,1])
        adjusted_score = min(1.0, ensemble_score * customer_risk_modifier)

        # Override: any single model at very high confidence
        any_override = (
            rule_score >= self.override_threshold or
            xgb_score >= self.override_threshold or
            gnn_score >= self.override_threshold
        )

        is_laundering = (adjusted_score >= self.threshold) or any_override

        # Suppress if whitelisted
        if whitelist_suppressed:
            is_laundering = False
            adjusted_score = min(adjusted_score, 0.2)  # dampen score visually

        # Determine best pattern guess
        detected_pattern = patterns[0] if patterns else self._infer_pattern(rule_result, xgb_score, gnn_score)
        severity = self._compute_severity(adjusted_score, is_laundering)

        return {
            "is_laundering": is_laundering,
            "final_score": round(adjusted_score, 4),
            "ensemble_score": round(ensemble_score, 4),
            "detected_pattern": detected_pattern,
            "all_patterns": patterns,
            "severity": severity,
            "component_scores": {
                "rule": round(rule_score, 4),
                "xgboost": round(xgb_score, 4),
                "gnn": round(gnn_score, 4),
            },
            "confidence_details": confidence_details,
            "whitelist_suppressed": whitelist_suppressed,
            "override_triggered": any_override,
            "involved_accounts": rule_result.get("involved_accounts", []),
        }

    def _infer_pattern(self, rule_result: Dict, xgb_score: float, gnn_score: float) -> str:
        """Infer most likely pattern when rule engine didn't trigger."""
        if gnn_score > 0.7:
            return "CYCLE"  # GNN tends to excel at structural patterns
        if xgb_score > 0.7:
            return "RANDOM"  # tabular model good at sequential patterns
        return "UNKNOWN"

    def _compute_severity(self, score: float, is_laundering: bool) -> str:
        """Convert score to severity level."""
        if not is_laundering:
            return "NONE"
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.70:
            return "HIGH"
        elif score >= 0.55:
            return "MEDIUM"
        else:
            return "LOW"

    def update_weights(self, rule_weight: float, xgb_weight: float, gnn_weight: float) -> None:
        """Dynamically update ensemble weights (e.g., from dashboard)."""
        total = rule_weight + xgb_weight + gnn_weight
        self.w_rule = rule_weight / total
        self.w_xgb = xgb_weight / total
        self.w_gnn = gnn_weight / total
        logger.info(f"Updated weights — Rule: {self.w_rule:.2f} | XGB: {self.w_xgb:.2f} | GNN: {self.w_gnn:.2f}")

    def update_threshold(self, threshold: float) -> None:
        """Update decision threshold (tunable from dashboard)."""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Decision threshold updated to {self.threshold:.3f}")
