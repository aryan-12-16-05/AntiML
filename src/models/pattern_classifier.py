"""
pattern_classifier.py — 8-way laundering pattern type classifier.
Trained as a multi-class classifier on confirmed laundering transactions.
Uses XGBoost for tabular features + rule engine pattern probabilities.
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb


PATTERN_TYPES = [
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "STACK",
    "RANDOM",
    "BIPARTITE",
    "GATHER-SCATTER",
    "SCATTER-GATHER",
    "UNKNOWN",
]


class PatternClassifier:
    """
    Multi-class classifier for identifying which laundering pattern a transaction belongs to.
    Only applied AFTER a transaction is flagged as laundering.

    Features used:
    - All tabular features (same as XGBoost)
    - Rule engine confidence scores per pattern (8 features)
    - Graph structural features
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[xgb.XGBClassifier] = None
        self.le = LabelEncoder()
        self.le.classes_ = np.array(PATTERN_TYPES)
        self.feature_names = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def build_features(self, tabular_features: np.ndarray,
                       rule_confidence: Dict[str, float]) -> np.ndarray:
        """
        Concatenate tabular features with rule confidence scores.
        rule_confidence: dict of {pattern_name: confidence_score}
        """
        rule_feats = np.array([
            rule_confidence.get(p, 0.0) for p in PATTERN_TYPES[:-1]  # exclude UNKNOWN
        ], dtype=np.float32)
        return np.concatenate([tabular_features.flatten(), rule_feats])

    def train(self, X: np.ndarray, y_patterns: List[str],
              feature_names=None) -> None:
        """
        Train on laundering-only transactions.
        X: feature matrix for laundering rows
        y_patterns: list of pattern names (from patterns file)
        """
        logger.info(f"Training Pattern Classifier on {len(X)} laundering samples ...")

        # Filter to known patterns
        valid_mask = [p in PATTERN_TYPES for p in y_patterns]
        X = X[valid_mask]
        y = [p for p, v in zip(y_patterns, valid_mask) if v]

        # Fit encoder ONLY on classes present in data (not all 8 patterns)
        # This ensures y_enc is in [0..n_classes-1] as XGBoost requires
        present_classes = sorted(set(y))
        self.le.fit(present_classes)
        y_enc = self.le.transform(y)
        self.feature_names = feature_names

        from collections import Counter
        logger.info(f"Pattern distribution: {Counter(y)}")
        logger.info(f"Classes encoded: {list(zip(self.le.classes_, range(len(self.le.classes_))))}")

        # Auto-detect GPU
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"

        n_classes = len(present_classes)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        params = dict(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            device=_device, tree_method="hist", random_state=42, n_jobs=-1,
            objective=objective,
        )
        if n_classes > 2:
            params["num_class"] = n_classes

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y_enc)

        preds = self.model.predict(X)
        logger.info(f"\n{classification_report(y_enc, preds, target_names=list(self.le.classes_), zero_division=0)}")


    def predict(self, x: np.ndarray) -> Dict:
        """
        Predict pattern type for a single flagged transaction.
        Returns: {pattern, probability, all_probs}
        """
        if self.model is None:
            return {"pattern": "UNKNOWN", "probability": 0.0, "all_probs": {}}

        x = x.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        pattern = self.le.inverse_transform([pred_idx])[0]

        all_probs = {
            self.le.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "pattern": pattern,
            "probability": float(probs[pred_idx]),
            "all_probs": all_probs,
        }

    def save(self, path: str) -> None:
        data = {
            "model": self.model,
            "le_classes": self.le.classes_,
            "feature_names": self.feature_names,
        }
        joblib.dump(data, path)
        logger.info(f"Pattern classifier saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.le.classes_ = data["le_classes"]
        self.feature_names = data.get("feature_names")
        logger.info(f"Pattern classifier loaded from {path}")
