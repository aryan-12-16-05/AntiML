"""
xgboost_model.py — XGBoost-based AML classifier.
Trained with SMOTE-balanced data, tuned threshold, and SHAP explainability.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                              f1_score, classification_report, confusion_matrix)
import shap
import joblib
from pathlib import Path
from loguru import logger
from typing import Optional, Tuple


class XGBoostAMLModel:
    """
    XGBoost classifier for transaction-level AML detection.
    Supports training, prediction, threshold tuning, and SHAP explanations.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[xgb.XGBClassifier] = None
        self.threshold: float = 0.5
        self.feature_names = None
        self.explainer: Optional[shap.TreeExplainer] = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def build_model(self, scale_pos_weight: float = 50.0) -> xgb.XGBClassifier:
        """Construct XGBoost model with CUDA acceleration and imbalance handling."""
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,  # handles class imbalance
            eval_metric=["auc", "aucpr"],
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",         # fast histogram-based (works on CPU + GPU)
            device="cuda" if self._cuda_available() else "cpu",
        )
        return self.model

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              scale_pos_weight: float = 50.0,
              feature_names=None) -> None:
        """Train the XGBoost model with early stopping."""
        logger.info(f"Training XGBoost — Train size: {len(X_train):,} | Positive rate: {y_train.mean():.3%}")
        self.feature_names = feature_names
        self.build_model(scale_pos_weight=scale_pos_weight)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        logger.info(f"Best iteration: {self.model.best_iteration}")

        # Build SHAP explainer
        logger.info("Building SHAP TreeExplainer ...")
        self.explainer = shap.TreeExplainer(self.model)

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                       min_recall: float = 0.75) -> float:
        """
        Tune decision threshold to maximize F1 at minimum recall >= min_recall.
        """
        probs = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, probs)

        best_f1, best_thresh = 0.0, 0.5
        for p, r, t in zip(precision, recall, thresholds):
            if r < min_recall:
                continue
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        self.threshold = float(best_thresh)
        logger.info(f"Optimal threshold: {self.threshold:.4f} (F1={best_f1:.4f} @ recall≥{min_recall})")
        return self.threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of laundering for each sample."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction using tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> Tuple[int, float]:
        """Predict single transaction. Returns (label, score)."""
        x = x.reshape(1, -1)
        score = float(self.predict_proba(x)[0])
        label = int(score >= self.threshold)
        return label, score

    def explain(self, X: np.ndarray, top_n: int = 10) -> dict:
        """
        Get SHAP explanation for given samples.
        Returns feature importances as dict.
        """
        if self.explainer is None:
            return {}
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        feat_names = self.feature_names or [f"f{i}" for i in range(len(mean_abs))]
        importance = dict(sorted(
            zip(feat_names, mean_abs.tolist()),
            key=lambda x: x[1], reverse=True
        )[:top_n])
        return importance

    def explain_single(self, x: np.ndarray) -> dict:
        """SHAP explanation for a single transaction."""
        if self.explainer is None:
            return {}
        x = x.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        feat_names = self.feature_names or [f"f{i}" for i in range(x.shape[1])]
        return {
            name: float(val)
            for name, val in zip(feat_names, shap_vals[0])
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Full evaluation report."""
        probs = self.predict_proba(X_test)
        preds = (probs >= self.threshold).astype(int)

        roc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        logger.info(f"XGBoost — ROC-AUC: {roc:.4f} | F1: {f1:.4f}")
        logger.info(f"\n{classification_report(y_test, preds, zero_division=0)}")

        return {
            "roc_auc": roc, "f1": f1,
            "confusion_matrix": cm.tolist(),
            "report": report,
            "threshold": self.threshold,
        }

    def save(self, path: str) -> None:
        """Save model and metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }
        joblib.dump(data, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.threshold = data["threshold"]
        self.feature_names = data.get("feature_names")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info(f"XGBoost model loaded from {path}")
