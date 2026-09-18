"""Evaluate the saved XGBoost and GNN AML models."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.gnn_model import GNNAMLModel
from src.models.xgboost_model import XGBoostAMLModel
from src.pipeline.data_prep import split_data
from src.pipeline.feature_engineering import (
    add_log_features,
    add_advanced_historical_features,
    add_causal_legacy_features,
    get_feature_matrix,
)
from src.pipeline.ingestion import load_all_data


def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the strictly causal feature set used by train.py."""
    return add_causal_legacy_features(add_advanced_historical_features(add_log_features(df)))


def print_metrics(name, y_true, y_pred, y_score, threshold=None):
    """Print binary classification metrics and the confusion matrix."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    print(f"\n{name} evaluation metrics")
    print("=" * (len(name) + 21))
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_true, y_score):.4f}")
    matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive support: {int(np.sum(y_true == 1))}")
    if threshold is not None:
        print(f"Threshold: {threshold:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix [TN, FP; FN, TP]:")
    print(matrix)


def evaluate_xgboost(models_dir: Path, test_df: pd.DataFrame):
    scaler = joblib.load(models_dir / "scaler.pkl")
    model = XGBoostAMLModel(model_path=str(models_dir / "xgboost_aml.pkl"))
    X_test, y_test = get_feature_matrix(test_df)
    probabilities = model.predict_proba(scaler.transform(X_test.values))
    predictions = (probabilities >= model.threshold).astype(int)
    print_metrics("XGBoost", y_test.values, predictions, probabilities, model.threshold)


def evaluate_gnn(models_dir: Path, train_df: pd.DataFrame, gnn_rows: int):
    model = GNNAMLModel(model_path=str(models_dir / "gnn_aml.pt"))
    graph_df = train_df.head(min(gnn_rows, len(train_df)))
    graph = model.build_pyg_graph(graph_df, model.account_index).to(model.device)
    model.model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model.model(graph)).cpu().numpy()
    labels = graph.y.cpu().numpy().astype(int)
    predictions = (probabilities >= model.threshold).astype(int)
    print_metrics("GNN (training graph)", labels, predictions, probabilities, model.threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--gnn-rows", type=int, default=500_000)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    print("Loading and preparing evaluation data ...", flush=True)
    df, _ = load_all_data(args.data_dir)
    df = add_training_features(df)
    df = add_advanced_historical_features(df)
    train_df, _, test_df = split_data(df)

    evaluate_xgboost(models_dir, test_df)
    evaluate_gnn(models_dir, train_df, args.gnn_rows)


if __name__ == "__main__":
    main()
