"""
data_prep.py — Train/Val/Test splits, class imbalance handling with SMOTE,
               and label encoding for model training.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from loguru import logger
import joblib
from pathlib import Path

from src.pipeline.feature_engineering import FEATURE_COLS, get_feature_matrix


def split_data(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15,
               random_state: int = 42):
    """
    Temporal train/val/test split.
    Uses time ordering: train = first 70%, val = next 15%, test = last 15%.
    This simulates real-world deployment where we predict on future transactions.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Split sizes — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    logger.info(f"Train laundering rate: {train_df['is_laundering'].mean()*100:.3f}%")
    logger.info(f"Val laundering rate:   {val_df['is_laundering'].mean()*100:.3f}%")
    logger.info(f"Test laundering rate:  {test_df['is_laundering'].mean()*100:.3f}%")

    return train_df, val_df, test_df


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                sampling_strategy: float = 0.1, random_state: int = 42):
    """
    Balance training set. Uses RandomOverSampler (instant even on 5M rows).
    SMOTE is too slow for 3.5M samples (KNN on millions = hours).
    XGBoost additionally gets scale_pos_weight, so oversampling just 2x is enough.
    sampling_strategy=0.1 means minority becomes 10% of majority.
    """
    from imblearn.over_sampling import RandomOverSampler
    n_pos = int(y_train.sum())
    n_neg = int((1 - y_train).sum())
    logger.info(f"Before oversampling — Positive: {n_pos:,} | Negative: {n_neg:,}")

    # Only oversample to sampling_strategy ratio to avoid memory explosion
    target_pos = int(n_neg * sampling_strategy)
    if target_pos <= n_pos:
        logger.info("No oversampling needed (already balanced enough).")
        return X_train, y_train

    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    X_res, y_res = ros.fit_resample(X_train, y_train)
    logger.info(f"After oversampling  — Positive: {int(y_res.sum()):,} | Negative: {int((1-y_res).sum()):,}")
    return X_res, y_res



def compute_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost (ratio of negatives to positives)."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    weight = n_neg / (n_pos + 1e-9)
    logger.info(f"scale_pos_weight = {weight:.2f}")
    return weight


def fit_scaler(X_train: np.ndarray, save_path: str = None) -> StandardScaler:
    """Fit StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    if save_path:
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved to {save_path}")
    return scaler


def prepare_for_training(df: pd.DataFrame, models_dir: str = "models",
                          use_smote: bool = True) -> dict:
    """
    Full pipeline: split → feature extract → scale → SMOTE.
    Returns dict with all splits and metadata.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = split_data(df)

    X_train, y_train = get_feature_matrix(train_df)
    X_val, y_val = get_feature_matrix(val_df)
    X_test, y_test = get_feature_matrix(test_df)

    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    # Fit scaler
    scaler = fit_scaler(X_train, save_path=f"{models_dir}/scaler.pkl")
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # SMOTE on training
    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train_sc, y_train)
    else:
        X_train_sm, y_train_sm = X_train_sc, y_train

    scale_pos_weight = compute_class_weight(y_train)

    return {
        "train": (X_train_sm, y_train_sm),
        "val": (X_val_sc, y_val),
        "test": (X_test_sc, y_test),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "scale_pos_weight": scale_pos_weight,
        "feature_cols": FEATURE_COLS,
    }
