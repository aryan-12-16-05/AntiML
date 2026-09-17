"""
train.py — End-to-end training pipeline for all AML models.
Run this once to train XGBoost, GNN, and Pattern Classifier.
Saves all artifacts to models/ directory.

Usage:
    .\\aml_env\\Scripts\\python train.py
    .\\aml_env\\Scripts\\python train.py --nrows 500000   # quick test
    .\\aml_env\\Scripts\\python train.py --skip-gnn       # skip GNN (CPU only)
"""
import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train AML Detection Models")
    parser.add_argument("--data-dir", default=".", help="Directory with data files")
    parser.add_argument("--models-dir", default="models", help="Output directory for models")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for quick testing")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN training")
    parser.add_argument("--gnn-epochs", type=int, default=30, help="GNN training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("AML SYSTEM — MODEL TRAINING")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Load and enrich data                                         #
    # ------------------------------------------------------------------ #
    logger.info("\n[1/6] Loading data ...")
    from src.pipeline.ingestion import load_all_data
    df, accounts_df = load_all_data(args.data_dir, nrows=args.nrows)
    logger.info(f"Dataset: {len(df):,} transactions | {df['is_laundering'].sum():,} laundering")

    # ------------------------------------------------------------------ #
    # Step 2: Feature engineering                                          #
    # ------------------------------------------------------------------ #
    logger.info("\n[2/6] Feature engineering ...")
    from src.pipeline.feature_engineering import add_log_features
    df = add_log_features(df)

    # Add simplified rolling features (fast approximation)
    logger.info("Computing rolling stats ...")
    import pandas as pd
    import numpy as np
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fast cumulative features — NO expanding() which is O(n²) per group
    # ------------------------------------------------------------------
    logger.info("  cumcount per account ...")
    df["from_tx_count_7d"] = df.groupby("from_account").cumcount() + 1
    df["to_tx_count_7d"]   = df.groupby("to_account").cumcount() + 1
    df["from_out_degree"]  = df["from_tx_count_7d"]
    df["from_in_degree"]   = df["to_tx_count_7d"]
    df["to_out_degree"]    = df["from_out_degree"]
    df["to_in_degree"]     = df["from_in_degree"]

    logger.info("  cumsum amounts ...")
    df["from_total_sent_7d"]     = df.groupby("from_account")["amount_received_usd"].cumsum()
    df["to_total_received_7d"]   = df.groupby("to_account")["amount_received_usd"].cumsum()

    # Fast running mean = cumsum / cumcount (avoids per-row lambda)
    logger.info("  running mean amounts ...")
    df["from_avg_amount_7d"] = df["from_total_sent_7d"] / df["from_tx_count_7d"]
    df["to_avg_received_7d"] = df["to_total_received_7d"] / df["to_tx_count_7d"]

    # Running std via Welford-equivalent: var = (sum_sq/n) - mean²
    # Use direct groupby cumsum on sq column (no lambda = faster)
    logger.info("  running std amounts ...")
    df["_amt_sq"] = df["amount_received_usd"] ** 2
    df["_from_sum_sq"] = df.groupby("from_account")["_amt_sq"].cumsum()
    df["from_std_amount_7d"] = np.sqrt(
        np.maximum(0, df["_from_sum_sq"] / df["from_tx_count_7d"] - df["from_avg_amount_7d"] ** 2)
    ).fillna(0)
    df.drop(columns=["_amt_sq", "_from_sum_sq"], inplace=True)

    # 7-day unique recipients/senders — use global per-account nunique as proxy
    # (true 7D window is too slow on 5M rows; global count still captures fan-out)
    logger.info("  unique recipients/senders ...")
    from_uniq = df.groupby("from_account")["to_account"].transform("nunique")
    to_uniq   = df.groupby("to_account")["from_account"].transform("nunique")
    df["from_unique_recipients_7d"] = from_uniq
    df["to_unique_senders_7d"]      = to_uniq

    logger.info("Feature engineering done.")


    # ------------------------------------------------------------------ #
    # Step 3: Data splits                                                  #
    # ------------------------------------------------------------------ #
    logger.info("\n[3/6] Preparing train/val/test splits ...")
    from src.pipeline.data_prep import prepare_for_training
    data = prepare_for_training(df, models_dir=args.models_dir, use_smote=True)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    scaler = data["scaler"]
    scale_pos_weight = data["scale_pos_weight"]
    feature_cols = data["feature_cols"]

    # ------------------------------------------------------------------ #
    # Step 4: Train XGBoost                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n[4/6] Training XGBoost model ...")
    from src.models.xgboost_model import XGBoostAMLModel
    xgb_model = XGBoostAMLModel()
    xgb_model.train(
        X_train, y_train,
        X_val, y_val,
        scale_pos_weight=scale_pos_weight,
        feature_names=feature_cols,
    )
    xgb_model.tune_threshold(X_val, y_val, min_recall=0.75)

    logger.info("XGBoost evaluation on test set:")
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save(f"{args.models_dir}/xgboost_aml.pkl")

    # ------------------------------------------------------------------ #
    # Step 5: Train GNN                                                    #
    # ------------------------------------------------------------------ #
    if not args.skip_gnn:
        logger.info("\n[5/6] Training Graph Transformer (GNN) ...")
        try:
            from src.models.gnn_model import GNNAMLModel, HAS_PYG
            if HAS_PYG:
                # Use subset for GNN (full 5M is too large for single graph training)
                max_gnn_rows = min(500_000, len(data["train_df"]))
                train_df_gnn = data["train_df"].head(max_gnn_rows)
                val_df_gnn = data["val_df"].head(50_000)

                gnn_model = GNNAMLModel(
                    hidden_dim=128,
                    num_heads=4,
                    num_layers=3,
                )
                gnn_model.train(
                    train_df=train_df_gnn,
                    val_df=val_df_gnn,
                    epochs=args.gnn_epochs,
                )
                gnn_model.save(f"{args.models_dir}/gnn_aml.pt")
                logger.info("GNN training complete.")
            else:
                logger.warning("PyG not available. Skipping GNN training.")
        except Exception as e:
            logger.error(f"GNN training failed: {e}. Continuing without GNN.")
    else:
        logger.info("[5/6] Skipping GNN training (--skip-gnn flag set)")

    # ------------------------------------------------------------------ #
    # Step 6: Train Pattern Classifier                                     #
    # ------------------------------------------------------------------ #
    logger.info("\n[6/6] Training Pattern Classifier ...")

    # Get laundering rows from training data
    train_df = data["train_df"]
    launder_df = train_df[train_df["is_laundering"] == 1].copy()

    if len(launder_df) > 50:
        logger.info(f"Labeling {len(launder_df):,} laundering rows with rule engine patterns ...")

        # Use the rule engine to label each laundering transaction with its pattern
        # This is far more reliable than trying to match the pattern file text
        from src.models.rule_engine import RuleEngine
        rule_engine = RuleEngine()

        # Build a mini graph just from laundering transactions for pattern detection
        import networkx as nx
        G = nx.MultiDiGraph()
        for _, row in launder_df.iterrows():
            G.add_edge(
                str(row["from_account"]),
                str(row["to_account"]),
                amount=float(row.get("amount_received_usd", 0)),
                timestamp=row.get("timestamp"),
                payment_format=str(row.get("payment_format", "")),
            )

        pattern_labels_list = []
        for _, row in launder_df.iterrows():
            tx = row.to_dict()
            tx["from_account"] = str(tx["from_account"])
            tx["to_account"] = str(tx["to_account"])
            result = rule_engine.evaluate(tx, G)
            # Pick the highest-confidence pattern, default to RANDOM
            pattern = result.get("detected_pattern", "RANDOM") or "RANDOM"
            pattern_labels_list.append(pattern)

        launder_df["pattern_label"] = pattern_labels_list

        # Log distribution
        from collections import Counter
        dist = Counter(pattern_labels_list)
        logger.info(f"Pattern distribution: {dict(dist)}")

        # Only train if we have multiple classes
        unique_patterns = list(set(pattern_labels_list))
        if len(unique_patterns) >= 2:
            from src.pipeline.feature_engineering import FEATURE_COLS
            X_launder = launder_df[[c for c in FEATURE_COLS if c in launder_df.columns]].fillna(0).values
            y_launder = launder_df["pattern_label"].tolist()

            from src.models.pattern_classifier import PatternClassifier
            pattern_clf = PatternClassifier()
            pattern_clf.train(X_launder, y_launder, feature_names=feature_cols)
            pattern_clf.save(f"{args.models_dir}/pattern_classifier.pkl")
        else:
            logger.warning(f"Only 1 pattern class found ({unique_patterns}). "
                           "Skipping pattern classifier — rule engine will handle classification.")
    else:
        logger.warning("Too few laundering samples for pattern classifier. Skipping.")


    # ------------------------------------------------------------------ #
    # Build customer profiles                                              #
    # ------------------------------------------------------------------ #
    logger.info("Building customer profiles ...")
    from src.customer.profiler import CustomerProfiler
    profiler = CustomerProfiler()
    profiler.build_profiles(df, accounts_df)
    profiler.save(f"{args.models_dir}/customer_profiles.pkl")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"XGBoost ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    logger.info(f"XGBoost F1:      {xgb_metrics['f1']:.4f}")
    logger.info(f"Models saved to: {args.models_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
