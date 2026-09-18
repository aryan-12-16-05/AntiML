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
    parser.add_argument("--min-recall", type=float, default=0.60, help="Minimum validation recall for XGBoost threshold")
    parser.add_argument("--min-precision", type=float, default=0.05, help="Minimum validation precision for XGBoost threshold")
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
    from src.pipeline.feature_engineering import (
        add_log_features,
        add_advanced_historical_features,
        add_causal_legacy_features,
    )
    df = add_log_features(df)
    df = add_advanced_historical_features(df)
    df = add_causal_legacy_features(df)

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
    xgb_model.tune_threshold(
        X_val,
        y_val,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
    )

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
        from src.pipeline.graph_builder import TransactionGraph
        G = TransactionGraph(window_days=7)
        G.add_transactions_bulk(launder_df)

        pattern_labels_list = []
        for _, row in launder_df.iterrows():
            tx = row.to_dict()
            tx["from_account"] = str(tx["from_account"])
            tx["to_account"] = str(tx["to_account"])
            result = rule_engine.evaluate(tx, G, recent_txs=train_df)
            # Pick the highest-confidence pattern, default to RANDOM
            confidence_details = result.get("confidence_details", {})
            pattern = max(confidence_details, key=confidence_details.get, default="RANDOM")
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
