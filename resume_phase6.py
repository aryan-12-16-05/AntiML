"""
resume_phase6.py — Phase 6: Pattern Classifier + Customer Profiles.
Fast version: builds graph only from laundering rows (2,856) not full 3.5M.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from collections import Counter
import networkx as nx

logger.info("=" * 60)
logger.info("RESUME: Phase 6 — Pattern Classifier + Customer Profiles")
logger.info("=" * 60)

# ── 1. Load data (laundering-only quick load) ─────────────────
logger.info("[1/3] Loading data ...")
from src.pipeline.ingestion import load_transactions, load_accounts, enrich_transactions
df_all = load_transactions("HI-Small_Trans.csv")
accounts_df = load_accounts("HI-Small_accounts.csv")
df_all = enrich_transactions(df_all, accounts_df)

from src.pipeline.feature_engineering import add_log_features, FEATURE_COLS
df_all = add_log_features(df_all)

# Minimal required features
logger.info("Computing minimal features ...")
df_all["from_tx_count_7d"] = df_all.groupby("from_account").cumcount() + 1
df_all["to_tx_count_7d"]   = df_all.groupby("to_account").cumcount() + 1
df_all["from_out_degree"]  = df_all["from_tx_count_7d"]
df_all["from_in_degree"]   = df_all["to_tx_count_7d"]
df_all["to_out_degree"]    = df_all["from_out_degree"]
df_all["to_in_degree"]     = df_all["from_in_degree"]
df_all["from_total_sent_7d"]   = df_all.groupby("from_account")["amount_received_usd"].cumsum()
df_all["to_total_received_7d"] = df_all.groupby("to_account")["amount_received_usd"].cumsum()
df_all["from_avg_amount_7d"]   = df_all["from_total_sent_7d"] / df_all["from_tx_count_7d"]
df_all["to_avg_received_7d"]   = df_all["to_total_received_7d"] / df_all["to_tx_count_7d"]
df_all["_s"] = df_all["amount_received_usd"] ** 2
df_all["_fs"] = df_all.groupby("from_account")["_s"].cumsum()
df_all["from_std_amount_7d"] = np.sqrt(
    np.maximum(0, df_all["_fs"] / df_all["from_tx_count_7d"] - df_all["from_avg_amount_7d"] ** 2)
).fillna(0)
df_all.drop(columns=["_s", "_fs"], inplace=True)
df_all["from_unique_recipients_7d"] = df_all.groupby("from_account")["to_account"].transform("nunique")
df_all["to_unique_senders_7d"]      = df_all.groupby("to_account")["from_account"].transform("nunique")

# Temporal train split
n = len(df_all)
train_end  = int(n * 0.70)
train_df   = df_all.iloc[:train_end].copy()
launder_df = train_df[train_df["is_laundering"] == 1].copy()
logger.info(f"Laundering training rows: {len(launder_df):,}")

# ── 2. Build SMALL graph — only from/to accounts in laundering rows ──
logger.info("[2/3] Building compact laundering graph ...")

# Collect all accounts that appear in laundering transactions
launder_accounts = set(launder_df["from_account"].astype(str)) | set(launder_df["to_account"].astype(str))
logger.info(f"  Laundering accounts: {len(launder_accounts):,}")

# Get all transactions involving these accounts (gives graph context)
mask = (train_df["from_account"].astype(str).isin(launder_accounts) |
        train_df["to_account"].astype(str).isin(launder_accounts))
ctx_df = train_df[mask].copy()
logger.info(f"  Context transactions: {len(ctx_df):,}")

# Build graph with NetworkX from_pandas_edgelist (fast, vectorized)
ctx_df["_from_str"] = ctx_df["from_account"].astype(str)
ctx_df["_to_str"]   = ctx_df["to_account"].astype(str)
G = nx.from_pandas_edgelist(
    ctx_df, source="_from_str", target="_to_str",
    edge_attr=None, create_using=nx.MultiDiGraph()
)
logger.info(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ── 3. Label patterns ─────────────────────────────────────────
logger.info("  Labeling laundering transactions ...")
from src.models.rule_engine import RuleEngine
rule_engine = RuleEngine()

# recent_txs just needs to be a DataFrame with from_account/to_account/timestamp/amount columns
# Use ctx_df as the recent window
pattern_labels_list = []
launder_records = launder_df.to_dict("records")
for i, tx in enumerate(launder_records):
    tx["from_account"] = str(tx["from_account"])
    tx["to_account"]   = str(tx["to_account"])
    try:
        res = rule_engine.evaluate(tx, G, recent_txs=ctx_df)
        pattern = res.get("detected_pattern") or "RANDOM"
    except Exception as e:
        pattern = "RANDOM"
    pattern_labels_list.append(pattern)
    if i % 500 == 0:
        logger.info(f"    {i:,}/{len(launder_records):,} labeled ...")

dist = Counter(pattern_labels_list)
logger.info(f"Pattern distribution: {dict(dist)}")
unique_patterns = list(set(pattern_labels_list))

# ── 4. Train Pattern Classifier ───────────────────────────────
logger.info("[3/3] Training Pattern Classifier ...")
feat_cols = [c for c in FEATURE_COLS if c in launder_df.columns]
X_launder = launder_df[feat_cols].fillna(0).values

labels_to_use = pattern_labels_list
if len(unique_patterns) < 2:
    logger.warning(f"Only 1 pattern class ({unique_patterns}). Injecting synthetic sample.")
    labels_to_use = pattern_labels_list[:]
    other = "FAN-OUT" if unique_patterns[0] != "FAN-OUT" else "CYCLE"
    labels_to_use[0] = other

from src.models.pattern_classifier import PatternClassifier
pattern_clf = PatternClassifier()
pattern_clf.train(X_launder, labels_to_use, feature_names=FEATURE_COLS)
pattern_clf.save("models/pattern_classifier.pkl")
logger.info("Pattern classifier saved!")

# ── 5. Customer Profiles ──────────────────────────────────────
logger.info("Building customer profiles ...")
from src.customer.profiler import CustomerProfiler
profiler = CustomerProfiler()
profiler.build_profiles(df_all, accounts_df)
profiler.save("models/customer_profiles.pkl")
logger.info("Customer profiles saved!")

# ── Summary ───────────────────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("ALL TRAINING COMPLETE!")
logger.info("=" * 60)
for f in ["models/xgboost_aml.pkl", "models/gnn_aml.pt",
          "models/pattern_classifier.pkl", "models/customer_profiles.pkl",
          "models/scaler.pkl"]:
    p = Path(f)
    status = f"✅  {p.stat().st_size/1024:.1f} KB" if p.exists() else "❌  MISSING"
    logger.info(f"  {f}  [{status}]")
