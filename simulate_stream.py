"""
simulate_stream.py — Replays transactions in timestamp order to the AML pipeline.
Simulates real-time transaction stream for testing and dashboard demos.

Usage:
    .\\aml_env\\Scripts\\python simulate_stream.py
    .\\aml_env\\Scripts\\python simulate_stream.py --speed 100 --max-txs 5000
    .\\aml_env\\Scripts\\python simulate_stream.py --include-laundering-only
"""
import asyncio
import argparse
import sys
import httpx
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

API_URL = "http://localhost:8000"


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate AML Transaction Stream")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="Transactions per second to send (default: 1000)")
    parser.add_argument("--max-txs", type=int, default=10000,
                        help="Maximum transactions to send (default: 10000)")
    parser.add_argument("--include-laundering-only", action="store_true",
                        help="Only send transactions that ARE laundering (for testing)")
    parser.add_argument("--data-dir", default=".", help="Data directory")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from row N of the dataset")
    return parser.parse_args()


async def simulate(args):
    logger.info("Loading transaction data ...")
    # Load enough rows to find laundering transactions if filtering
    load_nrows = args.start_from + args.max_txs * 10 if not args.include_laundering_only else None
    df = pd.read_csv(f"{args.data_dir}/HI-Small_Trans.csv", nrows=load_nrows)

    # Normalize column names (handle with/without header)
    if df.columns[0].lower() in ('timestamp', 'date', 'time'):
        df.columns = [
            "timestamp", "from_bank", "from_account",
            "to_bank", "to_account",
            "amount_received", "receiving_currency",
            "amount_paid", "payment_currency",
            "payment_format", "is_laundering"
        ]
    else:
        df.columns = [
            "timestamp", "from_bank", "from_account",
            "to_bank", "to_account",
            "amount_received", "receiving_currency",
            "amount_paid", "payment_currency",
            "payment_format", "is_laundering"
        ]
    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if args.include_laundering_only:
        df = df[df["is_laundering"] == 1].copy()
        logger.info(f"Filtered to {len(df):,} laundering-only transactions.")
        if len(df) == 0:
            logger.error("No laundering transactions found! Check your data file.")
            return

    df = df.iloc[args.start_from:args.start_from + args.max_txs].copy()
    delay = 1.0 / args.speed

    logger.info(f"Sending {len(df):,} transactions at {args.speed}/sec to {API_URL}")
    logger.info("Make sure the API server is running: python -m uvicorn src.api.main:app --reload")

    sent = 0
    flagged = 0
    errors = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check health
        try:
            resp = await client.get(f"{API_URL}/api/health")
            logger.info(f"API health: {resp.json()}")
        except Exception as e:
            logger.error(f"Cannot connect to API: {e}")
            logger.error("Start the API first: .\\aml_env\\Scripts\\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
            return

        for _, row in df.iterrows():
            payload = {
                "timestamp": str(row["timestamp"]),
                "from_bank": str(row["from_bank"]),
                "from_account": str(row["from_account"]),
                "to_bank": str(row["to_bank"]),
                "to_account": str(row["to_account"]),
                "amount_received": float(row["amount_received"]),
                "receiving_currency": str(row["receiving_currency"]),
                "amount_paid": float(row["amount_paid"]),
                "payment_currency": str(row["payment_currency"]),
                "payment_format": str(row["payment_format"]),
            }

            try:
                resp = await client.post(f"{API_URL}/api/ingest", json=payload)
                result = resp.json()
                sent += 1
                if result.get("is_laundering"):
                    flagged += 1
                    logger.info(
                        f"⚠ FLAGGED [{sent}] {row['from_account'][:8]}→{row['to_account'][:8]} "
                        f"| Pattern: {result.get('detected_pattern')} "
                        f"| Score: {result.get('final_score', 0):.3f}"
                    )
                elif sent % 500 == 0:
                    logger.info(f"[{sent}] Processed | Flagged: {flagged} | Errors: {errors}")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Error sending tx: {e}")

            if delay > 0:
                await asyncio.sleep(delay)

    logger.info(f"\n{'='*50}")
    logger.info(f"SIMULATION COMPLETE")
    logger.info(f"Sent: {sent:,} | Flagged: {flagged:,} | Errors: {errors:,}")
    if sent > 0:
        logger.info(f"Detection rate: {flagged/sent*100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(simulate(args))
