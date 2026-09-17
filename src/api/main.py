"""
main.py — FastAPI application entry point.
Initializes all models, alert manager, whitelist, and WebSocket support.
"""
import asyncio
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.detection.alert_manager import AlertManager
from src.customer.whitelist import Whitelist
from src.customer.profiler import CustomerProfiler
from src.models.ensemble import EnsembleDecisionMaker
from src.api.models_api import ModelWeightsRequest, TransactionIn

# Global instances
alert_manager: Optional[AlertManager] = None
whitelist: Optional[Whitelist] = None
customer_profiler: Optional[CustomerProfiler] = None
detector = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


ws_manager = ConnectionManager()


def get_alert_manager():
    return alert_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize all components."""
    global alert_manager, whitelist, customer_profiler, detector

    Path("data").mkdir(exist_ok=True)

    # Initialize Alert Manager
    alert_manager = AlertManager(db_url="sqlite+aiosqlite:///data/alerts.db")
    await alert_manager.init()

    # Register WS broadcast callback
    async def broadcast_alert(alert_dict):
        await ws_manager.broadcast({"type": "new_alert", "data": alert_dict})
    alert_manager.register_ws_callback(broadcast_alert)

    # Initialize Whitelist (shares same DB)
    whitelist = Whitelist(db_url="sqlite+aiosqlite:///data/alerts.db")
    await whitelist.init()

    # Initialize Customer Profiler (load from disk if exists)
    customer_profiler = CustomerProfiler()
    profiler_path = "models/customer_profiles.pkl"
    if Path(profiler_path).exists():
        customer_profiler.load(profiler_path)
        logger.info("Customer profiles loaded from disk.")

    # Initialize Detector (load trained models if exist)
    from src.detection.detector import StreamingDetector
    from src.pipeline.graph_builder import TransactionGraph

    graph = TransactionGraph(window_days=30)
    scaler = None
    xgb_model = None
    gnn_model = None
    pattern_classifier = None

    import joblib
    if Path("models/scaler.pkl").exists():
        scaler = joblib.load("models/scaler.pkl")
        logger.info("Scaler loaded.")

    from src.models.xgboost_model import XGBoostAMLModel
    if Path("models/xgboost_aml.pkl").exists():
        xgb_model = XGBoostAMLModel(model_path="models/xgboost_aml.pkl")

    from src.models.gnn_model import GNNAMLModel, HAS_PYG
    if HAS_PYG and Path("models/gnn_aml.pt").exists():
        gnn_model = GNNAMLModel(model_path="models/gnn_aml.pt")

    from src.models.pattern_classifier import PatternClassifier
    if Path("models/pattern_classifier.pkl").exists():
        pattern_classifier = PatternClassifier(model_path="models/pattern_classifier.pkl")

    detector = StreamingDetector(
        xgb_model=xgb_model,
        gnn_model=gnn_model,
        pattern_classifier=pattern_classifier,
        graph=graph,
        whitelist=whitelist,
        customer_profiler=customer_profiler,
        alert_manager=alert_manager,
        scaler=scaler,
    )
    logger.info("AML Streaming Detector initialized.")

    yield  # App is running

    logger.info("Shutting down AML API.")


# ------------------------------------------------------------------ #
#  FastAPI App                                                         #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="AML Detection API",
    description="Anti-Money Laundering Detection System API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from src.api.routes.alerts import router as alerts_router
from src.api.routes.decisions import router as decisions_router
from src.api.routes.profiles import router as profiles_router

app.include_router(alerts_router)
app.include_router(decisions_router)
app.include_router(profiles_router)


# ------------------------------------------------------------------ #
#  WebSocket endpoint for live alert feed                             #
# ------------------------------------------------------------------ #
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send recent alerts on connect
        if alert_manager:
            recent = await alert_manager.get_alerts(limit=20)
            await websocket.send_text(json.dumps({
                "type": "initial_alerts",
                "data": recent,
            }, default=str))
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ------------------------------------------------------------------ #
#  Transaction ingest endpoint (for streaming simulation)             #
# ------------------------------------------------------------------ #
@app.post("/api/ingest")
async def ingest_transaction(tx: TransactionIn):
    """Process a single incoming transaction through the AML pipeline."""
    if detector is None:
        raise HTTPException(503, "Detector not initialized")

    from src.pipeline.ingestion import CURRENCY_TO_USD
    import pandas as pd

    tx_dict = tx.model_dump()
    ts = pd.to_datetime(tx_dict["timestamp"])
    amount_usd = tx_dict["amount_received"] * CURRENCY_TO_USD.get(tx_dict["receiving_currency"], 1.0)

    enriched_tx = {
        **tx_dict,
        "timestamp": ts,
        "amount_received_usd": amount_usd,
        "amount_paid_usd": tx_dict["amount_paid"] * CURRENCY_TO_USD.get(tx_dict["payment_currency"], 1.0),
        "is_self_transaction": int(tx_dict["from_account"] == tx_dict["to_account"]),
        "same_bank": int(tx_dict["from_bank"] == tx_dict["to_bank"]),
        "currency_changed": int(tx_dict["receiving_currency"] != tx_dict["payment_currency"]),
        "amount_ratio": amount_usd / (tx_dict["amount_paid"] + 1e-9),
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "payment_format_enc": {"ACH": 0, "Cheque": 1, "Credit Card": 2, "Cash": 3,
                                "Reinvestment": 4, "Wire": 5, "Bitcoin": 6}.get(
            tx_dict["payment_format"], -1
        ),
    }

    result = await detector.process_transaction(enriched_tx)
    return {
        "tx_id": result.get("tx_id"),
        "is_laundering": result.get("is_laundering"),
        "final_score": result.get("final_score"),
        "detected_pattern": result.get("detected_pattern"),
        "severity": result.get("severity"),
        "component_scores": result.get("component_scores"),
    }


@app.get("/api/detector/stats")
async def get_detector_stats():
    """Get detector processing stats."""
    if detector is None:
        return {}
    return detector.stats


@app.post("/api/detector/weights")
async def update_model_weights(request: ModelWeightsRequest):
    """Dynamically update ensemble weights from dashboard."""
    if detector is None:
        raise HTTPException(503, "Detector not initialized")
    detector.update_ensemble_weights(
        request.rule_weight, request.xgb_weight, request.gnn_weight
    )
    if request.threshold is not None:
        detector.update_threshold(request.threshold)
    return {"success": True, "message": "Weights updated"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "xgboost": detector.xgb_model is not None if detector else False,
            "gnn": detector.gnn_model is not None if detector else False,
            "pattern_classifier": detector.pattern_classifier is not None if detector else False,
            "customer_profiler": len(customer_profiler.profiles) > 0 if customer_profiler else False,
        }
    }
