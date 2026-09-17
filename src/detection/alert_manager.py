"""
alert_manager.py — Creates, deduplicates, and manages laundering alerts.
Persists to SQLite database for dashboard consumption.
"""
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, JSON
from loguru import logger
import json


# ------------------------------------------------------------------ #
#  SQLAlchemy Models                                                   #
# ------------------------------------------------------------------ #
class Base(DeclarativeBase):
    pass


class AlertORM(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tx_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=True)
    from_account = Column(String, nullable=True)
    to_account = Column(String, nullable=True)
    from_bank = Column(String, nullable=True)
    to_bank = Column(String, nullable=True)
    amount_usd = Column(Float, nullable=True)
    payment_format = Column(String, nullable=True)

    # Detection results
    is_laundering = Column(Boolean, default=True)
    final_score = Column(Float, nullable=True)
    detected_pattern = Column(String, nullable=True)
    severity = Column(String, default="MEDIUM")
    retroactive = Column(Boolean, default=False)
    triggering_tx_id = Column(String, nullable=True)
    retroactive_reason = Column(Text, nullable=True)

    # Model scores
    rule_score = Column(Float, nullable=True)
    xgb_score = Column(Float, nullable=True)
    gnn_score = Column(Float, nullable=True)
    shap_values = Column(JSON, nullable=True)
    triggered_rules = Column(JSON, nullable=True)
    all_patterns = Column(JSON, nullable=True)

    # Account profiles
    from_entity_type = Column(String, nullable=True)
    to_entity_type = Column(String, nullable=True)
    involved_accounts = Column(JSON, nullable=True)

    # Review status
    status = Column(String, default="PENDING")  # PENDING | APPROVED | REJECTED
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------------------------------------------------ #
#  Alert Manager                                                       #
# ------------------------------------------------------------------ #
class AlertManager:
    """
    Manages the lifecycle of laundering alerts:
    - Creation from detection pipeline output
    - Deduplication (same tx_id → no double-alert)
    - Persistence to SQLite
    - WebSocket broadcast to dashboard
    """

    def __init__(self, db_url: str = "sqlite+aiosqlite:///data/alerts.db"):
        self.db_url = db_url
        self.engine = None
        self.session_factory = None
        self._seen_tx_ids: set = set()
        self._ws_callbacks = []
        self._alert_buffer: List[Dict] = []

    async def init(self) -> None:
        """Initialize database connection and create tables."""
        self.engine = create_async_engine(self.db_url, echo=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"AlertManager initialized with DB: {self.db_url}")

    async def create_alert(self, detection_result: Dict[str, Any]) -> Optional[Dict]:
        """
        Create and persist a new alert from detection result.
        Returns the alert dict or None if duplicate.
        """
        tx_id = detection_result.get("tx_id", str(uuid.uuid4()))

        # Dedup
        if tx_id in self._seen_tx_ids:
            return None
        self._seen_tx_ids.add(tx_id)

        alert_id = str(uuid.uuid4())
        ts = detection_result.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        alert = AlertORM(
            id=alert_id,
            tx_id=tx_id,
            timestamp=ts,
            from_account=str(detection_result.get("from_account", "")),
            to_account=str(detection_result.get("to_account", "")),
            from_bank=str(detection_result.get("from_bank", "")),
            to_bank=str(detection_result.get("to_bank", "")),
            amount_usd=float(detection_result.get("amount_received_usd", 0.0)),
            payment_format=detection_result.get("payment_format", ""),
            is_laundering=bool(detection_result.get("is_laundering", True)),
            final_score=float(detection_result.get("final_score", 0.0)),
            detected_pattern=detection_result.get("detected_pattern", "UNKNOWN"),
            severity=detection_result.get("severity", "MEDIUM"),
            retroactive=bool(detection_result.get("retroactive", False)),
            triggering_tx_id=detection_result.get("triggering_tx_id"),
            retroactive_reason=detection_result.get("retroactive_reason"),
            rule_score=detection_result.get("component_scores", {}).get("rule"),
            xgb_score=detection_result.get("component_scores", {}).get("xgboost"),
            gnn_score=detection_result.get("component_scores", {}).get("gnn"),
            shap_values=detection_result.get("shap_values"),
            triggered_rules=detection_result.get("triggered_patterns", []),
            all_patterns=detection_result.get("all_patterns", []),
            from_entity_type=detection_result.get("from_entity_type", "Unknown"),
            to_entity_type=detection_result.get("to_entity_type", "Unknown"),
            involved_accounts=detection_result.get("involved_accounts", []),
            status="PENDING",
            created_at=datetime.utcnow(),
        )

        async with self.session_factory() as session:
            session.add(alert)
            await session.commit()

        alert_dict = self._orm_to_dict(alert)
        self._alert_buffer.append(alert_dict)

        # Notify WebSocket listeners
        await self._broadcast(alert_dict)

        return alert_dict

    async def get_alerts(self, status: Optional[str] = None,
                         limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch alerts from DB with optional status filter."""
        from sqlalchemy import select, desc
        async with self.session_factory() as session:
            q = select(AlertORM).order_by(desc(AlertORM.created_at)).offset(offset).limit(limit)
            if status:
                q = q.where(AlertORM.status == status)
            result = await session.execute(q)
            rows = result.scalars().all()
        return [self._orm_to_dict(r) for r in rows]

    async def update_alert_status(self, alert_id: str, status: str,
                                   reviewer_note: str = "") -> Optional[Dict]:
        """Update alert review status (APPROVED / REJECTED)."""
        from sqlalchemy import select
        async with self.session_factory() as session:
            result = await session.execute(
                select(AlertORM).where(AlertORM.id == alert_id)
            )
            alert = result.scalar_one_or_none()
            if not alert:
                return None
            alert.status = status
            alert.reviewed_at = datetime.utcnow()
            alert.reviewer_note = reviewer_note
            await session.commit()
            return self._orm_to_dict(alert)

    async def get_stats(self) -> Dict:
        """Get alert statistics for dashboard KPI cards."""
        from sqlalchemy import select, func
        async with self.session_factory() as session:
            total = await session.scalar(select(func.count(AlertORM.id)))
            pending = await session.scalar(
                select(func.count(AlertORM.id)).where(AlertORM.status == "PENDING")
            )
            approved = await session.scalar(
                select(func.count(AlertORM.id)).where(AlertORM.status == "APPROVED")
            )
            rejected = await session.scalar(
                select(func.count(AlertORM.id)).where(AlertORM.status == "REJECTED")
            )
            retroactive = await session.scalar(
                select(func.count(AlertORM.id)).where(AlertORM.retroactive == True)
            )
            critical = await session.scalar(
                select(func.count(AlertORM.id)).where(AlertORM.severity == "CRITICAL")
            )
        return {
            "total": total or 0,
            "pending": pending or 0,
            "approved": approved or 0,
            "rejected": rejected or 0,
            "retroactive": retroactive or 0,
            "critical": critical or 0,
        }

    def register_ws_callback(self, callback) -> None:
        """Register a WebSocket broadcast callback."""
        self._ws_callbacks.append(callback)

    async def _broadcast(self, alert_dict: Dict) -> None:
        """Broadcast new alert to all registered WebSocket clients."""
        for cb in self._ws_callbacks:
            try:
                await cb(alert_dict)
            except Exception as e:
                logger.warning(f"WS broadcast error: {e}")

    @staticmethod
    def _orm_to_dict(alert: AlertORM) -> Dict:
        return {
            "id": alert.id,
            "tx_id": alert.tx_id,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "from_account": alert.from_account,
            "to_account": alert.to_account,
            "from_bank": alert.from_bank,
            "to_bank": alert.to_bank,
            "amount_usd": alert.amount_usd,
            "payment_format": alert.payment_format,
            "is_laundering": alert.is_laundering,
            "final_score": alert.final_score,
            "detected_pattern": alert.detected_pattern,
            "severity": alert.severity,
            "retroactive": alert.retroactive,
            "triggering_tx_id": alert.triggering_tx_id,
            "retroactive_reason": alert.retroactive_reason,
            "component_scores": {
                "rule": alert.rule_score,
                "xgboost": alert.xgb_score,
                "gnn": alert.gnn_score,
            },
            "shap_values": alert.shap_values,
            "triggered_rules": alert.triggered_rules,
            "all_patterns": alert.all_patterns,
            "from_entity_type": alert.from_entity_type,
            "to_entity_type": alert.to_entity_type,
            "involved_accounts": alert.involved_accounts or [],
            "status": alert.status,
            "reviewed_at": alert.reviewed_at.isoformat() if alert.reviewed_at else None,
            "reviewer_note": alert.reviewer_note,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
        }
