"""
alerts.py — Alert CRUD API routes.
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from src.api.models_api import AlertResponse, AlertListResponse, StatsResponse
from src.api.main import get_alert_manager

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status: PENDING|APPROVED|REJECTED"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List all alerts with optional status filter and pagination."""
    from src.api.main import alert_manager
    if alert_manager is None:
        return AlertListResponse(alerts=[], total=0, page=page, page_size=page_size)

    offset = (page - 1) * page_size
    alerts = await alert_manager.get_alerts(status=status, limit=page_size, offset=offset)
    return AlertListResponse(
        alerts=alerts,
        total=len(alerts),
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get alert statistics for dashboard KPIs."""
    from src.api.main import alert_manager
    if alert_manager is None:
        return StatsResponse(total=0, pending=0, approved=0, rejected=0, retroactive=0, critical=0)
    stats = await alert_manager.get_stats()
    return StatsResponse(**stats)


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    """Get a single alert by ID."""
    from src.api.main import alert_manager
    if alert_manager is None:
        raise HTTPException(404, "Alert manager not initialized")
    alerts = await alert_manager.get_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            return a
    raise HTTPException(404, f"Alert {alert_id} not found")
