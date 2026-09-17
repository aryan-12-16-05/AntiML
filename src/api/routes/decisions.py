"""
decisions.py — Alert review (Approve/Reject) API routes.
"""
from fastapi import APIRouter, HTTPException
from src.api.models_api import ReviewRequest, ReviewResponse

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


@router.post("/review", response_model=ReviewResponse)
async def review_alert(request: ReviewRequest):
    """
    Approve or reject an alert.
    If APPROVED: optionally add involved accounts to whitelist with TTL.
    """
    from src.api.main import alert_manager, whitelist

    if alert_manager is None:
        raise HTTPException(503, "Alert manager not initialized")

    updated = await alert_manager.update_alert_status(
        request.alert_id,
        request.action,
        request.note or ""
    )
    if not updated:
        raise HTTPException(404, f"Alert {request.alert_id} not found")

    wl_count = 0
    if request.action == "APPROVED" and whitelist and request.whitelist_accounts:
        wl_count = await whitelist.add(
            accounts=request.whitelist_accounts,
            alert_id=request.alert_id,
            reason=request.note or "Analyst approved",
            ttl_days=request.whitelist_ttl_days or 30,
        )

    return ReviewResponse(
        success=True,
        alert=updated,
        whitelist_added=wl_count,
        message=f"Alert {request.action} successfully. {wl_count} accounts whitelisted.",
    )


@router.get("/whitelist")
async def get_whitelist():
    """Get all active whitelist entries."""
    from src.api.main import whitelist
    if whitelist is None:
        return {"entries": []}
    entries = await whitelist.get_active_entries()
    return {"entries": entries}


@router.delete("/whitelist/{account}")
async def revoke_whitelist(account: str):
    """Manually revoke whitelist for an account."""
    from src.api.main import whitelist
    if whitelist is None:
        raise HTTPException(503, "Whitelist not initialized")
    removed = await whitelist.remove(account)
    return {"success": removed, "message": f"Account {account} removed from whitelist" if removed else "Not found"}
