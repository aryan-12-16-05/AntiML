"""
profiles.py — Customer profile API routes.
"""
from fastapi import APIRouter, Query
from typing import List

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.get("/{account}")
async def get_profile(account: str):
    """Get behavioral profile for an account."""
    from src.api.main import customer_profiler, whitelist
    if customer_profiler is None:
        return {"account": account, "error": "Profiler not initialized"}

    profile = customer_profiler.get_profile_summary(account)

    # Augment with whitelist info
    wl_active = False
    wl_expires = None
    if whitelist:
        wl_active = await whitelist.is_suppressed(account)
        entries = await whitelist.get_active_entries()
        for e in entries:
            if e["account"] == account:
                wl_expires = e["expires_at"]
                break

    return {
        **profile,
        "whitelist_active": wl_active,
        "whitelist_expires_at": wl_expires,
    }


@router.get("/")
async def search_profiles(
    query: str = Query("", description="Search by account or entity name"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search customer profiles."""
    from src.api.main import customer_profiler
    if customer_profiler is None:
        return {"profiles": []}

    query_lower = query.lower()
    results = [
        p for acc, p in list(customer_profiler.profiles.items())[:500]
        if query_lower in str(acc).lower() or query_lower in str(p.get("entity_name", "")).lower()
    ][:limit]
    return {"profiles": results}
