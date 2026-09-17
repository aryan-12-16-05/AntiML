"""
whitelist.py — Time-bounded analyst approval whitelist.
When an analyst approves an alert, the involved accounts are added
to a whitelist with a TTL (default 30 days).
After TTL expires, accounts return to full detection.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, DateTime, Boolean, Text
from sqlalchemy import select
from loguru import logger


class WhitelistBase(DeclarativeBase):
    pass


class WhitelistEntryORM(WhitelistBase):
    __tablename__ = "whitelist"

    id = Column(String, primary_key=True)
    account = Column(String, nullable=False, index=True)
    approved_by_alert_id = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    approved_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)


class Whitelist:
    """
    Time-bounded account whitelist.
    - Analyst approves an alert → all involved accounts added with TTL
    - During TTL: those accounts don't trigger new alerts
    - After TTL: accounts return to full detection automatically
    - TTL can be extended or revoked manually
    """

    def __init__(self, db_url: str = "sqlite+aiosqlite:///data/alerts.db",
                 default_ttl_days: int = 30):
        self.db_url = db_url
        self.default_ttl_days = default_ttl_days
        self.engine = None
        self.session_factory = None
        # In-memory cache for fast lookup
        self._cache: Dict[str, datetime] = {}

    async def init(self) -> None:
        """Initialize database and load active whitelist into cache."""
        self.engine = create_async_engine(self.db_url, echo=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(WhitelistBase.metadata.create_all)
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        await self._refresh_cache()
        logger.info(f"Whitelist initialized. Active entries: {len(self._cache)}")

    async def add(self, accounts: List[str], alert_id: str = None,
                  reason: str = "", ttl_days: Optional[int] = None) -> int:
        """
        Add accounts to whitelist with TTL.
        Returns number of new entries added.
        """
        ttl = ttl_days or self.default_ttl_days
        expires = datetime.utcnow() + timedelta(days=ttl)
        count = 0

        async with self.session_factory() as session:
            for acc in accounts:
                import uuid
                entry = WhitelistEntryORM(
                    id=str(uuid.uuid4()),
                    account=str(acc),
                    approved_by_alert_id=alert_id,
                    reason=reason,
                    approved_at=datetime.utcnow(),
                    expires_at=expires,
                    is_active=True,
                )
                session.add(entry)
                self._cache[str(acc)] = expires
                count += 1
            await session.commit()

        logger.info(f"Added {count} accounts to whitelist (TTL={ttl}d, expires={expires.date()})")
        return count

    async def remove(self, account: str) -> bool:
        """Manually remove account from whitelist (revoke approval)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(WhitelistEntryORM)
                .where(WhitelistEntryORM.account == str(account))
                .where(WhitelistEntryORM.is_active == True)
            )
            entries = result.scalars().all()
            for entry in entries:
                entry.is_active = False
            await session.commit()

        self._cache.pop(str(account), None)
        logger.info(f"Removed account {account} from whitelist.")
        return len(entries) > 0

    async def is_suppressed(self, *accounts: str) -> bool:
        """Check if any of the given accounts is in active whitelist."""
        now = datetime.utcnow()
        for acc in accounts:
            expires = self._cache.get(str(acc))
            if expires and expires > now:
                return True
        # Check expired entries in cache
        expired = [a for a, exp in self._cache.items() if exp <= now]
        for a in expired:
            del self._cache[a]
        return False

    async def get_active_entries(self) -> List[Dict]:
        """Get all currently active whitelist entries."""
        now = datetime.utcnow()
        async with self.session_factory() as session:
            result = await session.execute(
                select(WhitelistEntryORM)
                .where(WhitelistEntryORM.is_active == True)
                .where(WhitelistEntryORM.expires_at > now)
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "account": r.account,
                "alert_id": r.approved_by_alert_id,
                "reason": r.reason,
                "approved_at": r.approved_at.isoformat(),
                "expires_at": r.expires_at.isoformat(),
                "days_remaining": (r.expires_at - now).days,
            }
            for r in rows
        ]

    async def extend_ttl(self, account: str, extra_days: int) -> bool:
        """Extend TTL for an account."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(WhitelistEntryORM)
                .where(WhitelistEntryORM.account == str(account))
                .where(WhitelistEntryORM.is_active == True)
            )
            entries = result.scalars().all()
            for entry in entries:
                entry.expires_at = entry.expires_at + timedelta(days=extra_days)
                self._cache[str(account)] = entry.expires_at
            await session.commit()
        return len(entries) > 0

    async def _refresh_cache(self) -> None:
        """Reload active whitelist from DB into memory cache."""
        now = datetime.utcnow()
        async with self.session_factory() as session:
            result = await session.execute(
                select(WhitelistEntryORM)
                .where(WhitelistEntryORM.is_active == True)
                .where(WhitelistEntryORM.expires_at > now)
            )
            rows = result.scalars().all()
        self._cache = {r.account: r.expires_at for r in rows}
