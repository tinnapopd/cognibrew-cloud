import os
from datetime import datetime, timezone
from typing import Optional

from cognibrew.logger import Logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = Logger().get_logger()

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://cognibrew:cognibrew@postgres:5432/cognibrew",
)

engine = create_async_engine(_DATABASE_URL, echo=False, pool_size=5)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ConfidenceHistoryRow(Base):
    __tablename__ = "confidence_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
    threshold: Mapped[float]
    f1_score: Mapped[float]
    precision: Mapped[float]
    recall: Mapped[float]
    version: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        unique=True, nullable=False, index=True
    )
    current_threshold: Mapped[Optional[float]] = mapped_column(nullable=True)
    current_version: Mapped[int] = mapped_column(default=0)
    notes: Mapped[Optional[str]] = mapped_column(nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised")


async def save_confidence(
    threshold: float,
    f1_score: float,
    precision: float,
    recall: float,
    username: Optional[str] = None,
) -> int:
    """Store a new confidence history record. Returns the new version number."""
    async with async_session() as session:
        # Get next version
        stmt = select(func.coalesce(func.max(ConfidenceHistoryRow.version), 0))
        if username:
            stmt = stmt.where(ConfidenceHistoryRow.username == username)
        result = await session.execute(stmt)
        current_max = result.scalar() or 0
        new_version = current_max + 1

        row = ConfidenceHistoryRow(
            username=username,
            threshold=threshold,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            version=new_version,
        )
        session.add(row)

        # Upsert user profile if username is provided
        if username:
            existing = await session.execute(
                select(UserProfile).where(UserProfile.username == username)
            )
            profile = existing.scalar_one_or_none()
            if profile:
                profile.current_threshold = threshold
                profile.current_version = new_version
                profile.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    UserProfile(
                        username=username,
                        current_threshold=threshold,
                        current_version=new_version,
                    )
                )

        await session.commit()
        return new_version


async def get_latest_threshold(username: Optional[str] = None) -> dict | None:
    """Return the latest threshold (optionally per user)."""
    async with async_session() as session:
        stmt = (
            select(ConfidenceHistoryRow)
            .order_by(ConfidenceHistoryRow.version.desc())
            .limit(1)
        )
        if username:
            stmt = stmt.where(ConfidenceHistoryRow.username == username)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if not row:
            return None
        return {
            "threshold": row.threshold,
            "version": row.version,
            "f1_score": row.f1_score,
            "username": row.username,
            "updated_at": row.created_at.isoformat()
            if row.created_at
            else None,
        }


async def get_user_threshold(username: str) -> dict | None:
    """Return the latest threshold specifically for a user."""
    return await get_latest_threshold(username=username)
