"""
datascout.storage.database
──────────────────────────────────────────────────────
Simple SQLite database initialization with graceful degradation.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("datascout.storage.database")

_engine = None
_initialized = False


async def init_database() -> None:
    """Initialize the database. Degrades gracefully on failure."""
    global _engine, _initialized
    if _initialized:
        return

    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import declarative_base
        from datascout.infrastructure.config.settings import get_settings

        settings = get_settings()
        db_url = settings.database_url

        _engine = create_async_engine(db_url, echo=False)
        _initialized = True
        logger.info("database_engine_created", extra={"url": db_url.split("///")[0]})
    except Exception as exc:
        logger.warning("database_init_failed", extra={"error": str(exc)})
        _initialized = True  # Don't retry on every request
        raise


async def close_database() -> None:
    """Close database connections on shutdown."""
    global _engine
    if _engine is not None:
        try:
            await _engine.dispose()
            logger.info("database_closed")
        except Exception as exc:
            logger.warning("database_close_failed", extra={"error": str(exc)})


def get_engine():
    return _engine
