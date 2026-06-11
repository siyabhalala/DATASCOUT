"""
datascout.storage.repositories.search_repository
──────────────────────────────────────────────────
Persistent storage for search history using SQLite via ``aiosqlite``.

``SearchRepository`` records every search request (query, result count,
sources used, latency) to a local SQLite database.  It is used by the API to
surface search analytics and by the agent to estimate query difficulty based
on historical success rates.

Resilience guarantees
─────────────────────
All public methods return safe defaults on any error (``False``, ``[]``,
``0``) and **never raise** to the caller.  The database is created lazily on
first write, so a missing or inaccessible DB file degrades gracefully without
crashing the API at startup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

__all__ = [
    "SearchRecord",
    "SearchRepository",
]

_TABLE_NAME = "search_history"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query               TEXT    NOT NULL,
    results_count       INTEGER NOT NULL DEFAULT 0,
    sources_used        TEXT    NOT NULL DEFAULT '[]',
    confidence          TEXT    NOT NULL DEFAULT 'unknown',
    processing_time_ms  INTEGER NOT NULL DEFAULT 0,
    searched_at         TEXT    NOT NULL,
    request_id          TEXT    NOT NULL DEFAULT ''
);
"""

_INSERT_SQL = f"""
INSERT INTO {_TABLE_NAME}
    (query, results_count, sources_used, confidence, processing_time_ms,
     searched_at, request_id)
VALUES
    (?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_SQL = f"""
SELECT query, results_count, sources_used, confidence, processing_time_ms,
       searched_at, request_id
FROM {_TABLE_NAME}
ORDER BY id DESC
LIMIT ?;
"""

_COUNT_SQL = f"SELECT COUNT(*) FROM {_TABLE_NAME};"


@dataclass
class SearchRecord:
    """A single search event persisted to the history table.

    Attributes
    ----------
    query:
        The raw search query string submitted by the user.
    results_count:
        Number of results returned for this query.
    sources_used:
        List of adapter/source names that were queried (e.g.
        ``["kaggle", "huggingface"]``).
    confidence:
        The pipeline's confidence label for these results
        (``"high"``, ``"medium"``, ``"low"``).
    processing_time_ms:
        Total wall-clock latency from request received to response sent,
        in milliseconds.
    searched_at:
        UTC datetime when the search was performed.
    request_id:
        Unique identifier for this request (UUID or similar), useful for
        correlating with application logs.
    """

    query: str
    results_count: int
    sources_used: List[str] = field(default_factory=list)
    confidence: str = "unknown"
    processing_time_ms: int = 0
    searched_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    request_id: str = ""


class SearchRepository:
    """Async repository for reading and writing search history records.

    Uses ``aiosqlite`` for non-blocking SQLite I/O.  The database file and
    table are created automatically on the first write.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to ``"./datascout.db"``
        relative to the process working directory.

    Example
    -------
    >>> repo = SearchRepository(db_path="./data/datascout.db")
    >>> record = SearchRecord(
    ...     query="image classification dataset",
    ...     results_count=5,
    ...     sources_used=["kaggle", "huggingface"],
    ...     confidence="high",
    ...     processing_time_ms=340,
    ...     request_id="req-abc123",
    ... )
    >>> await repo.save_search(record)
    True
    """

    def __init__(self, db_path: str = "./datascout.db") -> None:
        """Initialise the repository.

        Parameters
        ----------
        db_path:
            Filesystem path for the SQLite database file.
        """
        self._db_path: str = db_path
        self._table_ensured: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def save_search(self, record: SearchRecord) -> bool:
        """Persist *record* to the ``search_history`` table.

        Parameters
        ----------
        record:
            The search record to save.

        Returns
        -------
        bool:
            ``True`` on success, ``False`` on any error.  Never raises.
        """
        try:
            await self._ensure_table()
            searched_at_str = record.searched_at.isoformat() if record.searched_at else datetime.now(tz=timezone.utc).isoformat()
            sources_json = json.dumps(record.sources_used or [])
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    _INSERT_SQL,
                    (
                        record.query,
                        record.results_count,
                        sources_json,
                        record.confidence,
                        record.processing_time_ms,
                        searched_at_str,
                        record.request_id,
                    ),
                )
                await db.commit()
            logger.debug(
                "search_record_saved",
                extra={"query": record.query[:80], "request_id": record.request_id},
            )
            return True
        except Exception as exc:
            logger.warning(
                "search_repository_save_failed",
                extra={"error": str(exc), "db_path": self._db_path},
            )
            return False

    async def get_recent(self, limit: int = 10) -> List[SearchRecord]:
        """Return the *limit* most recent search records.

        Parameters
        ----------
        limit:
            Maximum number of records to return.  Defaults to 10.

        Returns
        -------
        list[SearchRecord]:
            Records ordered most-recent first.  Empty list on any error.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(_SELECT_RECENT_SQL, (limit,)) as cursor:
                    rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        except Exception as exc:
            logger.warning(
                "search_repository_get_recent_failed",
                extra={"error": str(exc), "db_path": self._db_path},
            )
            return []

    async def get_count(self) -> int:
        """Return the total number of search records in the table.

        Returns
        -------
        int:
            Total count, or ``0`` on any error.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(_COUNT_SQL) as cursor:
                    row = await cursor.fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning(
                "search_repository_count_failed",
                extra={"error": str(exc), "db_path": self._db_path},
            )
            return 0

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _ensure_table(self) -> None:
        """Create the ``search_history`` table if it does not exist.

        Called lazily on the first write so that a missing DB file is created
        only when needed.  Sets ``self._table_ensured = True`` after success
        to avoid redundant ``CREATE TABLE IF NOT EXISTS`` calls.
        """
        if self._table_ensured:
            return
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(_CREATE_TABLE_SQL)
                await db.commit()
            self._table_ensured = True
            logger.debug(
                "search_history_table_ensured",
                extra={"db_path": self._db_path},
            )
        except Exception as exc:
            logger.warning(
                "search_repository_ensure_table_failed",
                extra={"error": str(exc), "db_path": self._db_path},
            )
            # Don't set _table_ensured — retry on next call
            raise  # re-raise so save_search can catch and return False

    def _row_to_record(self, row: aiosqlite.Row) -> SearchRecord:
        """Convert an ``aiosqlite.Row`` to a :class:`SearchRecord`.

        Parameters
        ----------
        row:
            A row returned by a SELECT query with columns in the order defined
            by ``_SELECT_RECENT_SQL``.

        Returns
        -------
        SearchRecord:
            Reconstructed record object.
        """
        try:
            sources_used: list[str] = json.loads(row["sources_used"] or "[]")
        except (json.JSONDecodeError, TypeError):
            sources_used = []

        try:
            searched_at = datetime.fromisoformat(row["searched_at"])
            if searched_at.tzinfo is None:
                searched_at = searched_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            searched_at = datetime.now(tz=timezone.utc)

        return SearchRecord(
            query=row["query"] or "",
            results_count=int(row["results_count"] or 0),
            sources_used=sources_used,
            confidence=row["confidence"] or "unknown",
            processing_time_ms=int(row["processing_time_ms"] or 0),
            searched_at=searched_at,
            request_id=row["request_id"] or "",
        )