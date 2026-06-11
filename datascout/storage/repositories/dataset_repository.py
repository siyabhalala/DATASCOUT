"""
datascout.storage.repositories.dataset_repository
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Repository pattern for dataset CRUD, upsert,
search, and analytics. Decouples business logic from ORM details.

SYSTEM DESIGN DECISIONS:

  1. WHY repository pattern (not direct ORM access)?
     - Business logic (crawler, scoring, API) never imports DatasetORM
     - To swap SQLite → PostgreSQL: change engine URL, zero other changes
     - To add Redis caching: wrap repository methods, zero change to callers
     - To test without DB: inject MockDatasetRepository, zero change to tests

  2. WHY upsert (not insert-or-ignore)?
     - INSERT OR IGNORE: silently drops metadata updates for existing records
     - INSERT OR REPLACE: deletes + reinserts (loses ingested_at, breaks FK refs)
     - Upsert (select + update or insert): preserves ingested_at, updates crawled_at
     - At crawl time: existing records get metadata refreshed, timestamps updated

  3. WHY flush() in create/upsert (not commit())?
     - Repository doesn't own the transaction — the caller does
     - Flush sends SQL to DB without committing the transaction
     - Caller can create multiple records in one atomic transaction then commit
     - If repository committed: multi-step crawl becomes N separate transactions

  4. WHY _orm_to_contract and _contract_to_orm as private static methods?
     - All ORM↔contract conversion in ONE place
     - Adding a new field: one change in _contract_to_orm, propagates everywhere
     - Without this: scattered field mappings across search/create/upsert methods

  5. WHY search_by_query returns RawDataset (not DatasetORM)?
     - Callers (agent, API) work with RawDataset — the contract layer
     - Returning ORM objects would leak storage implementation to business logic
     - Caller should never see DatasetORM outside storage/

FAILURE SCENARIOS HANDLED:
  - Record not found → None returned (not exception) for get_by_*
  - ORM → contract conversion failure → logged + record skipped (not crash)
  - Empty search results → empty list (not exception)
  - count_* on empty table → 0

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update, String
from sqlalchemy.ext.asyncio import AsyncSession

from datascout.contracts import RawDataset
from datascout.contracts.states import (
    DataDomain, DataFormat, LicenseType,
    StageStatus, normalize_domain, normalize_format, normalize_license,
)
from datascout.contracts.task_types import Modality, TaskType, normalize_modality, normalize_task_type
from ..models import DatasetORM

logger = logging.getLogger("datascout.storage.repositories.dataset_repository")


class DatasetRepository:
    """
    Async CRUD repository for DatasetORM.
    One instance per session — do not share across sessions.

    Usage:
        async with get_session() as session:
            repo = DatasetRepository(session)
            dataset = await repo.get_by_canonical_id("kaggle:titanic")
            await session.commit()
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ─────────────────────────────────────────────────────────────────────────
    # CREATE / UPSERT
    # ─────────────────────────────────────────────────────────────────────────

    async def create(self, dataset: RawDataset) -> DatasetORM:
        """
        Insert a new dataset record.
        Caller must call session.commit() after this to persist.

        Raises:
            IntegrityError: if (source, source_id) already exists — use upsert instead
        """
        orm = self._contract_to_orm(dataset)
        self._session.add(orm)
        await self._session.flush()  # Get auto-generated id without committing
        logger.debug(
            "dataset_created",
            extra={"canonical_id": dataset.canonical_id, "id": orm.id},
        )
        return orm

    async def upsert(self, dataset: RawDataset) -> tuple[DatasetORM, bool]:
        """
        Insert or update a dataset record based on (source, source_id).

        Returns:
            (orm_record, was_created): was_created=True if inserted, False if updated

        WHY two-step (select then insert/update):
        - Preserves ingested_at on update (keeps original discovery timestamp)
        - Updates crawled_at on every call (tracks last verification time)
        - Works across both SQLite and PostgreSQL without dialect-specific SQL
        """
        existing = await self.get_by_source(dataset.source, dataset.source_id)

        if existing is not None:
            # Update existing record — preserve ingested_at, update crawled_at
            self._update_orm(existing, dataset)
            await self._session.flush()
            logger.debug(
                "dataset_updated",
                extra={"canonical_id": dataset.canonical_id, "id": existing.id},
            )
            return existing, False
        else:
            orm = await self.create(dataset)
            return orm, True

    async def update_analysis(
        self,
        canonical_id: str,
        class_imbalance_score: Optional[float] = None,
        duplicate_percentage: Optional[float] = None,
        blur_quality_score: Optional[float] = None,
        suggested_fixes: Optional[list[str]] = None,
        analysis_version: str = "3.0.0",
    ) -> bool:
        """
        Update Phase 9 analysis fields for an existing record.
        Returns True if record found and updated, False if not found.
        Caller must commit.
        """
        result = await self._session.execute(
            select(DatasetORM).where(DatasetORM.canonical_id == canonical_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            logger.warning("update_analysis_not_found", extra={"canonical_id": canonical_id})
            return False

        if class_imbalance_score is not None:
            orm.class_imbalance_score = class_imbalance_score
        if duplicate_percentage is not None:
            orm.duplicate_percentage = duplicate_percentage
        if blur_quality_score is not None:
            orm.blur_quality_score = blur_quality_score
        if suggested_fixes is not None:
            orm.suggested_fixes = suggested_fixes

        orm.analysis_version = analysis_version
        orm.analyzed_at = datetime.now(timezone.utc)

        await self._session.flush()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # READ
    # ─────────────────────────────────────────────────────────────────────────

    async def get_by_id(self, record_id: int) -> Optional[RawDataset]:
        """Lookup by surrogate primary key."""
        orm = await self._session.get(DatasetORM, record_id)
        return self._orm_to_contract(orm) if orm else None

    async def get_by_canonical_id(self, canonical_id: str) -> Optional[RawDataset]:
        """Lookup by canonical_id (e.g. 'kaggle:titanic')."""
        result = await self._session.execute(
            select(DatasetORM).where(DatasetORM.canonical_id == canonical_id)
        )
        orm = result.scalar_one_or_none()
        return self._orm_to_contract(orm) if orm else None

    async def get_by_source(self, source: str, source_id: str) -> Optional[DatasetORM]:
        """
        Lookup raw ORM record by (source, source_id).
        Returns ORM (not contract) so upsert can modify it directly.
        """
        result = await self._session.execute(
            select(DatasetORM).where(
                DatasetORM.source == source,
                DatasetORM.source_id == source_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_fingerprint(self, fingerprint: str) -> Optional[RawDataset]:
        """Lookup by SHA-256 fingerprint — used for cross-source dedup check."""
        result = await self._session.execute(
            select(DatasetORM).where(DatasetORM.dataset_fingerprint == fingerprint)
        )
        orm = result.scalar_one_or_none()
        return self._orm_to_contract(orm) if orm else None

    async def search(
        self,
        *,
        query_text: Optional[str] = None,
        source: Optional[str] = None,
        data_domain: Optional[str] = None,
        min_completeness: float = 0.0,
        exclude_duplicates: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RawDataset]:
        """
        Multi-criteria search returning RawDataset list.

        WHY LIKE search (not FTS)?
        - SQLite doesn't have built-in full-text search (needs FTS5 extension)
        - LIKE '%query%' works on both SQLite and PostgreSQL without extensions
        - Phase 10 scoring uses semantic similarity for precision — this is recall
        - Upgrade path: add PostgreSQL tsvector index for production

        Args:
            query_text: Substring match against title + description_clean
            source: Filter by adapter source ("kaggle", "huggingface", "openml")
            data_domain: Filter by domain enum value
            min_completeness: Minimum metadata_completeness score
            exclude_duplicates: Skip records marked is_duplicate=True
            limit: Max records to return
            offset: For pagination
        """
        stmt = select(DatasetORM)

        if query_text:
            pattern = f"%{query_text.lower()}%"
            from sqlalchemy import or_, func as sa_func
            stmt = stmt.where(
                or_(
                    DatasetORM.title.ilike(pattern),
                    DatasetORM.description_clean.ilike(pattern),
                    DatasetORM.tags.cast(String).ilike(pattern),
                )
            )

        if source:
            stmt = stmt.where(DatasetORM.source == source)

        if data_domain:
            stmt = stmt.where(DatasetORM.data_domain == data_domain)

        if min_completeness > 0.0:
            stmt = stmt.where(DatasetORM.metadata_completeness >= min_completeness)

        if exclude_duplicates:
            stmt = stmt.where(DatasetORM.is_duplicate == False)  # noqa: E712

        stmt = (
            stmt
            .order_by(DatasetORM.metadata_completeness.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        datasets: list[RawDataset] = []
        for orm in rows:
            try:
                ds = self._orm_to_contract(orm)
                if ds:
                    datasets.append(ds)
            except Exception as e:
                logger.warning(
                    "orm_to_contract_failed",
                    extra={"id": orm.id, "canonical_id": orm.canonical_id, "error": str(e)},
                )
                continue

        return datasets

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────

    async def count_total(self, exclude_duplicates: bool = True) -> int:
        """Total dataset count in the index."""
        stmt = select(func.count(DatasetORM.id))
        if exclude_duplicates:
            stmt = stmt.where(DatasetORM.is_duplicate == False)  # noqa: E712
        result = await self._session.execute(stmt)
        return result.scalar_one() or 0

    async def count_by_source(self) -> dict[str, int]:
        """Count datasets per source adapter."""
        result = await self._session.execute(
            select(DatasetORM.source, func.count(DatasetORM.id))
            .where(DatasetORM.is_duplicate == False)  # noqa: E712
            .group_by(DatasetORM.source)
        )
        return {row[0]: row[1] for row in result.all()}

    async def count_by_domain(self) -> dict[str, int]:
        """Count datasets per data domain."""
        result = await self._session.execute(
            select(DatasetORM.data_domain, func.count(DatasetORM.id))
            .where(
                DatasetORM.is_duplicate == False,  # noqa: E712
                DatasetORM.data_domain.is_not(None),
            )
            .group_by(DatasetORM.data_domain)
        )
        return {row[0]: row[1] for row in result.all()}

    async def avg_completeness_by_source(self) -> dict[str, float]:
        """Average metadata_completeness per source."""
        result = await self._session.execute(
            select(
                DatasetORM.source,
                func.avg(DatasetORM.metadata_completeness),
            )
            .where(DatasetORM.is_duplicate == False)  # noqa: E712
            .group_by(DatasetORM.source)
        )
        return {row[0]: round(row[1] or 0.0, 3) for row in result.all()}

    async def get_stale_records(
        self, older_than: datetime, limit: int = 1000
    ) -> list[str]:
        """
        Return canonical_ids of records not crawled since older_than.
        Used by crawler to refresh stale metadata.
        """
        result = await self._session.execute(
            select(DatasetORM.canonical_id)
            .where(DatasetORM.crawled_at < older_than)
            .order_by(DatasetORM.crawled_at.asc())
            .limit(limit)
        )
        return [row[0] for row in result.all()]

    # ─────────────────────────────────────────────────────────────────────────
    # ORM ↔ CONTRACT CONVERSION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _contract_to_orm(dataset: RawDataset) -> DatasetORM:
        """Convert RawDataset contract → DatasetORM for insertion."""
        now = datetime.now(timezone.utc)
        return DatasetORM(
            dataset_fingerprint=dataset.dataset_fingerprint,
            canonical_id=dataset.canonical_id,
            is_duplicate=dataset.is_duplicate,
            duplicate_of=dataset.duplicate_of,
            source=dataset.source,
            source_id=dataset.source_id,
            source_url=dataset.source_url,
            ingestion_version=dataset.ingestion_version,
            schema_version=dataset.schema_version,
            title=dataset.title,
            description=dataset.description,
            description_clean=dataset.description_clean,
            description_short=dataset.description_short,
            tags=dataset.tags,
            tags_primary=dataset.tags_primary,
            author=dataset.author,
            language=dataset.language,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            file_size_bytes=dataset.file_size_bytes,
            data_format=dataset.data_format.value if dataset.data_format else None,
            license_type=dataset.license_type.value if dataset.license_type else None,
            data_domain=dataset.data_domain.value if dataset.data_domain else None,
            task_types=[t.value for t in dataset.task_types] if dataset.task_types else [],
            modalities=[m.value for m in dataset.modalities] if dataset.modalities else [],
            download_count=dataset.download_count,
            upvote_count=dataset.upvote_count,
            last_updated=dataset.last_updated,
            metadata_completeness=dataset.metadata_completeness,
            has_description=dataset.has_description,
            has_schema_info=dataset.has_schema_info,
            has_size_info=dataset.has_size_info,
            has_license_info=dataset.has_license_info,
            description_length=dataset.description_length,
            tag_count=dataset.tag_count,
            missing_fields_count=dataset.missing_fields_count,
            extra=dataset.extra,
            ingested_at=now,
            crawled_at=now,
        )

    @staticmethod
    def _update_orm(orm: DatasetORM, dataset: RawDataset) -> None:
        """
        Update mutable fields on an existing ORM record.
        NEVER updates ingested_at (preserves original discovery time).
        ALWAYS updates crawled_at (marks as recently verified).
        """
        orm.dataset_fingerprint = dataset.dataset_fingerprint
        orm.title = dataset.title
        orm.description = dataset.description
        orm.description_clean = dataset.description_clean
        orm.description_short = dataset.description_short
        orm.tags = dataset.tags
        orm.tags_primary = dataset.tags_primary
        orm.author = dataset.author
        orm.language = dataset.language
        orm.row_count = dataset.row_count
        orm.column_count = dataset.column_count
        orm.file_size_bytes = dataset.file_size_bytes
        orm.data_format = dataset.data_format.value if dataset.data_format else None
        orm.license_type = dataset.license_type.value if dataset.license_type else None
        orm.data_domain = dataset.data_domain.value if dataset.data_domain else None
        orm.task_types = [t.value for t in dataset.task_types] if dataset.task_types else []
        orm.modalities = [m.value for m in dataset.modalities] if dataset.modalities else []
        orm.download_count = dataset.download_count
        orm.upvote_count = dataset.upvote_count
        orm.last_updated = dataset.last_updated
        orm.metadata_completeness = dataset.metadata_completeness
        orm.has_description = dataset.has_description
        orm.has_schema_info = dataset.has_schema_info
        orm.has_size_info = dataset.has_size_info
        orm.has_license_info = dataset.has_license_info
        orm.description_length = dataset.description_length
        orm.tag_count = dataset.tag_count
        orm.missing_fields_count = dataset.missing_fields_count
        orm.is_duplicate = dataset.is_duplicate
        orm.duplicate_of = dataset.duplicate_of
        orm.schema_version = dataset.schema_version
        orm.extra = dataset.extra
        orm.crawled_at = datetime.now(timezone.utc)

    @staticmethod
    def _orm_to_contract(orm: Optional[DatasetORM]) -> Optional[RawDataset]:
        """
        Convert DatasetORM → RawDataset contract.
        Returns None if ORM is None (allows safe chaining).
        """
        if orm is None:
            return None
        try:
            from contracts.models import (
                compute_fingerprint, compute_primary_tags, compute_completeness,
            )
            return RawDataset(
                dataset_fingerprint=orm.dataset_fingerprint,
                canonical_id=orm.canonical_id,
                is_duplicate=orm.is_duplicate,
                duplicate_of=orm.duplicate_of,
                source=orm.source,
                source_url=orm.source_url,
                source_id=orm.source_id,
                ingestion_timestamp=orm.ingested_at or datetime.now(timezone.utc),
                pipeline_version="3.0.0",
                ingestion_version=orm.ingestion_version,
                schema_version=orm.schema_version or "3.0.0",
                title=orm.title or "Untitled",
                description=orm.description or "",
                tags=orm.tags or [],
                tags_primary=orm.tags_primary or [],
                row_count=orm.row_count,
                column_count=orm.column_count,
                file_size_bytes=orm.file_size_bytes,
                data_format=normalize_format(orm.data_format) if orm.data_format else None,
                license_type=normalize_license(orm.license_type) if orm.license_type else None,
                data_domain=normalize_domain(orm.data_domain) if orm.data_domain else None,
                task_types=[normalize_task_type(t) for t in (orm.task_types or [])],
                modalities=[normalize_modality(m) for m in (orm.modalities or [])],
                language=orm.language,
                last_updated=orm.last_updated,
                download_count=orm.download_count,
                upvote_count=orm.upvote_count,
                author=orm.author,
                missing_fields_count=orm.missing_fields_count,
                metadata_completeness=orm.metadata_completeness,
                has_description=orm.has_description,
                has_schema_info=orm.has_schema_info,
                has_size_info=orm.has_size_info,
                has_license_info=orm.has_license_info,
                description_length=orm.description_length,
                tag_count=orm.tag_count,
                extra=orm.extra or {},
            )
        except Exception as e:
            logger.error(
                "orm_to_contract_error",
                extra={"orm_id": orm.id if orm else None, "error": str(e)},
                exc_info=True,
            )
            return None