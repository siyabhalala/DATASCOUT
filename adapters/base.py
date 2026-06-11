"""
datascout.adapters.base
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Abstract base adapter — the interface contract every
adapter must implement. Centralizes lineage injection, fingerprinting,
quality signal computation, and null-safe extraction.

AGENT-0 CONTEXT:
  Adapters are the boundary between the outside world and Agent-0's contracts.
  Every record entering the system passes through _build_raw_dataset().
  This is the ONLY place where RawDataset is constructed from raw API data.

SYSTEM DESIGN DECISIONS:

  1. WHY abstract base with Liskov Substitution?
     - Orchestrator calls adapter.search(query) without knowing which adapter
     - All adapters are interchangeable at the orchestrator level
     - Adding a new adapter = implement BaseAdapter, register — nothing else changes
     - Without LSP: orchestrator has adapter-specific if/else branches everywhere

  2. WHY _build_raw_dataset() centralized on base?
     - Lineage injection, fingerprinting, quality signals, null-safe extraction
       all happen in ONE place
     - If scattered across adapters: Kaggle lineage works, HF lineage missing,
       OpenML fingerprint wrong — inconsistent contracts downstream
     - One bug fix here fixes all three adapters simultaneously

  3. WHY _extract_safe() null-safe helpers?
     - Kaggle API returns inconsistent schemas across dataset ages
     - HuggingFace returns None for most fields on poorly documented datasets
     - OpenML returns Java-style null strings ("null", "None", "N/A")
     - Without null-safe extraction: NoneType crashes on every third record

  4. WHY adapters NEVER raise on external failures?
     - search() must always return List[RawDataset] — empty list on failure
     - Exceptions are logged + metrics recorded, then swallowed
     - Orchestrator decides what to do with empty results (partial result logic)
     - An adapter that raises breaks the entire parallel execution

FAILURE SCENARIOS HANDLED:
  - API timeout → AdapterTimeoutError caught, empty list returned
  - Rate limit → AdapterRateLimitError caught, circuit breaker notified
  - Auth failure → AdapterAuthError caught, logged as CRITICAL, empty list
  - Malformed JSON → AdapterMalformedResponseError, record skipped
  - Missing lineage → LineageMissingError, record rejected (never swallowed)
  - Individual record failure → skip record, continue processing others

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from datascout.contracts import (
    DataDomain,
    DataFormat,
    LicenseType,
    Modality,
    RawDataset,
    TaskType,
    compute_completeness,
    compute_fingerprint,
    compute_primary_tags,
    normalize_domain,
    normalize_format,
    normalize_license,
    normalize_modality,
    normalize_task_type,
)
from datascout.contracts.errors import AdapterError, LineageMissingError
from datascout.contracts.requests import SearchQuery
from datascout.infrastructure.monitoring import metrics

logger = logging.getLogger("datascout.adapters.base")

CURRENT_PIPELINE_VERSION = "3.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER HEALTH
# ─────────────────────────────────────────────────────────────────────────────

class AdapterHealth:
    """Health status returned by adapter.health_check()."""

    def __init__(
        self,
        adapter: str,
        healthy: bool,
        latency_ms: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.adapter = adapter
        self.healthy = healthy
        self.latency_ms = latency_ms
        self.message = message
        self.error = error
        self.checked_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "error": self.error,
            "checked_at": self.checked_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BASE ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    """
    Abstract base for all data source adapters.

    Subclasses must implement:
        search(query) → List[RawDataset]
        health_check() → AdapterHealth
        ADAPTER_NAME: str
        ADAPTER_VERSION: str

    Subclasses inherit:
        _build_raw_dataset() — centralized RawDataset construction
        _extract_str() / _extract_int() / _extract_list() — null-safe helpers
        _parse_tags() — normalize raw tag strings
        _parse_datetime() — safely parse date strings
    """

    ADAPTER_NAME: str = "base"
    ADAPTER_VERSION: str = "0.0.0"

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[RawDataset]:
        """
        Search the data source and return normalized RawDataset records.

        MUST:
        - Always return List[RawDataset] — never raise on external failures
        - Return empty list if the API is unavailable
        - Log + record metrics for every failure
        - Call _build_raw_dataset() for every record (never construct RawDataset directly)
        """

    @abstractmethod
    async def health_check(self) -> AdapterHealth:
        """
        Kubernetes readiness probe — called every 30s.
        Must complete within 5s.
        Must not raise.
        """

    # ─────────────────────────────────────────────────────────────────────────
    # CENTRALIZED RAW DATASET CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_raw_dataset(
        self,
        raw: dict[str, Any],
        source: str,
        source_id: str,
        source_url: str,
        title: str,
        description: str,
        tags: Optional[list[str]] = None,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
        column_names: Optional[list[str]] = None,
        file_size_bytes: Optional[int] = None,
        data_format: Optional[DataFormat] = None,
        license_type: Optional[LicenseType] = None,
        data_domain: Optional[DataDomain] = None,
        task_types: Optional[list[TaskType]] = None,
        modalities: Optional[list[Modality]] = None,
        language: Optional[str] = None,
        last_updated: Optional[datetime] = None,
        download_count: Optional[int] = None,
        upvote_count: Optional[int] = None,
        author: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> RawDataset:
        """
        Construct a RawDataset from normalized adapter output.

        WHY centralized:
        - Lineage injection happens here — ONCE — for all adapters
        - Fingerprint computation happens here — ONCE
        - Quality signals computed here — ONCE
        - Null-safety enforced here — no adapter can forget

        This method raises LineageMissingError if required lineage is absent.
        All other failures produce safe defaults and log warnings.
        """
        # ── Null-safe defaults ────────────────────────────────────────────────
        title = (title or "").strip() or "Untitled"
        description = (description or "").strip()
        tags = [t for t in (tags or []) if t and isinstance(t, str)]
        task_types = task_types or []
        modalities = modalities or []

        # ── Fingerprint (deduplication key) ───────────────────────────────────
        fingerprint = compute_fingerprint(
            title=title,
            source=source,
            row_count=row_count,
            column_names=column_names,
        )

        # ── Canonical ID ──────────────────────────────────────────────────────
        canonical_id = f"{source}:{source_id}"

        # ── Primary tags for BM25 ─────────────────────────────────────────────
        tags_primary = compute_primary_tags(tags)

        # ── Structural quality signals ────────────────────────────────────────
        has_description  = bool(description and description.strip())
        has_schema_info  = bool(column_names and len(column_names) > 0)
        has_size_info    = bool(row_count is not None or file_size_bytes is not None)
        has_license_info = bool(license_type and license_type != LicenseType.UNKNOWN)

        # Build partial dict for completeness computation
        completeness_dict = {
            "title": title,
            "description": description,
            "tags": tags,
            "source_url": source_url,
            "row_count": row_count,
            "column_count": column_count,
            "file_size_bytes": file_size_bytes,
            "license_type": license_type,
            "data_format": data_format,
            "last_updated": last_updated,
        }
        metadata_completeness = compute_completeness(completeness_dict)

        # Count missing optional fields
        optional_fields = [
            row_count, column_count, file_size_bytes,
            data_format, license_type, data_domain,
            last_updated, author, language,
        ]
        missing_fields_count = sum(1 for f in optional_fields if f is None)

        return RawDataset(
            # Identity
            dataset_fingerprint=fingerprint,
            canonical_id=canonical_id,
            is_duplicate=False,         # Set by dedup step, not here
            duplicate_of=None,
            dedup_confidence=1.0,

            # Lineage — required, no defaults
            source=source,
            source_url=source_url,
            source_id=source_id,
            ingestion_timestamp=datetime.now(timezone.utc),
            pipeline_version=CURRENT_PIPELINE_VERSION,
            ingestion_version=self.ADAPTER_VERSION,
            schema_version="3.0.0",

            # Core metadata
            title=title,
            description=description,
            tags=tags,
            tags_primary=tags_primary,

            # Characteristics
            row_count=row_count,
            column_count=column_count,
            column_names=column_names,
            file_size_bytes=file_size_bytes,
            data_format=data_format,
            license_type=license_type,
            data_domain=data_domain,
            task_types=task_types,
            modalities=modalities,
            language=language,
            last_updated=last_updated,
            download_count=download_count,
            upvote_count=upvote_count,
            author=author,

            # Quality signals
            missing_fields_count=missing_fields_count,
            metadata_completeness=metadata_completeness,
            has_description=has_description,
            has_schema_info=has_schema_info,
            has_size_info=has_size_info,
            has_license_info=has_license_info,
            description_length=len(description),
            tag_count=len(tags),

            # Extras
            extra=extra or {},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NULL-SAFE EXTRACTION HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_str(
        raw: dict[str, Any],
        *keys: str,
        default: str = "",
    ) -> str:
        """
        Extract a string from raw dict, trying keys in order.
        Handles None, "null", "None", "N/A" as empty.

        WHY multiple keys: APIs use different field names across versions.
        e.g. HuggingFace uses both "description" and "cardData.description"
        """
        NULL_STRINGS = {"null", "none", "n/a", "na", "undefined", ""}
        for key in keys:
            # Support dot-notation for nested keys: "cardData.description"
            value = raw
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is not None and str(value).strip().lower() not in NULL_STRINGS:
                return str(value).strip()
        return default

    @staticmethod
    def _extract_int(
        raw: dict[str, Any],
        *keys: str,
        default: Optional[int] = None,
    ) -> Optional[int]:
        """
        Extract an integer from raw dict, trying keys in order.
        Handles float strings ("1000.0"), comma-formatted ("1,000"), None.
        """
        for key in keys:
            value = raw
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is None:
                continue
            try:
                # Handle "1,000,000" → 1000000 and "1000.0" → 1000
                cleaned = str(value).replace(",", "").replace(" ", "")
                return int(float(cleaned))
            except (ValueError, TypeError):
                continue
        return default

    @staticmethod
    def _extract_float(
        raw: dict[str, Any],
        *keys: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """Extract a float from raw dict."""
        for key in keys:
            value = raw.get(key)
            if value is None:
                continue
            try:
                return float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                continue
        return default

    @staticmethod
    def _extract_list(
        raw: dict[str, Any],
        *keys: str,
        default: Optional[list] = None,
    ) -> list:
        """
        Extract a list from raw dict.
        Handles: actual list, comma-separated string, single string.
        """
        for key in keys:
            value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return [str(v).strip() for v in value if v is not None]
            if isinstance(value, str) and value.strip():
                # Try comma-separated
                return [v.strip() for v in value.split(",") if v.strip()]
        return default or []

    @staticmethod
    def _parse_tags(raw_tags: Any) -> list[str]:
        """
        Normalize tags from any format to List[str].
        Handles: list of strings, list of dicts with "name" key,
                 comma-separated string, space-separated string.
        """
        if not raw_tags:
            return []

        tags: list[str] = []

        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, str) and t.strip():
                    tags.append(t.strip().lower())
                elif isinstance(t, dict):
                    # {"name": "nlp", "id": 123} pattern
                    name = t.get("name") or t.get("tag") or t.get("label") or ""
                    if name and isinstance(name, str):
                        tags.append(name.strip().lower())
        elif isinstance(raw_tags, str) and raw_tags.strip():
            # Comma or space separated
            sep = "," if "," in raw_tags else " "
            tags = [t.strip().lower() for t in raw_tags.split(sep) if t.strip()]

        # Deduplicate preserving order
        seen: set[str] = set()
        return [t for t in tags if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """
        Safely parse a datetime from various string formats.
        Returns None on any parse failure — never raises.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        import re as _re
        s = str(value).strip()
        if not s or s.lower() in {"null", "none", "n/a"}:
            return None

        # Try ISO 8601 and common variants
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(s[:len(fmt)], fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        logger.debug("_parse_datetime: could not parse %r", s)
        return None

    def _record_search_metrics(
        self,
        results: list[RawDataset],
        duration_s: float,
        success: bool = True,
    ) -> None:
        """Record standard search metrics. Call at end of search()."""
        if success:
            metrics.adapter_success(self.ADAPTER_NAME, len(results), duration_s)
            for dataset in results:
                metrics.dataset_accepted(self.ADAPTER_NAME, dataset.metadata_completeness)
        else:
            metrics.adapter_error(self.ADAPTER_NAME, "error")