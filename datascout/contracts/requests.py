"""
datascout.contracts.requests
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Input contracts — validates and normalizes all inputs
before they enter the pipeline.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY SearchFilters separate from SearchPreferences?
     - Filters are HARD constraints: license_type=CC0 → never return non-CC0 results
     - Preferences are SOFT signals: prefer_recent → boost recent, don't exclude old
     - Mixing them makes ranking logic ambiguous: constraint or bias?
     - Separate objects: adapters apply filters as boolean gates,
                         ranking agent applies preferences as score adjustments

  2. WHY auto-generate query_id on construction?
     - query_id is the distributed trace anchor for the entire pipeline
     - Every log entry, metric, pipeline stage, response carries this ID
     - Generating at construction ensures it exists before any logging happens
     - No "unknown" request IDs anywhere in traces

  3. WHY sanitize raw_query at field validation time?
     - Injection-safe: strip leading/trailing whitespace
     - Length-capped at 500 chars: prevents DoS via massive queries
     - Normalized early: one sanitization point, not scattered across pipeline

  4. WHY max_results clamped to [1, 50]?
     - Fewer than 1: nonsensical
     - More than 50: downstream ranking + LLM explanation becomes expensive
     - 50 results × ~2KB each = 100KB per response — acceptable ceiling

FAILURE SCENARIOS HANDLED:
  - raw_query > 500 chars → QueryTooLongError (STRICT) / truncate+warn (GRACEFUL)
  - max_results outside [1, 50] → clamped to bounds + log
  - Invalid enum values in filters → ValidationError
  - Empty raw_query → ValidationError (always — cannot search for nothing)

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add language filter to SearchFilters
  Breaking: v4.0.0 — rename raw_query → query_text

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .errors import QueryTooLongError, ValidationError
from .states import DataDomain, DataFormat, LicenseType, ValidationMode, get_validation_mode
from .task_types import Modality, TaskType

logger = logging.getLogger("datascout.contracts.requests")

MAX_QUERY_CHARS   = 500
MAX_RESULTS_FLOOR = 1
MAX_RESULTS_CAP   = 50


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH FILTERS — hard constraints applied as boolean gates
# ─────────────────────────────────────────────────────────────────────────────

class SearchFilters(BaseModel):
    """
    Hard constraints on search results.
    Every filter is a boolean gate — if set, datasets that don't match are excluded.

    WHY hard vs soft distinction:
    - license_type=CC0 is non-negotiable for commercial projects
    - min_rows=10000 is a training data requirement — less is useless
    - These cannot be treated as "prefer" — they are "must"
    """

    model_config = ConfigDict(frozen=True)

    # Dataset size constraints
    min_rows: Optional[int] = None              # Minimum row count
    max_rows: Optional[int] = None              # Maximum row count
    min_columns: Optional[int] = None           # Minimum column count

    # Format and type constraints
    data_formats: list[DataFormat] = []         # Empty = no format filter
    domains: list[DataDomain] = []              # Empty = all domains
    license_types: list[LicenseType] = []       # Empty = all licenses
    task_types: list[TaskType] = []             # Empty = all tasks
    modalities: list[Modality] = []             # Empty = all modalities

    # Quality constraints
    min_completeness: Optional[float] = None    # Minimum metadata_completeness
    require_description: bool = False           # Must have non-empty description
    require_schema_info: bool = False           # Must have column names
    require_license_info: bool = False          # Must have known license

    # Language
    language: Optional[str] = None             # ISO 639-1 code e.g. "en"

    # Recency
    updated_after: Optional[datetime] = None   # Only datasets updated after this date

    @field_validator("min_completeness")
    @classmethod
    def validate_completeness(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValidationError(
                message=f"min_completeness must be in [0.0, 1.0], got {v}",
                field="min_completeness",
                invalid_value=v,
                constraint="[0.0, 1.0]",
            )
        return v

    @model_validator(mode="after")
    def validate_row_range(self) -> "SearchFilters":
        if (
            self.min_rows is not None
            and self.max_rows is not None
            and self.min_rows > self.max_rows
        ):
            raise ValidationError(
                message=f"min_rows ({self.min_rows}) cannot exceed max_rows ({self.max_rows})",
                field="min_rows",
                constraint="min_rows <= max_rows",
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH PREFERENCES — soft signals applied as ranking adjustments
# ─────────────────────────────────────────────────────────────────────────────

class SearchPreferences(BaseModel):
    """
    Soft preference signals for ranking adjustment.
    These boost or penalize — they do NOT exclude.

    WHY soft vs hard:
    - prefer_recent: boosts newer datasets but old classics like Titanic still rank
    - prefer_popular: weights download_count — but popularity bias is adjustable
    - prefer_complete: weights metadata_completeness — but incomplete datasets still show
    """

    model_config = ConfigDict(frozen=True)

    prefer_recent: bool = False             # Boost datasets updated recently
    prefer_popular: bool = False            # Boost datasets with high download_count
    prefer_complete: bool = True            # Boost datasets with high completeness
    result_diversity: float = 0.3           # 0.0 = pure relevance | 1.0 = max diversity
    explanation_detail: str = "standard"    # "minimal" | "standard" | "detailed"
    include_duplicates: bool = False        # Whether to include duplicate-flagged datasets

    @field_validator("result_diversity")
    @classmethod
    def validate_diversity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValidationError(
                message=f"result_diversity must be in [0.0, 1.0], got {v}",
                field="result_diversity",
                invalid_value=v,
                constraint="[0.0, 1.0]",
            )
        return round(v, 2)

    @field_validator("explanation_detail")
    @classmethod
    def validate_explanation_detail(cls, v: str) -> str:
        valid = {"minimal", "standard", "detailed"}
        if v not in valid:
            raise ValidationError(
                message=f"explanation_detail must be one of {valid}, got '{v}'",
                field="explanation_detail",
                invalid_value=v,
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH QUERY — core query object
# ─────────────────────────────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    """
    The primary input object for a dataset search request.

    query_id is auto-generated as UUID4 if not provided.
    It serves as the distributed trace anchor for the entire pipeline.
    """

    model_config = ConfigDict(populate_by_name=True)

    # ── Core query ────────────────────────────────────────────────────────────
    raw_query: str                          # User's natural language input
    expanded_query: Optional[str] = None    # Enriched query from QueryUnderstandingPipeline (synonyms, domain terms)
    max_results: int = 10                   # Number of results to return [1, 50]

    # ── Filtering + preferences ───────────────────────────────────────────────
    filters: SearchFilters = SearchFilters()
    preferences: SearchPreferences = SearchPreferences()

    # ── Tracing ───────────────────────────────────────────────────────────────
    query_id: str = ""                      # Auto-generated UUID4 if not provided
    session_id: Optional[str] = None        # Optional session grouping
    timestamp: datetime = None              # type: ignore[assignment]  # Set in model_validator

    # ── Conversation context ─────────────────────────────────────────────────
    previous_query: Optional[str] = None    # Previous query in a refinement conversation

    # ── Adapter targeting (optional) ─────────────────────────────────────────
    sources: list[str] = []                 # Empty = all sources | ["kaggle"] = Kaggle only

    # ── Metadata ─────────────────────────────────────────────────────────────
    client_version: Optional[str] = None    # Caller's version (for compatibility tracking)

    @field_validator("raw_query")
    @classmethod
    def validate_raw_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValidationError(
                message="raw_query cannot be empty.",
                field="raw_query",
                constraint="non-empty string",
            )
        mode = get_validation_mode()
        if len(stripped) > MAX_QUERY_CHARS:
            if mode == ValidationMode.STRICT:
                raise QueryTooLongError(length=len(stripped), max_chars=MAX_QUERY_CHARS)
            else:
                logger.warning(
                    "query_truncated",
                    extra={"original_length": len(stripped), "max": MAX_QUERY_CHARS},
                )
                stripped = stripped[:MAX_QUERY_CHARS]
        return stripped

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        clamped = max(MAX_RESULTS_FLOOR, min(MAX_RESULTS_CAP, v))
        if clamped != v:
            logger.warning(
                "max_results_clamped",
                extra={"original": v, "clamped": clamped},
            )
        return clamped

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        from .states import VALID_SOURCES
        invalid = [s for s in v if s not in VALID_SOURCES]
        if invalid:
            raise ValidationError(
                message=f"Unknown sources: {invalid}. Valid: {list(VALID_SOURCES)}",
                field="sources",
                invalid_value=invalid,
            )
        return v

    @model_validator(mode="after")
    def set_defaults(self) -> "SearchQuery":
        """Auto-generate query_id and timestamp if not provided."""
        if not self.query_id:
            self.query_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "raw_query": self.raw_query,
            "max_results": self.max_results,
            "sources": self.sources,
            "filters": self.filters.model_dump(),
            "preferences": self.preferences.model_dump(),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT QUERY — SearchQuery extended for multi-agent calls
# ─────────────────────────────────────────────────────────────────────────────

class AgentQuery(SearchQuery):
    """
    Extended SearchQuery for inter-agent communication.

    Adds tracing fields for distributed multi-agent pipelines:
    - calling_agent: which agent is making this call
    - parent_request_id: traces back to the original user request
    - pipeline_context: arbitrary key/value metadata for the pipeline
    - priority: for queue-based architectures
    """

    calling_agent: str = "orchestrator"         # Which agent originated this call
    parent_request_id: Optional[str] = None     # Links to original user SearchQuery.query_id
    pipeline_context: dict[str, Any] = {}       # Arbitrary pipeline metadata
    priority: int = 5                           # 1 (highest) → 10 (lowest)
    timeout_seconds: float = 10.0               # Per-agent SLA

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValidationError(
                message=f"priority must be in [1, 10], got {v}",
                field="priority",
                invalid_value=v,
                constraint="[1, 10]",
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "calling_agent": self.calling_agent,
            "parent_request_id": self.parent_request_id,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
        })
        return d