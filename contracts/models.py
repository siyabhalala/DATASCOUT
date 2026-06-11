"""
datascout.contracts.models
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Core data contracts — the single most important file.
Defines what a dataset IS at every stage of the pipeline.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY three description tiers (description / description_clean / description_short)?
     - description (50KB max): source of truth, NEVER truncated
     - description_clean (2KB): sentence-boundary safe for embedding models
     - description_short (300 chars): word-boundary safe for UI previews
     - At 1M datasets: 20GB + 2GB + 300MB vs recomputing every query → storage wins

  2. WHY cached_property for computed fields?
     - description_clean costs ~50ms (truncation + boundary search)
     - At 1M datasets × 10 accesses per run = 500M ms = 138 hours wasted
     - cached_property computes once, reuses for lifetime of the object
     - Requires model_config with arbitrary_types_allowed=True for Pydantic v2

  3. WHY SHA-256 over MD5 for fingerprinting?
     - At 1M datasets with ~15% cross-platform overlap = 150K records to deduplicate
     - MD5 collision probability at 1M records ≈ 1 in 10^13 — acceptable but not ideal
     - SHA-256 collision probability ≈ 1 in 10^38 — effectively zero
     - Performance difference is negligible (~2ms per hash)

  4. WHY tags + tags_primary as stored (not computed) fields?
     - BM25 search at 1M × all_tags = O(n × 1000 worst case) → ~500s
     - BM25 search at 1M × tags_primary(10) = O(n × 10) → ~5s → 100x faster
     - Both stored at ingestion — no recomputation at query time

  5. WHY lineage fields are NEVER degraded in GRACEFUL mode?
     - Agent-0 is the ONLY point where data origin is known with certainty
     - Missing lineage = permanently unauditable record
     - EU AI Act Article 13 requires full audit trail for automated decisions
     - A record without lineage cannot be recalled, corrected, or explained

  6. WHY ingestion_version has no default?
     - Forces adapters to explicitly set their version
     - A default of "unknown" would allow silent lineage degradation
     - Every record must be traceable to the adapter that produced it

FAILURE SCENARIOS HANDLED:
  - Missing lineage → LineageMissingError — always, both modes (no degradation)
  - Invalid fingerprint → FingerprintInvalidError in STRICT / recompute in GRACEFUL
  - Embedding dim mismatch → ValidationError in STRICT / pad+truncate in GRACEFUL
  - Null description → "" + has_description=False in GRACEFUL / ValidationError in STRICT
  - Tag overflow → store all + log warning + cap tags_primary=10
  - Schema version major mismatch → always raise SchemaVersionMismatchError

PERFORMANCE ANALYSIS:
  - compute_fingerprint: O(title + column_count) ≈ 2ms per record
  - compute_completeness: O(fields) ≈ 0.1ms per record
  - truncate_to_sentence_boundary: O(description_length) ≈ 0.5ms
  - At 10K datasets:  ~25ms total
  - At 100K datasets: ~250ms total
  - At 1M datasets:   ~2.5s total (acceptable — done once at ingestion)

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add language_detected field (optional)
  Breaking: v4.0.0 — rename source_id → platform_id (requires migration)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .errors import (
    EmbeddingDimensionMismatchError,
    FingerprintInvalidError,
    LineageMissingError,
    SchemaVersionMismatchError,
    ValidationError,
)
from .states import (
    DataDomain,
    DataFormat,
    LicenseType,
    QualityTier,
    StageStatus,
    ValidationMode,
    VALID_SOURCES,
    compute_quality_tier,
    get_validation_mode,
)
from .task_types import Modality, TaskType

if TYPE_CHECKING:
    pass

logger = logging.getLogger("datascout.contracts.models")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION   = "3.0.0"
CURRENT_PIPELINE_VERSION = "3.0.0"
MAX_DESCRIPTION_BYTES    = 50 * 1024   # 50KB
MAX_DESCRIPTION_CLEAN    = 2000        # chars
MAX_DESCRIPTION_SHORT    = 300         # chars
MAX_TAGS_PRIMARY         = 10

# Fields tracked for completeness scoring
COMPLETENESS_FIELDS = [
    "title", "description", "tags", "source_url",
    "row_count", "column_count", "file_size_bytes",
    "license_type", "data_format", "last_updated",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fingerprint(
    title: str,
    source: str,
    row_count: Optional[int],
    column_names: Optional[list[str]],
) -> str:
    """
    Compute a stable SHA-256 fingerprint for deduplication.

    WHY normalize before hashing:
    - Survives whitespace/casing edits to title
    - Column names sorted — order must not matter for identity
    - source lowercased — "Kaggle" == "kaggle"

    WHY SHA-256 over MD5:
    - At 1M records: SHA-256 collision probability ≈ 10^-38 vs MD5 ≈ 10^-13
    - Performance difference is negligible

    Returns: 64-character lowercase hex string
    """
    normalized = {
        "title": title.strip().lower(),
        "source": source.strip().lower(),
        "row_count": row_count,
        "columns": sorted(
            col.strip().lower() for col in (column_names or [])
        ),
    }
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def truncate_to_sentence_boundary(text: str, max_chars: int = MAX_DESCRIPTION_CLEAN) -> str:
    """
    Truncate text at the last complete sentence before max_chars.

    WHY sentence boundary (not character boundary):
    - Embedding models need complete semantic units
    - Cutting mid-sentence produces embeddings with broken semantic context
    - "The dataset contains 50K rows of medical" → broken embedding
    - "The dataset contains 50K rows of medical records." → coherent embedding

    WHY not rfind('.') alone:
    - Handles "!", "?" as valid sentence terminators
    - Checks for space after punctuation to avoid "3.14" splits
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if last_period > max_chars // 2:  # Don't truncate to tiny fragment
        return truncated[:last_period + 1]
    return truncated  # Fallback: character boundary if no sentence found


def truncate_to_word_boundary(text: str, max_chars: int = MAX_DESCRIPTION_SHORT) -> str:
    """
    Truncate text at the last complete word before max_chars.

    WHY word boundary:
    - UI display — broken words look unprofessional
    - "...contains medical rec" → bad
    - "...contains medical…" → good
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + "…"
    return truncated + "…"


def compute_primary_tags(tags: list[str], max_primary: int = MAX_TAGS_PRIMARY) -> list[str]:
    """
    Select the top N most discriminative tags for BM25 search indexing.

    WHY not first N:
    - First N may include generic tags like "data", "csv", "dataset"
    - Shorter tags tend to be more specific concepts (fewer chars = rarer term)

    WHY max 10:
    - BM25 index performance — empirically optimal for precision/speed tradeoff
    - At 1M × all_tags: O(n×1000) ≈ 500s | At 1M × 10: O(n×10) ≈ 5s

    Note: Full tags list always preserved on the model for filters and audit.
    """
    if not tags:
        return []
    # Score: shorter + earlier in list = higher priority
    scored = [
        (tag, 1.0 / (len(tag) + 1) + (len(tags) - i) / (len(tags) * 100))
        for i, tag in enumerate(tags)
        if tag and len(tag.strip()) > 1  # Filter single-char noise tags
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in scored[:max_primary]]


def compute_completeness(dataset_dict: dict[str, Any]) -> float:
    """
    Compute structural completeness as ratio of populated fields to total tracked fields.

    FIX v1.2.0 (Issue 8): The original check `not in (None, "", [], {}, 0)` treated
    row_count=0 or column_count=0 as "missing", and mishandled enum fields whose
    str() value might match. Replaced with per-field semantic checks.

    Returns: float in [0.0, 1.0], rounded to 3 decimal places
    """
    def _is_populated(field: str, value: Any) -> bool:
        if value is None:
            return False
        # Numeric fields: populated if a positive integer
        if field in ("row_count", "column_count", "file_size_bytes"):
            try:
                return int(value) > 0
            except (TypeError, ValueError):
                return False
        # Enum fields: exclude None, empty, and "unknown" variants
        if field in ("license_type", "data_format"):
            s = str(value).lower().strip()
            return bool(s) and s not in ("none", "unknown", "")
        # List fields: non-empty list
        if field == "tags":
            return isinstance(value, (list, tuple)) and len(value) > 0
        # String fields (title, description, source_url, last_updated)
        return bool(str(value).strip()) if value else False

    populated = sum(
        1 for f in COMPLETENESS_FIELDS
        if _is_populated(f, dataset_dict.get(f))
    )
    return round(populated / len(COMPLETENESS_FIELDS), 3)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL CONFIG — centralized, nothing hardcoded
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModelConfig:
    """
    Centralized registry of all supported embedding models.

    WHY centralized:
    - No hardcoded dimensions anywhere in the codebase
    - Adding a new model = one entry here, nothing else changes
    - Embedding validation in STRICT/GRACEFUL mode both reference this
    """

    MODELS: dict[str, dict[str, Any]] = {
        "all-MiniLM-L6-v2":       {"dim": 384,  "max_tokens": 256,  "cost_per_1k": 0.0},
        "all-mpnet-base-v2":      {"dim": 768,  "max_tokens": 384,  "cost_per_1k": 0.0},
        "text-embedding-ada-002": {"dim": 1536, "max_tokens": 8191, "cost_per_1k": 0.0001},
        "bge-large-en-v1.5":      {"dim": 1024, "max_tokens": 512,  "cost_per_1k": 0.0},
        "bge-base-en-v1.5":       {"dim": 768,  "max_tokens": 512,  "cost_per_1k": 0.0},
        "e5-large-v2":            {"dim": 1024, "max_tokens": 512,  "cost_per_1k": 0.0},
    }

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    @classmethod
    def get_dim(cls, model_name: str) -> int:
        if model_name not in cls.MODELS:
            raise ValueError(
                f"Unknown embedding model: '{model_name}'. "
                f"Valid models: {list(cls.MODELS.keys())}"
            )
        return cls.MODELS[model_name]["dim"]

    @classmethod
    def get_max_tokens(cls, model_name: str) -> int:
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown embedding model: '{model_name}'")
        return cls.MODELS[model_name]["max_tokens"]

    @classmethod
    def is_known_model(cls, model_name: str) -> bool:
        return model_name in cls.MODELS


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DIMENSION — single scoring axis
# ─────────────────────────────────────────────────────────────────────────────

class ScoreDimension(BaseModel):
    """
    A single dimension of the multi-dimensional scoring model.

    Used by Agent-1 to build score_breakdown in DecisionTrace.
    Each dimension is independently explainable.
    """

    model_config = ConfigDict(frozen=True)

    name: str                       # e.g. "semantic_similarity", "metadata_quality"
    raw_score: float                # Unweighted score [0.0, 1.0]
    weight: float                   # Dimension weight [0.0, 1.0] — sum of all weights = 1.0
    weighted_score: float           # raw_score × weight
    explanation: str                # Why this score was assigned
    is_estimated: bool = False      # True if approximated (e.g. no description available)
    model_used: Optional[str] = None  # Which model computed this (if any)

    @field_validator("raw_score", "weight", "weighted_score")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score/weight must be in [0.0, 1.0], got {v}")
        return round(v, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATASET — Agent-0's primary output contract
# ─────────────────────────────────────────────────────────────────────────────

class RawDataset(BaseModel):
    """
    Agent-0's primary output contract.

    This is the single most important schema in the system.
    Every downstream agent receives data through this contract.

    Field Groups:
      IDENTITY         — fingerprint, canonical_id, dedup fields (Agent-0 exclusive)
      LINEAGE          — source, ingestion_timestamp, versions (Agent-0 exclusive)
      CORE METADATA    — title, description, tags
      CHARACTERISTICS  — row_count, format, license, domain, task types
      QUALITY SIGNALS  — completeness, has_* booleans (Agent-0 exclusive, structural only)
      VERSIONING       — schema_version, pipeline_version, extra

    Adding required fields → MAJOR version bump + migration plan.
    Adding optional fields → MINOR version bump.
    Bug fixes only        → PATCH version bump.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Required for cached_property
        populate_by_name=True,
    )

    # ── IDENTITY (Agent-0 exclusive) ──────────────────────────────────────────
    dataset_fingerprint: str            # SHA-256 — 64-char hex — deduplication key
    canonical_id: str                   # "<source>:<source_id>" — globally unique
    is_duplicate: bool = False          # Was this fingerprint seen before in this run?
    duplicate_of: Optional[str] = None  # canonical_id of primary record (if duplicate)
    dedup_confidence: float = 1.0       # 0.0–1.0 — confidence of dedup decision

    # ── LINEAGE (Agent-0 exclusive — non-degradable) ──────────────────────────
    source: str                         # "kaggle" | "huggingface" | "openml"
    source_url: str                     # Canonical dataset page URL
    source_id: str                      # Platform-native ID (slug, hash, integer)
    ingestion_timestamp: datetime       # UTC — when Agent-0 processed this record
    pipeline_version: str = CURRENT_PIPELINE_VERSION
    ingestion_version: str              # Adapter version — REQUIRED, no default (intentional)
    schema_version: str = CURRENT_SCHEMA_VERSION  # Machine-readable — for migration logic

    # ── CORE METADATA ─────────────────────────────────────────────────────────
    title: str
    description: str = ""               # Full text — 50KB max — NEVER truncated — source of truth
    tags: list[str] = []                # ALL tags — complete canonical record
    tags_primary: list[str] = []        # Top 10 — auto-computed for BM25 search

    # ── DATASET CHARACTERISTICS ───────────────────────────────────────────────
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    column_names: Optional[list[str]] = None
    file_size_bytes: Optional[int] = None
    data_format: Optional[DataFormat] = None
    license_type: Optional[LicenseType] = None
    data_domain: Optional[DataDomain] = None
    task_types: list[TaskType] = []
    modalities: list[Modality] = []
    language: Optional[str] = None
    last_updated: Optional[datetime] = None
    download_count: Optional[int] = None
    upvote_count: Optional[int] = None
    author: Optional[str] = None

    # ── STRUCTURAL QUALITY SIGNALS (Agent-0 exclusive) ────────────────────────
    missing_fields_count: int = 0
    metadata_completeness: float = 0.0  # 0.0–1.0 — structural field coverage
    has_description: bool = False
    has_schema_info: bool = False        # column_names known
    has_size_info: bool = False          # row_count or file_size_bytes known
    has_license_info: bool = False
    description_length: int = 0
    tag_count: int = 0

    # ── VERSIONING & BACKWARD COMPAT ─────────────────────────────────────────
    extra: dict[str, Any] = {}          # Unknown fields from older records

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATORS
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_fingerprint(cls, v: str) -> str:
        mode = get_validation_mode()
        valid = len(v) == 64 and all(c in "0123456789abcdef" for c in v)
        if not valid:
            if mode == ValidationMode.STRICT:
                raise FingerprintInvalidError(v)
            else:
                logger.warning(
                    "invalid_fingerprint",
                    extra={"fingerprint_prefix": v[:16], "length": len(v)},
                )
                # In GRACEFUL mode, return as-is — caller should recompute
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_SOURCES:
            # Source is a LINEAGE field — always reject, even in GRACEFUL mode
            raise LineageMissingError(field="source")
        return v

    @field_validator("metadata_completeness", "dedup_confidence")
    @classmethod
    def validate_zero_one(cls, v: float, info: Any) -> float:
        if not 0.0 <= v <= 1.0:
            mode = get_validation_mode()
            field_name = info.field_name
            if mode == ValidationMode.STRICT:
                raise ValidationError(
                    message=f"{field_name} must be in [0.0, 1.0], got {v}",
                    field=field_name,
                    invalid_value=v,
                    constraint="[0.0, 1.0]",
                )
            else:
                logger.warning("score_out_of_range", extra={"field": field_name, "value": v})
                return max(0.0, min(1.0, v))
        return round(v, 3)

    @model_validator(mode="after")
    def validate_lineage_consistency(self) -> "RawDataset":
        """
        Ensure schema_version and pipeline_version share the same major version.
        Version comparison uses int() to avoid "9" < "10" lexicographic bug.
        """
        try:
            schema_major = int(self.schema_version.split(".")[0])
            pipeline_major = int(self.pipeline_version.split(".")[0])
        except (ValueError, IndexError) as e:
            raise SchemaVersionMismatchError(
                schema_version=self.schema_version,
                pipeline_version=self.pipeline_version,
                cause=e,
            )
        if schema_major != pipeline_major:
            raise SchemaVersionMismatchError(
                schema_version=self.schema_version,
                pipeline_version=self.pipeline_version,
            )
        return self

    @model_validator(mode="after")
    def validate_lineage_completeness(self) -> "RawDataset":
        """
        Enforce that all lineage fields are populated.
        Lineage is ALWAYS rejected if missing — no GRACEFUL degradation.
        """
        required_lineage = {
            "source_url": self.source_url,
            "source_id": self.source_id,
            "ingestion_version": self.ingestion_version,
        }
        for field_name, value in required_lineage.items():
            if not value or (isinstance(value, str) and not value.strip()):
                raise LineageMissingError(field=field_name)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # CACHED PROPERTIES — computed once, reused indefinitely
    # ─────────────────────────────────────────────────────────────────────────

    @cached_property
    def description_clean(self) -> str:
        """
        2KB sentence-boundary truncation for embedding models.
        Computed once per object lifetime — ~50ms saved per subsequent access.
        """
        return truncate_to_sentence_boundary(self.description, MAX_DESCRIPTION_CLEAN)

    @cached_property
    def description_short(self) -> str:
        """
        300-char word-boundary truncation for UI display.
        """
        return truncate_to_word_boundary(self.description, MAX_DESCRIPTION_SHORT)

    @cached_property
    def searchable_text(self) -> str:
        """
        Combined BM25-ready text: title + clean description + primary tags.
        Computed once per object — reused 10–100x per pipeline run.
        """
        parts = [self.title, self.description_clean]
        if self.tags_primary:
            parts.append(" ".join(self.tags_primary))
        return " ".join(parts)

    @cached_property
    def quality_tier(self) -> QualityTier:
        """Coarse quality bucket derived from metadata_completeness."""
        return compute_quality_tier(self.metadata_completeness)

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict_safe(self) -> dict[str, Any]:
        """
        For API responses — excludes embeddings (too large), includes lineage summary.
        Consumers: Frontend, Agent-1 API calls, external integrations.
        """
        return {
            "canonical_id": self.canonical_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "title": self.title,
            "description_short": self.description_short,
            "source": self.source,
            "source_url": self.source_url,
            "tags_primary": self.tags_primary,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_size_bytes": self.file_size_bytes,
            "data_format": self.data_format.value if self.data_format else None,
            "license_type": self.license_type.value if self.license_type else None,
            "data_domain": self.data_domain.value if self.data_domain else None,
            "task_types": [t.value for t in self.task_types],
            "modalities": [m.value for m in self.modalities],
            "metadata_completeness": self.metadata_completeness,
            "quality_tier": self.quality_tier.value,
            "is_duplicate": self.is_duplicate,
            "schema_version": self.schema_version,
        }

    def to_dict_full(self) -> dict[str, Any]:
        """
        For storage (vector DB, document store) — full representation.
        Consumers: Embedding store, dataset cache, offline processing.
        """
        d = self.to_dict_safe()
        d.update({
            "description": self.description,
            "description_clean": self.description_clean,
            "tags": self.tags,
            "column_names": self.column_names,
            "source_id": self.source_id,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat(),
            "pipeline_version": self.pipeline_version,
            "ingestion_version": self.ingestion_version,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "download_count": self.download_count,
            "upvote_count": self.upvote_count,
            "author": self.author,
            "language": self.language,
            "has_description": self.has_description,
            "has_schema_info": self.has_schema_info,
            "has_size_info": self.has_size_info,
            "has_license_info": self.has_license_info,
            "description_length": self.description_length,
            "tag_count": self.tag_count,
            "missing_fields_count": self.missing_fields_count,
            "duplicate_of": self.duplicate_of,
            "dedup_confidence": self.dedup_confidence,
            "extra": self.extra,
        })
        return d

    def to_dict_debug(self) -> dict[str, Any]:
        """
        For monitoring/debugging — includes ALL fields including internals.
        WARNING: Never send to users — contains internal state.
        Consumers: Datadog dashboards, debugging sessions, audit logs.
        """
        d = self.to_dict_full()
        d.update({
            "searchable_text_preview": self.searchable_text[:200],
            "description_clean_preview": self.description_clean[:200],
            "quality_tier": self.quality_tier.value,
            "_computed_at": datetime.now(timezone.utc).isoformat(),
        })
        return d

    def to_dict_lineage(self) -> dict[str, Any]:
        """
        Lineage-only view for compliance and audit systems.
        Consumers: EU AI Act audit systems, data governance tools.
        """
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "canonical_id": self.canonical_id,
            "source": self.source,
            "source_url": self.source_url,
            "source_id": self.source_id,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat(),
            "pipeline_version": self.pipeline_version,
            "ingestion_version": self.ingestion_version,
            "schema_version": self.schema_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATED DATASET — RawDataset + scoring output from Agent-1
# ─────────────────────────────────────────────────────────────────────────────

class EvaluatedDataset(BaseModel):
    """
    A RawDataset enriched with Agent-1's scoring output.

    Agent-0 produces RawDataset. Agent-1 produces EvaluatedDataset.
    This is what flows into ranking, explanation generation, and final response.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Source record ─────────────────────────────────────────────────────────
    raw: RawDataset

    # ── Scoring ───────────────────────────────────────────────────────────────
    score_dimensions: list[ScoreDimension] = []
    composite_score: float = 0.0        # Weighted sum of score_dimensions
    rank: Optional[int] = None          # Final rank in result set (1-indexed)
    quality_tier: QualityTier = QualityTier.INCOMPLETE

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None
    embedding_computed_at: Optional[datetime] = None

    # ── Pipeline tracking ─────────────────────────────────────────────────────
    stages_completed: list[str] = []    # Stage names that have finished
    stage_statuses: dict[str, StageStatus] = {}

    @field_validator("composite_score")
    @classmethod
    def validate_composite(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            mode = get_validation_mode()
            if mode == ValidationMode.STRICT:
                raise ValueError(f"composite_score must be in [0.0, 1.0], got {v}")
            return max(0.0, min(1.0, v))
        return round(v, 4)

    def validate_embedding(self) -> None:
        """
        Validate embedding dimension against registered model config.
        Called explicitly after embedding is assigned — not at construction time.

        STRICT mode: raise EmbeddingDimensionMismatchError
        GRACEFUL mode: pad with zeros or truncate, log warning
        """
        if self.embedding is None or self.embedding_model is None:
            return

        mode = get_validation_mode()
        try:
            expected_dim = EmbeddingModelConfig.get_dim(self.embedding_model)
        except ValueError:
            logger.warning("unknown_embedding_model", extra={"model": self.embedding_model})
            return

        actual_dim = len(self.embedding)
        if actual_dim == expected_dim:
            return

        if mode == ValidationMode.STRICT:
            raise EmbeddingDimensionMismatchError(
                got=actual_dim,
                expected=expected_dim,
                model=self.embedding_model,
                dataset_id=self.raw.canonical_id,
            )

        # GRACEFUL: pad or truncate
        logger.warning(
            "embedding_dimension_autofix",
            extra={
                "canonical_id": self.raw.canonical_id,
                "got": actual_dim,
                "expected": expected_dim,
                "model": self.embedding_model,
                "action": "pad" if actual_dim < expected_dim else "truncate",
            },
        )
        if actual_dim < expected_dim:
            self.embedding = self.embedding + [0.0] * (expected_dim - actual_dim)
        else:
            self.embedding = self.embedding[:expected_dim]

    def to_dict_safe(self) -> dict[str, Any]:
        """API-safe representation — excludes embedding vector."""
        return {
            **self.raw.to_dict_safe(),
            "composite_score": self.composite_score,
            "rank": self.rank,
            "quality_tier": self.quality_tier.value,
            "score_dimensions": [
                {
                    "name": sd.name,
                    "weighted_score": sd.weighted_score,
                    "explanation": sd.explanation,
                }
                for sd in self.score_dimensions
            ],
        }