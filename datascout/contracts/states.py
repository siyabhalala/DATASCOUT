"""
datascout.contracts.states
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Shared vocabulary — every categorical value in the system.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY enums over Literal strings?
     - Enums have methods: LicenseType.CC0.is_commercial_allowed() is impossible with str
     - Enums are iterable for validation loops
     - Enums prevent typos at compile time — caught by IDE, not at runtime
     - Enums show up in autocomplete — strings do not
     - Enums are serializable to/from string via .value and Enum(value)

  2. WHY normalize_domain() / normalize_format() fuzzy mappers?
     - Kaggle returns "Computer Vision", HuggingFace "computer-vision", OpenML "image-clf"
     - Without normalization: 3 separate buckets in search, broken dedup, broken filters
     - Normalized once at Agent-0 ingestion, used clean everywhere downstream

  3. WHY ValidationMode as an env-var-driven enum?
     - STRICT: dev/test — fail fast, catch bugs immediately
     - GRACEFUL: production — degrade gracefully, never crash on bad external data
     - Single global switch — no per-module configuration needed

FAILURE SCENARIOS HANDLED:
  - Unknown domain string → normalize_domain() → DataDomain.OTHER + log warning
  - Unknown format string → normalize_format() → DataFormat.OTHER + log warning
  - Unknown license string → normalize_license() → LicenseType.UNKNOWN + log warning

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add SYNTHETIC as DataDomain
  Breaking: v4.0.0 — rename enum values requires migration of stored records

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger("datascout.contracts.states")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION MODE — global switch: STRICT (dev) vs GRACEFUL (prod)
# ─────────────────────────────────────────────────────────────────────────────

class ValidationMode(str, Enum):
    """
    Global validation mode switch.
    Set via VALIDATION_MODE environment variable.
    STRICT  → fail fast in dev/test, raise immediately on bad data
    GRACEFUL → degrade safely in prod, log + continue on bad data
    """
    STRICT   = "STRICT"
    GRACEFUL = "GRACEFUL"


def get_validation_mode() -> ValidationMode:
    """Read validation mode from environment. Defaults to GRACEFUL for safety."""
    raw = os.getenv("VALIDATION_MODE", "GRACEFUL").upper().strip()
    try:
        return ValidationMode(raw)
    except ValueError:
        logger.warning(
            "Unknown VALIDATION_MODE=%r, defaulting to GRACEFUL", raw
        )
        return ValidationMode.GRACEFUL


VALIDATION_MODE: ValidationMode = get_validation_mode()


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOMAIN
# ─────────────────────────────────────────────────────────────────────────────

class DataDomain(str, Enum):
    """
    High-level ML domain taxonomy.
    Used for filtering, routing, and display.
    Stored as string value in all serialization targets.
    """
    COMPUTER_VISION  = "computer_vision"
    NLP              = "nlp"
    TABULAR          = "tabular"
    TIME_SERIES      = "time_series"
    AUDIO            = "audio"
    MULTIMODAL       = "multimodal"
    GEOSPATIAL       = "geospatial"
    MEDICAL          = "medical"
    FINANCE          = "finance"
    SCIENTIFIC       = "scientific"
    SOCIAL           = "social"
    OTHER            = "other"


# Fuzzy mapping from raw API strings → DataDomain
# Each key is a normalized (lower, stripped) substring to match against
_DOMAIN_KEYWORD_MAP: list[tuple[str, DataDomain]] = [
    ("computer_vision",    DataDomain.COMPUTER_VISION),
    ("computer-vision",    DataDomain.COMPUTER_VISION),
    ("image",              DataDomain.COMPUTER_VISION),
    ("vision",             DataDomain.COMPUTER_VISION),
    ("cv",                 DataDomain.COMPUTER_VISION),
    ("nlp",                DataDomain.NLP),
    ("natural_language",   DataDomain.NLP),
    ("natural-language",   DataDomain.NLP),
    ("text",               DataDomain.NLP),
    ("language",           DataDomain.NLP),
    ("time_series",        DataDomain.TIME_SERIES),
    ("time-series",        DataDomain.TIME_SERIES),
    ("timeseries",         DataDomain.TIME_SERIES),
    ("temporal",           DataDomain.TIME_SERIES),
    ("audio",              DataDomain.AUDIO),
    ("speech",             DataDomain.AUDIO),
    ("sound",              DataDomain.AUDIO),
    ("multimodal",         DataDomain.MULTIMODAL),
    ("multi_modal",        DataDomain.MULTIMODAL),
    ("multi-modal",        DataDomain.MULTIMODAL),
    ("geospatial",         DataDomain.GEOSPATIAL),
    ("geo",                DataDomain.GEOSPATIAL),
    ("spatial",            DataDomain.GEOSPATIAL),
    ("map",                DataDomain.GEOSPATIAL),
    ("medical",            DataDomain.MEDICAL),
    ("health",             DataDomain.MEDICAL),
    ("clinical",           DataDomain.MEDICAL),
    ("biomedical",         DataDomain.MEDICAL),
    ("finance",            DataDomain.FINANCE),
    ("financial",          DataDomain.FINANCE),
    ("stock",              DataDomain.FINANCE),
    ("economic",           DataDomain.FINANCE),
    ("scientific",         DataDomain.SCIENTIFIC),
    ("science",            DataDomain.SCIENTIFIC),
    ("research",           DataDomain.SCIENTIFIC),
    ("social",             DataDomain.SOCIAL),
    ("tabular",            DataDomain.TABULAR),
    ("structured",         DataDomain.TABULAR),
    ("classification",     DataDomain.TABULAR),  # fallback
    ("regression",         DataDomain.TABULAR),  # fallback
]


def normalize_domain(raw: Optional[str]) -> DataDomain:
    """
    Map a raw platform-returned domain string to DataDomain enum.

    WHY fuzzy matching: Kaggle/HF/OpenML each use different string conventions.
    WHY return OTHER on failure: Never crash ingestion — quality signal captures it.
    """
    if not raw:
        return DataDomain.OTHER
    normalized = raw.strip().lower().replace(" ", "_")
    # Try exact enum value match first (fastest path)
    try:
        return DataDomain(normalized)
    except ValueError:
        pass
    # Fuzzy keyword match
    for keyword, domain in _DOMAIN_KEYWORD_MAP:
        if keyword in normalized:
            return domain
    logger.debug("normalize_domain: unknown domain %r → OTHER", raw)
    return DataDomain.OTHER


# ─────────────────────────────────────────────────────────────────────────────
# DATA FORMAT
# ─────────────────────────────────────────────────────────────────────────────

class DataFormat(str, Enum):
    """
    File/storage format of a dataset.
    ARFF included for OpenML compatibility (its native format).
    """
    CSV          = "csv"
    JSON         = "json"
    JSONL        = "jsonl"
    PARQUET      = "parquet"
    ARROW        = "arrow"
    IMAGES       = "images"
    AUDIO_FILES  = "audio_files"
    VIDEO        = "video"
    HDF5         = "hdf5"
    SQLITE       = "sqlite"
    ARFF         = "arff"       # OpenML native
    XLSX         = "xlsx"
    XML          = "xml"
    TEXT         = "text"
    OTHER        = "other"


_FORMAT_KEYWORD_MAP: list[tuple[str, DataFormat]] = [
    ("csv",       DataFormat.CSV),
    ("json",      DataFormat.JSON),
    ("jsonl",     DataFormat.JSONL),
    ("ndjson",    DataFormat.JSONL),
    ("parquet",   DataFormat.PARQUET),
    ("arrow",     DataFormat.ARROW),
    ("feather",   DataFormat.ARROW),
    ("image",     DataFormat.IMAGES),
    ("png",       DataFormat.IMAGES),
    ("jpg",       DataFormat.IMAGES),
    ("jpeg",      DataFormat.IMAGES),
    ("audio",     DataFormat.AUDIO_FILES),
    ("mp3",       DataFormat.AUDIO_FILES),
    ("wav",       DataFormat.AUDIO_FILES),
    ("video",     DataFormat.VIDEO),
    ("mp4",       DataFormat.VIDEO),
    ("hdf5",      DataFormat.HDF5),
    ("h5",        DataFormat.HDF5),
    ("sqlite",    DataFormat.SQLITE),
    ("db",        DataFormat.SQLITE),
    ("arff",      DataFormat.ARFF),
    ("xlsx",      DataFormat.XLSX),
    ("excel",     DataFormat.XLSX),
    ("xml",       DataFormat.XML),
    ("txt",       DataFormat.TEXT),
    ("text",      DataFormat.TEXT),
]


def normalize_format(raw: Optional[str]) -> DataFormat:
    """Map raw format string → DataFormat enum with fuzzy fallback."""
    if not raw:
        return DataFormat.OTHER
    normalized = raw.strip().lower()
    try:
        return DataFormat(normalized)
    except ValueError:
        pass
    for keyword, fmt in _FORMAT_KEYWORD_MAP:
        if keyword in normalized:
            return fmt
    logger.debug("normalize_format: unknown format %r → OTHER", raw)
    return DataFormat.OTHER


# ─────────────────────────────────────────────────────────────────────────────
# LICENSE TYPE
# ─────────────────────────────────────────────────────────────────────────────

class LicenseType(str, Enum):
    """
    Dataset license taxonomy.
    Commercial use flag derived from license type — not stored separately.
    """
    CC0          = "cc0"           # Public domain — fully open
    CC_BY        = "cc_by"         # Attribution required
    CC_BY_SA     = "cc_by_sa"      # Share-alike required
    CC_BY_NC     = "cc_by_nc"      # Non-commercial only
    CC_BY_NC_SA  = "cc_by_nc_sa"   # Non-commercial + share-alike
    CC_BY_ND     = "cc_by_nd"      # No derivatives
    MIT          = "mit"
    APACHE_2     = "apache_2"
    GPL_2        = "gpl_2"
    GPL_3        = "gpl_3"
    LGPL         = "lgpl"
    BSD          = "bsd"
    PROPRIETARY  = "proprietary"   # All rights reserved
    UNKNOWN      = "unknown"       # License not specified


_LICENSE_KEYWORD_MAP: list[tuple[str, LicenseType]] = [
    ("cc0",             LicenseType.CC0),
    ("cc-0",            LicenseType.CC0),
    ("public domain",   LicenseType.CC0),
    ("cc by-nc-sa",     LicenseType.CC_BY_NC_SA),
    ("cc-by-nc-sa",     LicenseType.CC_BY_NC_SA),
    ("cc by-nc",        LicenseType.CC_BY_NC),
    ("cc-by-nc",        LicenseType.CC_BY_NC),
    ("cc by-nd",        LicenseType.CC_BY_ND),
    ("cc-by-nd",        LicenseType.CC_BY_ND),
    ("cc by-sa",        LicenseType.CC_BY_SA),
    ("cc-by-sa",        LicenseType.CC_BY_SA),
    ("cc by",           LicenseType.CC_BY),
    ("cc-by",           LicenseType.CC_BY),
    ("mit",             LicenseType.MIT),
    ("apache 2",        LicenseType.APACHE_2),
    ("apache-2",        LicenseType.APACHE_2),
    ("apache2",         LicenseType.APACHE_2),
    ("gpl-3",           LicenseType.GPL_3),
    ("gpl3",            LicenseType.GPL_3),
    ("gpl-2",           LicenseType.GPL_2),
    ("gpl2",            LicenseType.GPL_2),
    ("lgpl",            LicenseType.LGPL),
    ("bsd",             LicenseType.BSD),
    ("proprietary",     LicenseType.PROPRIETARY),
    ("all rights",      LicenseType.PROPRIETARY),
    ("commercial",      LicenseType.PROPRIETARY),
]

# Which licenses allow commercial use
_COMMERCIAL_ALLOWED: frozenset[LicenseType] = frozenset({
    LicenseType.CC0,
    LicenseType.CC_BY,
    LicenseType.CC_BY_SA,
    LicenseType.MIT,
    LicenseType.APACHE_2,
    LicenseType.BSD,
})

# Human-readable display names
_LICENSE_DISPLAY_NAMES: dict[LicenseType, str] = {
    LicenseType.CC0:         "CC0 (Public Domain)",
    LicenseType.CC_BY:       "CC BY 4.0",
    LicenseType.CC_BY_SA:    "CC BY-SA 4.0",
    LicenseType.CC_BY_NC:    "CC BY-NC 4.0",
    LicenseType.CC_BY_NC_SA: "CC BY-NC-SA 4.0",
    LicenseType.CC_BY_ND:    "CC BY-ND 4.0",
    LicenseType.MIT:         "MIT License",
    LicenseType.APACHE_2:    "Apache 2.0",
    LicenseType.GPL_2:       "GPL v2",
    LicenseType.GPL_3:       "GPL v3",
    LicenseType.LGPL:        "LGPL",
    LicenseType.BSD:         "BSD License",
    LicenseType.PROPRIETARY: "Proprietary",
    LicenseType.UNKNOWN:     "Unknown",
}


def normalize_license(raw: Optional[str]) -> LicenseType:
    """Map raw license string → LicenseType enum with fuzzy fallback."""
    if not raw:
        return LicenseType.UNKNOWN
    normalized = raw.strip().lower()
    try:
        return LicenseType(normalized)
    except ValueError:
        pass
    for keyword, lt in _LICENSE_KEYWORD_MAP:
        if keyword in normalized:
            return lt
    logger.debug("normalize_license: unknown license %r → UNKNOWN", raw)
    return LicenseType.UNKNOWN


def is_commercial_allowed(license_type: LicenseType) -> bool:
    """Return True if this license permits commercial use."""
    return license_type in _COMMERCIAL_ALLOWED


def license_display_name(license_type: LicenseType) -> str:
    """Return human-readable license name for UI display."""
    return _LICENSE_DISPLAY_NAMES.get(license_type, str(license_type.value))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE STATUS
# ─────────────────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    """
    Pipeline stage execution status.
    Used in PipelineStage tracking and EvaluatedDataset stage history.
    """
    PENDING   = "pending"    # Not yet started
    RUNNING   = "running"    # Currently executing
    DONE      = "done"       # Completed successfully
    FAILED    = "failed"     # Completed with error
    SKIPPED   = "skipped"    # Deliberately bypassed (e.g. duplicate record)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE IDENTIFIERS — valid data sources
# ─────────────────────────────────────────────────────────────────────────────

class DataSource(str, Enum):
    """
    Valid data source identifiers.
    Used in lineage fields and adapter routing.
    """
    KAGGLE        = "kaggle"
    HUGGINGFACE   = "huggingface"
    OPENML        = "openml"


VALID_SOURCES: frozenset[str] = frozenset(s.value for s in DataSource)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY TIER — coarse bucket for ranking display
# ─────────────────────────────────────────────────────────────────────────────

class QualityTier(str, Enum):
    """
    Coarse quality bucket assigned after scoring.
    Used for UI badges and coarse filtering.
    GOLD: completeness ≥ 0.85 | SILVER: ≥ 0.65 | BRONZE: ≥ 0.45 | INCOMPLETE: < 0.45
    """
    GOLD       = "gold"
    SILVER     = "silver"
    BRONZE     = "bronze"
    INCOMPLETE = "incomplete"


def compute_quality_tier(completeness: float) -> QualityTier:
    """
    Map metadata_completeness score → QualityTier bucket.

    WHY thresholds at 0.85/0.65/0.45:
    - Empirically tuned on Kaggle + HuggingFace dataset distributions
    - GOLD = genuinely well-documented datasets worth highlighting
    - INCOMPLETE = datasets where critical fields are missing
    """
    if completeness >= 0.85:
        return QualityTier.GOLD
    elif completeness >= 0.65:
        return QualityTier.SILVER
    elif completeness >= 0.45:
        return QualityTier.BRONZE
    else:
        return QualityTier.INCOMPLETE