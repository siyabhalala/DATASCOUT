"""
datascout.contracts.errors.codes
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Complete error taxonomy — every error has a unique code,
metadata, HTTP status, and retryability flag.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY integer error codes?
     - String matching is fragile and case-sensitive
     - Integer ranges allow category-level grouping (1xxx=validation, 2xxx=adapter)
     - Monitoring dashboards aggregate by range without regex
     - Machine-readable: downstream can branch on exact code without string parsing

  2. WHY ErrorMeta dataclass per code?
     - Retry logic reads is_retryable — no string parsing
     - FastAPI reads http_status — no manual mapping
     - Monitoring reads severity — no conditional logging
     - All metadata co-located with the code definition — no split-brain

  3. WHY ERROR_REGISTRY as a dict?
     - Single lookup for all error metadata
     - Testable: can assert every ErrorCode has a registry entry
     - Dashboard-configurable: monitoring tools can read severity from registry

CODE RANGES:
  1000–1999  Validation errors     → never retryable, client fault
  2000–2999  Adapter errors        → sometimes retryable, external fault
  3000–3999  Pipeline errors       → sometimes retryable, internal fault
  4000–4999  Infrastructure errors → sometimes retryable, system fault
  5000–5999  LLM errors            → sometimes retryable, external fault
  9000+      Unknown/catchall      → investigate manually

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add codes for new adapter types
  Breaking: v4.0.0 — renumbering any existing code

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ERROR SEVERITY
# ─────────────────────────────────────────────────────────────────────────────

class ErrorSeverity(str):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CODES
# ─────────────────────────────────────────────────────────────────────────────

class ErrorCode(IntEnum):
    """
    Unique integer codes for every error the system can produce.

    Ranges:
      1000–1999  Validation
      2000–2999  Adapter / External API
      3000–3999  Pipeline
      4000–4999  Infrastructure
      5000–5999  LLM
      9000+      Unknown
    """

    # ── Validation (1000–1999) ────────────────────────────────────────────────
    VALIDATION_ERROR               = 1000
    FINGERPRINT_INVALID            = 1001  # SHA-256 not 64 lowercase hex chars
    LINEAGE_MISSING                = 1002  # Required lineage field absent — always reject
    EMBEDDING_DIMENSION_MISMATCH   = 1003  # Vector dim != expected for model
    SCHEMA_VERSION_MISMATCH        = 1004  # Major version incompatible
    INVALID_SOURCE                 = 1005  # Unknown adapter source string
    INVALID_COMPLETENESS_SCORE     = 1006  # metadata_completeness outside [0, 1]
    INVALID_DEDUP_CONFIDENCE       = 1007  # dedup_confidence outside [0, 1]
    MISSING_REQUIRED_FIELD         = 1008  # Required field is None
    INVALID_FIELD_VALUE            = 1009  # Field value violates constraint
    QUERY_TOO_LONG                 = 1010  # raw_query exceeds max_chars
    INVALID_MAX_RESULTS            = 1011  # max_results outside [1, 50]
    CROSS_FIELD_VIOLATION          = 1012  # Cross-field consistency check failed

    # ── Adapter / External (2000–2999) ────────────────────────────────────────
    ADAPTER_TIMEOUT                = 2000  # HTTP request timed out
    ADAPTER_RATE_LIMITED           = 2001  # 429 Too Many Requests
    ADAPTER_AUTH_FAILED            = 2002  # 401/403 — credentials invalid
    ADAPTER_NOT_FOUND              = 2003  # 404 — resource does not exist
    ADAPTER_MALFORMED_RESPONSE     = 2004  # Response is not parseable JSON
    ADAPTER_SERVER_ERROR           = 2005  # 500/502/503/504
    ADAPTER_CONNECTION_ERROR       = 2006  # Network-level failure (DNS, refused)
    ADAPTER_NOT_REGISTERED         = 2007  # Requested adapter not in registry
    ADAPTER_HEALTH_CHECK_FAILED    = 2008  # Readiness probe returned unhealthy

    # ── Pipeline (3000–3999) ──────────────────────────────────────────────────
    INSUFFICIENT_ADAPTERS          = 3000  # Fewer than min_successful_adapters succeeded
    DEDUP_FAILURE                  = 3001  # Fingerprint computation crashed
    SCORING_FAILURE                = 3002  # Agent-1 scoring raised an exception
    PIPELINE_TIMEOUT               = 3003  # Global pipeline SLA exceeded
    STAGE_TRANSITION_ERROR         = 3004  # Invalid stage status transition
    RESULT_MERGE_FAILURE           = 3005  # Cross-adapter result merge failed

    # ── Infrastructure (4000–4999) ────────────────────────────────────────────
    CIRCUIT_BREAKER_OPEN           = 4000  # Circuit is OPEN — request rejected fast
    CIRCUIT_BREAKER_HALF_OPEN_FAIL = 4001  # Recovery probe failed — re-opened
    RETRY_EXHAUSTED                = 4002  # All retry attempts consumed
    HEALTH_CHECK_FAILED            = 4003  # Service health probe failed
    METRIC_EXPORT_FAILED           = 4004  # Prometheus/OTel export error (non-fatal)

    # ── LLM (5000–5999) ───────────────────────────────────────────────────────
    LLM_TIMEOUT                    = 5000  # LLM API request timed out
    LLM_CONTEXT_OVERFLOW           = 5001  # Input exceeds model token limit
    LLM_FALLBACK_EXHAUSTED         = 5002  # All configured LLM fallbacks failed
    LLM_RATE_LIMITED               = 5003  # LLM API rate limit hit
    LLM_MALFORMED_RESPONSE         = 5004  # LLM output unparseable
    LLM_AUTH_FAILED                = 5005  # LLM API key invalid/expired

    # ── Unknown (9000+) ───────────────────────────────────────────────────────
    UNKNOWN_ERROR                  = 9000  # Unclassified — investigate manually


# ─────────────────────────────────────────────────────────────────────────────
# ERROR META
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ErrorMeta:
    """
    Metadata attached to each ErrorCode.

    WHY frozen dataclass:
    - Error metadata is immutable — should never be modified at runtime
    - Hashable: can be used as dict keys if needed

    Fields:
      message          Template for the error message (can include {placeholders})
      http_status      HTTP response code for API boundary
      is_retryable     Whether this error class is worth retrying
      retry_after_s    Default retry delay in seconds (None = use backoff algorithm)
      severity         Log severity level
      category         Broad category for dashboard grouping
    """
    message: str
    http_status: int
    is_retryable: bool
    retry_after_s: Optional[float]
    severity: str
    category: str


# ─────────────────────────────────────────────────────────────────────────────
# ERROR REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ERROR_REGISTRY: dict[ErrorCode, ErrorMeta] = {

    # ── Validation ────────────────────────────────────────────────────────────
    ErrorCode.VALIDATION_ERROR: ErrorMeta(
        message="Validation failed: {detail}",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.FINGERPRINT_INVALID: ErrorMeta(
        message="Dataset fingerprint must be 64-char lowercase SHA-256 hex, got {length} chars.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="validation",
    ),
    ErrorCode.LINEAGE_MISSING: ErrorMeta(
        message="Required lineage field '{field}' is missing. Record rejected — lineage is non-negotiable.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.CRITICAL, category="validation",
    ),
    ErrorCode.EMBEDDING_DIMENSION_MISMATCH: ErrorMeta(
        message="Embedding dimension mismatch: got {got}, expected {expected} for model {model}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.SCHEMA_VERSION_MISMATCH: ErrorMeta(
        message="Schema major version mismatch: schema_version={schema_v}, pipeline_version={pipeline_v}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="validation",
    ),
    ErrorCode.INVALID_SOURCE: ErrorMeta(
        message="Unknown data source '{source}'. Valid sources: kaggle, huggingface, openml.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="validation",
    ),
    ErrorCode.INVALID_COMPLETENESS_SCORE: ErrorMeta(
        message="metadata_completeness must be in [0.0, 1.0], got {value}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.INVALID_DEDUP_CONFIDENCE: ErrorMeta(
        message="dedup_confidence must be in [0.0, 1.0], got {value}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.MISSING_REQUIRED_FIELD: ErrorMeta(
        message="Required field '{field}' is None or empty.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="validation",
    ),
    ErrorCode.INVALID_FIELD_VALUE: ErrorMeta(
        message="Field '{field}' has invalid value: {value}. Constraint: {constraint}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.QUERY_TOO_LONG: ErrorMeta(
        message="Query exceeds maximum length of {max_chars} characters.",
        http_status=400, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.INVALID_MAX_RESULTS: ErrorMeta(
        message="max_results must be between 1 and 50, got {value}.",
        http_status=400, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="validation",
    ),
    ErrorCode.CROSS_FIELD_VIOLATION: ErrorMeta(
        message="Cross-field validation failed: {detail}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="validation",
    ),

    # ── Adapter ───────────────────────────────────────────────────────────────
    ErrorCode.ADAPTER_TIMEOUT: ErrorMeta(
        message="Adapter '{adapter}' timed out after {timeout_s}s.",
        http_status=504, is_retryable=True, retry_after_s=2.0,
        severity=ErrorSeverity.WARNING, category="adapter",
    ),
    ErrorCode.ADAPTER_RATE_LIMITED: ErrorMeta(
        message="Adapter '{adapter}' rate-limited. Retry after {retry_after}s.",
        http_status=429, is_retryable=True, retry_after_s=60.0,
        severity=ErrorSeverity.WARNING, category="adapter",
    ),
    ErrorCode.ADAPTER_AUTH_FAILED: ErrorMeta(
        message="Adapter '{adapter}' authentication failed. Check API credentials.",
        http_status=401, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.CRITICAL, category="adapter",
    ),
    ErrorCode.ADAPTER_NOT_FOUND: ErrorMeta(
        message="Adapter '{adapter}' returned 404 for resource '{resource}'.",
        http_status=404, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="adapter",
    ),
    ErrorCode.ADAPTER_MALFORMED_RESPONSE: ErrorMeta(
        message="Adapter '{adapter}' returned unparseable response: {detail}.",
        http_status=502, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="adapter",
    ),
    ErrorCode.ADAPTER_SERVER_ERROR: ErrorMeta(
        message="Adapter '{adapter}' returned server error {status_code}.",
        http_status=502, is_retryable=True, retry_after_s=5.0,
        severity=ErrorSeverity.ERROR, category="adapter",
    ),
    ErrorCode.ADAPTER_CONNECTION_ERROR: ErrorMeta(
        message="Adapter '{adapter}' connection failed: {detail}.",
        http_status=503, is_retryable=True, retry_after_s=3.0,
        severity=ErrorSeverity.ERROR, category="adapter",
    ),
    ErrorCode.ADAPTER_NOT_REGISTERED: ErrorMeta(
        message="Adapter '{adapter}' is not registered in AdapterRegistry.",
        http_status=500, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="adapter",
    ),
    ErrorCode.ADAPTER_HEALTH_CHECK_FAILED: ErrorMeta(
        message="Adapter '{adapter}' health check failed: {detail}.",
        http_status=503, is_retryable=True, retry_after_s=30.0,
        severity=ErrorSeverity.WARNING, category="adapter",
    ),

    # ── Pipeline ──────────────────────────────────────────────────────────────
    ErrorCode.INSUFFICIENT_ADAPTERS: ErrorMeta(
        message="Only {succeeded}/{total} adapters succeeded. Minimum required: {minimum}.",
        http_status=503, is_retryable=True, retry_after_s=5.0,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),
    ErrorCode.DEDUP_FAILURE: ErrorMeta(
        message="Deduplication fingerprint computation failed for dataset '{dataset_id}': {detail}.",
        http_status=500, is_retryable=True, retry_after_s=1.0,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),
    ErrorCode.SCORING_FAILURE: ErrorMeta(
        message="Scoring failed for dataset '{dataset_id}': {detail}.",
        http_status=500, is_retryable=True, retry_after_s=2.0,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),
    ErrorCode.PIPELINE_TIMEOUT: ErrorMeta(
        message="Pipeline exceeded global SLA of {timeout_s}s.",
        http_status=504, is_retryable=True, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),
    ErrorCode.STAGE_TRANSITION_ERROR: ErrorMeta(
        message="Invalid stage transition: {from_stage} → {to_stage}.",
        http_status=500, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),
    ErrorCode.RESULT_MERGE_FAILURE: ErrorMeta(
        message="Failed to merge results from adapters: {detail}.",
        http_status=500, is_retryable=True, retry_after_s=1.0,
        severity=ErrorSeverity.ERROR, category="pipeline",
    ),

    # ── Infrastructure ────────────────────────────────────────────────────────
    ErrorCode.CIRCUIT_BREAKER_OPEN: ErrorMeta(
        message="Circuit breaker OPEN for '{service}'. Will retry at {retry_at}.",
        http_status=503, is_retryable=True, retry_after_s=60.0,
        severity=ErrorSeverity.WARNING, category="infrastructure",
    ),
    ErrorCode.CIRCUIT_BREAKER_HALF_OPEN_FAIL: ErrorMeta(
        message="Circuit breaker recovery probe failed for '{service}'. Re-opened.",
        http_status=503, is_retryable=True, retry_after_s=120.0,
        severity=ErrorSeverity.WARNING, category="infrastructure",
    ),
    ErrorCode.RETRY_EXHAUSTED: ErrorMeta(
        message="All {max_attempts} retry attempts exhausted for '{operation}'.",
        http_status=503, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.ERROR, category="infrastructure",
    ),
    ErrorCode.HEALTH_CHECK_FAILED: ErrorMeta(
        message="Health check failed for component '{component}': {detail}.",
        http_status=503, is_retryable=True, retry_after_s=10.0,
        severity=ErrorSeverity.WARNING, category="infrastructure",
    ),
    ErrorCode.METRIC_EXPORT_FAILED: ErrorMeta(
        message="Metric export failed (non-fatal): {detail}.",
        http_status=200, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="infrastructure",
    ),

    # ── LLM ───────────────────────────────────────────────────────────────────
    ErrorCode.LLM_TIMEOUT: ErrorMeta(
        message="LLM '{model}' timed out after {timeout_s}s.",
        http_status=504, is_retryable=True, retry_after_s=3.0,
        severity=ErrorSeverity.WARNING, category="llm",
    ),
    ErrorCode.LLM_CONTEXT_OVERFLOW: ErrorMeta(
        message="LLM '{model}' input exceeds token limit: {tokens} > {max_tokens}.",
        http_status=422, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.WARNING, category="llm",
    ),
    ErrorCode.LLM_FALLBACK_EXHAUSTED: ErrorMeta(
        message="All LLM fallbacks exhausted. Models tried: {models}.",
        http_status=503, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.CRITICAL, category="llm",
    ),
    ErrorCode.LLM_RATE_LIMITED: ErrorMeta(
        message="LLM '{model}' rate-limited. Retry after {retry_after}s.",
        http_status=429, is_retryable=True, retry_after_s=30.0,
        severity=ErrorSeverity.WARNING, category="llm",
    ),
    ErrorCode.LLM_MALFORMED_RESPONSE: ErrorMeta(
        message="LLM '{model}' returned unparseable output: {detail}.",
        http_status=502, is_retryable=True, retry_after_s=1.0,
        severity=ErrorSeverity.WARNING, category="llm",
    ),
    ErrorCode.LLM_AUTH_FAILED: ErrorMeta(
        message="LLM '{model}' API key invalid or expired.",
        http_status=401, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.CRITICAL, category="llm",
    ),

    # ── Unknown ───────────────────────────────────────────────────────────────
    ErrorCode.UNKNOWN_ERROR: ErrorMeta(
        message="Unexpected error: {detail}. Please investigate.",
        http_status=500, is_retryable=False, retry_after_s=None,
        severity=ErrorSeverity.CRITICAL, category="unknown",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY INTEGRITY CHECK — run at import time
# ─────────────────────────────────────────────────────────────────────────────

def _validate_registry() -> None:
    """Ensure every ErrorCode has a corresponding registry entry."""
    missing = [code for code in ErrorCode if code not in ERROR_REGISTRY]
    if missing:
        raise RuntimeError(
            f"ERROR_REGISTRY is incomplete. Missing entries for: {missing}. "
            "Every ErrorCode must have a corresponding ErrorMeta."
        )


_validate_registry()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def is_retryable(code: ErrorCode) -> bool:
    """Return True if this error class is worth retrying."""
    meta = ERROR_REGISTRY.get(code)
    return meta.is_retryable if meta else False


def get_http_status(code: ErrorCode) -> int:
    """Return the HTTP status code for this error. Defaults to 500."""
    meta = ERROR_REGISTRY.get(code)
    return meta.http_status if meta else 500


def get_retry_after(code: ErrorCode) -> Optional[float]:
    """Return suggested retry delay in seconds, or None to use backoff algorithm."""
    meta = ERROR_REGISTRY.get(code)
    return meta.retry_after_s if meta else None


def get_severity(code: ErrorCode) -> str:
    """Return log severity string for this error code."""
    meta = ERROR_REGISTRY.get(code)
    return meta.severity if meta else ErrorSeverity.ERROR


def get_category(code: ErrorCode) -> str:
    """Return broad category string for dashboard grouping."""
    meta = ERROR_REGISTRY.get(code)
    return meta.category if meta else "unknown"


def is_validation_error(code: ErrorCode) -> bool:
    """Return True if this is a validation-range error (1000–1999)."""
    return 1000 <= int(code) <= 1999


def is_adapter_error(code: ErrorCode) -> bool:
    """Return True if this is an adapter-range error (2000–2999)."""
    return 2000 <= int(code) <= 2999


def is_infrastructure_error(code: ErrorCode) -> bool:
    """Return True if this is an infrastructure-range error (4000–4999)."""
    return 4000 <= int(code) <= 4999