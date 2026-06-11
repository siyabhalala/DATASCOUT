"""
datascout.contracts.errors.exceptions
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Typed exception hierarchy — every exception carries
an error code, retry metadata, and structured context. No bare Exception raises.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY typed exceptions over bare Exception + message strings?
     - Retry logic calls exception.is_retryable — no string parsing
     - Circuit breaker counts AdapterTimeoutError specifically — not "timeout" substrings
     - FastAPI handler calls exception.http_status — no manual mapping switch
     - Monitoring aggregates by exception type — not by regex on message strings
     - catch AdapterRateLimitError vs catch AdapterError → different recovery paths

  2. WHY carry context dict on every exception?
     - "Adapter timeout" is useless. "kaggle adapter timed out after 6s on query 'XYZ'" is actionable.
     - Structured context fields → Datadog indexed search fields
     - Logged at ERROR level with full context — single log entry per failure

  3. WHY LineageMissingError is non-degradable even in GRACEFUL mode?
     - Lineage is the ONLY place origin is known with certainty
     - If lineage is missing, the record is permanently unauditable
     - No safe default exists for "where did this come from?"
     - Rejecting the record is the only correct action

FAILURE SCENARIOS HANDLED:
  - Every exception class maps to an ErrorCode in the registry
  - is_retryable and retry_after_seconds derived from registry at construction
  - http_status derived from registry at construction — no manual sync needed

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add exceptions for new adapter types
  Breaking: v4.0.0 — restructuring exception hierarchy

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .codes import ErrorCode, ERROR_REGISTRY, get_http_status, is_retryable, get_retry_after


# ─────────────────────────────────────────────────────────────────────────────
# BASE EXCEPTION
# ─────────────────────────────────────────────────────────────────────────────

class DataScoutError(Exception):
    """
    Base class for all DATASCOUT exceptions.

    Every exception in the system:
    - Has a unique ErrorCode
    - Has a structured context dict (for logging)
    - Knows if it's retryable
    - Knows its HTTP status code
    - Carries a UTC timestamp
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.context: dict[str, Any] = context or {}
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        # Derived from registry — no manual sync needed
        self.http_status: int = get_http_status(code)
        self.is_retryable: bool = is_retryable(code)
        self.retry_after_seconds: Optional[float] = get_retry_after(code)

    def to_dict(self) -> dict[str, Any]:
        """Structured representation for logging and API error responses."""
        return {
            "error_code": int(self.code),
            "error_name": self.code.name,
            "message": self.message,
            "http_status": self.http_status,
            "is_retryable": self.is_retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.name}, "
            f"message={self.message!r}, "
            f"retryable={self.is_retryable})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(DataScoutError):
    """
    Raised when field-level or cross-field validation fails.
    Never retryable — fix the input data.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        invalid_value: Any = None,
        constraint: Optional[str] = None,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        cause: Optional[Exception] = None,
    ) -> None:
        context = {
            "field": field,
            "invalid_value": str(invalid_value) if invalid_value is not None else None,
            "constraint": constraint,
        }
        super().__init__(message=message, code=code, context=context, cause=cause)
        self.field = field
        self.invalid_value = invalid_value
        self.constraint = constraint


class FingerprintInvalidError(ValidationError):
    """Raised when dataset_fingerprint is not a valid 64-char SHA-256 hex string."""

    def __init__(self, fingerprint: str, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"Fingerprint must be 64-char lowercase SHA-256 hex, got {len(fingerprint)} chars.",
            field="dataset_fingerprint",
            invalid_value=fingerprint[:16] + "..." if len(fingerprint) > 16 else fingerprint,
            constraint="len == 64 and all chars in 0-9a-f",
            code=ErrorCode.FINGERPRINT_INVALID,
            cause=cause,
        )


class LineageMissingError(ValidationError):
    """
    Raised when a required lineage field is absent.

    CRITICAL: This error is NEVER degraded gracefully, even in GRACEFUL mode.
    If lineage cannot be captured, the record is rejected unconditionally.
    There is no safe default for 'where did this data come from?'
    """

    def __init__(self, field: str, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=(
                f"Required lineage field '{field}' is missing. "
                "Record rejected — lineage is non-negotiable and cannot be reconstructed later."
            ),
            field=field,
            constraint="required — no safe default exists",
            code=ErrorCode.LINEAGE_MISSING,
            cause=cause,
        )


class EmbeddingDimensionMismatchError(ValidationError):
    """Raised when embedding vector dimension doesn't match model's expected dimension."""

    def __init__(
        self,
        got: int,
        expected: int,
        model: str,
        dataset_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Embedding dimension mismatch: got {got}, expected {expected} for model '{model}'.",
            field="embedding",
            invalid_value=got,
            constraint=f"must be {expected} for {model}",
            code=ErrorCode.EMBEDDING_DIMENSION_MISMATCH,
            cause=cause,
        )
        self.context["got"] = got
        self.context["expected"] = expected
        self.context["model"] = model
        self.context["dataset_id"] = dataset_id


class SchemaVersionMismatchError(ValidationError):
    """Raised when schema_version and pipeline_version have incompatible major versions."""

    def __init__(
        self,
        schema_version: str,
        pipeline_version: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Major version mismatch: schema_version={schema_version}, "
                f"pipeline_version={pipeline_version}. Run migration before processing."
            ),
            field="schema_version",
            constraint="schema_version.major must equal pipeline_version.major",
            code=ErrorCode.SCHEMA_VERSION_MISMATCH,
            cause=cause,
        )
        self.context["schema_version"] = schema_version
        self.context["pipeline_version"] = pipeline_version


class InvalidSourceError(ValidationError):
    """Raised when source field contains an unknown data source identifier."""

    def __init__(self, source: str, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"Unknown data source '{source}'. Valid: kaggle, huggingface, openml.",
            field="source",
            invalid_value=source,
            constraint="must be one of: kaggle, huggingface, openml",
            code=ErrorCode.INVALID_SOURCE,
            cause=cause,
        )


class QueryTooLongError(ValidationError):
    """Raised when raw_query exceeds character limit."""

    def __init__(self, length: int, max_chars: int = 500, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"Query length {length} exceeds maximum of {max_chars} characters.",
            field="raw_query",
            invalid_value=length,
            constraint=f"max {max_chars} characters",
            code=ErrorCode.QUERY_TOO_LONG,
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class AdapterError(DataScoutError):
    """Base class for all adapter / external API failures."""

    def __init__(
        self,
        message: str,
        adapter: str,
        code: ErrorCode = ErrorCode.ADAPTER_SERVER_ERROR,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = {"adapter": adapter, **(context or {})}
        super().__init__(message=message, code=code, context=ctx, cause=cause)
        self.adapter = adapter


class AdapterTimeoutError(AdapterError):
    """
    Raised when an adapter HTTP request times out.
    Retryable — transient network condition.
    Circuit breaker tracks these to decide when to open.
    """

    def __init__(
        self,
        adapter: str,
        timeout_s: float,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' timed out after {timeout_s}s.",
            adapter=adapter,
            code=ErrorCode.ADAPTER_TIMEOUT,
            context={"timeout_s": timeout_s},
            cause=cause,
        )
        self.timeout_s = timeout_s


class AdapterRateLimitError(AdapterError):
    """
    Raised when adapter returns 429 Too Many Requests.
    Retryable — uses retry_after from response headers if available.
    """

    def __init__(
        self,
        adapter: str,
        retry_after: Optional[float] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' rate-limited. Retry after {retry_after or 60}s.",
            adapter=adapter,
            code=ErrorCode.ADAPTER_RATE_LIMITED,
            context={"retry_after": retry_after},
            cause=cause,
        )
        # Override retry_after_seconds with the actual value from headers
        if retry_after is not None:
            self.retry_after_seconds = retry_after


class AdapterAuthError(AdapterError):
    """
    Raised when adapter returns 401/403.
    NOT retryable — fix the credentials, retrying will keep failing.
    """

    def __init__(self, adapter: str, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' authentication failed. Check API credentials.",
            adapter=adapter,
            code=ErrorCode.ADAPTER_AUTH_FAILED,
            cause=cause,
        )


class AdapterNotFoundError(AdapterError):
    """Raised when adapter returns 404 for a specific resource."""

    def __init__(
        self,
        adapter: str,
        resource: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' returned 404 for resource '{resource}'.",
            adapter=adapter,
            code=ErrorCode.ADAPTER_NOT_FOUND,
            context={"resource": resource},
            cause=cause,
        )


class AdapterMalformedResponseError(AdapterError):
    """Raised when adapter returns a response that cannot be parsed."""

    def __init__(
        self,
        adapter: str,
        detail: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' returned unparseable response: {detail}",
            adapter=adapter,
            code=ErrorCode.ADAPTER_MALFORMED_RESPONSE,
            context={"detail": detail},
            cause=cause,
        )


class AdapterServerError(AdapterError):
    """Raised when adapter returns 5xx server error. Retryable."""

    def __init__(
        self,
        adapter: str,
        status_code: int,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' returned server error {status_code}.",
            adapter=adapter,
            code=ErrorCode.ADAPTER_SERVER_ERROR,
            context={"status_code": status_code},
            cause=cause,
        )


class AdapterConnectionError(AdapterError):
    """Raised on network-level failures (DNS, connection refused). Retryable."""

    def __init__(
        self,
        adapter: str,
        detail: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Adapter '{adapter}' connection failed: {detail}",
            adapter=adapter,
            code=ErrorCode.ADAPTER_CONNECTION_ERROR,
            context={"detail": detail},
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class PipelineError(DataScoutError):
    """Base class for pipeline-level failures."""
    pass


class InsufficientAdaptersError(PipelineError):
    """
    Raised by orchestrator when fewer than min_successful_adapters returned results.
    Signals partial-result or total-failure condition.
    """

    def __init__(
        self,
        succeeded: int,
        total: int,
        minimum: int,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Only {succeeded}/{total} adapters succeeded. "
                f"Minimum required: {minimum}."
            ),
            code=ErrorCode.INSUFFICIENT_ADAPTERS,
            context={"succeeded": succeeded, "total": total, "minimum": minimum},
            cause=cause,
        )
        self.succeeded = succeeded
        self.total = total
        self.minimum = minimum


class DedupFailureError(PipelineError):
    """Raised when fingerprint computation fails for a dataset."""

    def __init__(
        self,
        dataset_id: str,
        detail: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Deduplication failed for dataset '{dataset_id}': {detail}",
            code=ErrorCode.DEDUP_FAILURE,
            context={"dataset_id": dataset_id, "detail": detail},
            cause=cause,
        )


class ScoringFailureError(PipelineError):
    """Raised when Agent-1 scoring raises an exception."""

    def __init__(
        self,
        dataset_id: str,
        detail: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Scoring failed for dataset '{dataset_id}': {detail}",
            code=ErrorCode.SCORING_FAILURE,
            context={"dataset_id": dataset_id, "detail": detail},
            cause=cause,
        )


class PipelineTimeoutError(PipelineError):
    """Raised when global pipeline SLA is exceeded."""

    def __init__(self, timeout_s: float, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"Pipeline exceeded global SLA of {timeout_s}s.",
            code=ErrorCode.PIPELINE_TIMEOUT,
            context={"timeout_s": timeout_s},
            cause=cause,
        )


# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class InfrastructureError(DataScoutError):
    """Base class for infrastructure-level failures."""
    pass


class CircuitBreakerOpenError(InfrastructureError):
    """
    Raised when a circuit breaker is OPEN and rejecting requests fast.
    Retryable after will_retry_at timestamp.
    Orchestrator catches this and marks the adapter unavailable without blocking.
    """

    def __init__(
        self,
        service: str,
        will_retry_at: datetime,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Circuit breaker OPEN for '{service}'. "
                f"Will probe recovery at {will_retry_at.isoformat()}."
            ),
            code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            context={"service": service, "will_retry_at": will_retry_at.isoformat()},
            cause=cause,
        )
        self.service = service
        self.will_retry_at = will_retry_at


class RetryExhaustedError(InfrastructureError):
    """Raised when all retry attempts are consumed without success."""

    def __init__(
        self,
        operation: str,
        max_attempts: int,
        last_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"All {max_attempts} retry attempts exhausted for '{operation}'.",
            code=ErrorCode.RETRY_EXHAUSTED,
            context={"operation": operation, "max_attempts": max_attempts},
            cause=last_error,
        )
        self.max_attempts = max_attempts


# ─────────────────────────────────────────────────────────────────────────────
# LLM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class LLMError(DataScoutError):
    """Base class for LLM API failures."""

    def __init__(
        self,
        message: str,
        model: str,
        code: ErrorCode = ErrorCode.LLM_TIMEOUT,
        context: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = {"model": model, **(context or {})}
        super().__init__(message=message, code=code, context=ctx, cause=cause)
        self.model = model


class LLMTimeoutError(LLMError):
    """Raised when LLM API request times out. Retryable."""

    def __init__(self, model: str, timeout_s: float, cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"LLM '{model}' timed out after {timeout_s}s.",
            model=model,
            code=ErrorCode.LLM_TIMEOUT,
            context={"timeout_s": timeout_s},
            cause=cause,
        )


class LLMContextOverflowError(LLMError):
    """
    Raised when LLM input exceeds token limit.
    NOT retryable — must reduce input size first.
    """

    def __init__(
        self,
        model: str,
        tokens: int,
        max_tokens: int,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"LLM '{model}' input {tokens} tokens exceeds limit of {max_tokens}.",
            model=model,
            code=ErrorCode.LLM_CONTEXT_OVERFLOW,
            context={"tokens": tokens, "max_tokens": max_tokens},
            cause=cause,
        )


class LLMFallbackExhaustedError(LLMError):
    """Raised when all configured LLM fallbacks have failed."""

    def __init__(self, models: list[str], cause: Optional[Exception] = None) -> None:
        super().__init__(
            message=f"All LLM fallbacks exhausted. Models tried: {models}.",
            model=models[-1] if models else "unknown",
            code=ErrorCode.LLM_FALLBACK_EXHAUSTED,
            context={"models_tried": models},
            cause=cause,
        )


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is hit. Retryable."""

    def __init__(
        self,
        model: str,
        retry_after: Optional[float] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"LLM '{model}' rate-limited. Retry after {retry_after or 30}s.",
            model=model,
            code=ErrorCode.LLM_RATE_LIMITED,
            context={"retry_after": retry_after},
            cause=cause,
        )
        if retry_after is not None:
            self.retry_after_seconds = retry_after