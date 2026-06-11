"""
datascout.contracts.errors
─────────────────────────────────────────────────────
Public surface for the errors subpackage.
All downstream code imports from here — never from internal modules directly.
"""

from .codes import (
    ErrorCode,
    ErrorMeta,
    ErrorSeverity,
    ERROR_REGISTRY,
    is_retryable,
    get_http_status,
    get_retry_after,
    get_severity,
    get_category,
    is_validation_error,
    is_adapter_error,
    is_infrastructure_error,
)
from .exceptions import (
    DataScoutError,
    # Validation
    ValidationError,
    FingerprintInvalidError,
    LineageMissingError,
    EmbeddingDimensionMismatchError,
    SchemaVersionMismatchError,
    InvalidSourceError,
    QueryTooLongError,
    # Adapter
    AdapterError,
    AdapterTimeoutError,
    AdapterRateLimitError,
    AdapterAuthError,
    AdapterNotFoundError,
    AdapterMalformedResponseError,
    AdapterServerError,
    AdapterConnectionError,
    # Pipeline
    PipelineError,
    InsufficientAdaptersError,
    DedupFailureError,
    ScoringFailureError,
    PipelineTimeoutError,
    # Infrastructure
    InfrastructureError,
    CircuitBreakerOpenError,
    RetryExhaustedError,
    # LLM
    LLMError,
    LLMTimeoutError,
    LLMContextOverflowError,
    LLMFallbackExhaustedError,
    LLMRateLimitError,
)

__all__ = [
    # Codes
    "ErrorCode", "ErrorMeta", "ErrorSeverity", "ERROR_REGISTRY",
    "is_retryable", "get_http_status", "get_retry_after",
    "get_severity", "get_category",
    "is_validation_error", "is_adapter_error", "is_infrastructure_error",
    # Exceptions
    "DataScoutError",
    "ValidationError", "FingerprintInvalidError", "LineageMissingError",
    "EmbeddingDimensionMismatchError", "SchemaVersionMismatchError",
    "InvalidSourceError", "QueryTooLongError",
    "AdapterError", "AdapterTimeoutError", "AdapterRateLimitError",
    "AdapterAuthError", "AdapterNotFoundError", "AdapterMalformedResponseError",
    "AdapterServerError", "AdapterConnectionError",
    "PipelineError", "InsufficientAdaptersError", "DedupFailureError",
    "ScoringFailureError", "PipelineTimeoutError",
    "InfrastructureError", "CircuitBreakerOpenError", "RetryExhaustedError",
    "LLMError", "LLMTimeoutError", "LLMContextOverflowError",
    "LLMFallbackExhaustedError", "LLMRateLimitError",
]