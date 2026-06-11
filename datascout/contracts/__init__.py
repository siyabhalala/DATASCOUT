"""
datascout.contracts
─────────────────────────────────────────────────────
Public API surface for the entire contracts package.

All downstream agents, adapters, and infrastructure import from here.
Never import directly from internal submodules.

  # Correct:
  from contracts import RawDataset, SearchQuery, AgentResponse

  # Wrong:
  from contracts.models import RawDataset
"""

from .states import (
    ValidationMode,
    DataDomain,
    DataFormat,
    LicenseType,
    StageStatus,
    DataSource,
    QualityTier,
    VALID_SOURCES,
    get_validation_mode,
    normalize_domain,
    normalize_format,
    normalize_license,
    is_commercial_allowed,
    license_display_name,
    compute_quality_tier,
)
from .task_types import (
    TaskType,
    Modality,
    TaskCompatibility,
    TASK_MODALITY_MAP,
    normalize_task_type,
    normalize_modality,
    compute_task_compatibility,
)
from .errors import (
    ErrorCode,
    ErrorMeta,
    ErrorSeverity,
    ERROR_REGISTRY,
    is_retryable,
    get_http_status,
    get_retry_after,
    get_severity,
    get_category,
    DataScoutError,
    ValidationError,
    FingerprintInvalidError,
    LineageMissingError,
    EmbeddingDimensionMismatchError,
    SchemaVersionMismatchError,
    InvalidSourceError,
    QueryTooLongError,
    AdapterError,
    AdapterTimeoutError,
    AdapterRateLimitError,
    AdapterAuthError,
    AdapterNotFoundError,
    AdapterMalformedResponseError,
    AdapterServerError,
    AdapterConnectionError,
    PipelineError,
    InsufficientAdaptersError,
    DedupFailureError,
    ScoringFailureError,
    PipelineTimeoutError,
    InfrastructureError,
    CircuitBreakerOpenError,
    RetryExhaustedError,
    LLMError,
    LLMTimeoutError,
    LLMContextOverflowError,
    LLMFallbackExhaustedError,
    LLMRateLimitError,
)
from .models import (
    EmbeddingModelConfig,
    ScoreDimension,
    RawDataset,
    EvaluatedDataset,
    CURRENT_SCHEMA_VERSION,
    CURRENT_PIPELINE_VERSION,
    COMPLETENESS_FIELDS,
    MAX_DESCRIPTION_CLEAN,
    MAX_DESCRIPTION_SHORT,
    MAX_TAGS_PRIMARY,
    compute_fingerprint,
    truncate_to_sentence_boundary,
    truncate_to_word_boundary,
    compute_primary_tags,
    compute_completeness,
)
from .requests import (
    SearchFilters,
    SearchPreferences,
    SearchQuery,
    AgentQuery,
)
from .responses import (
    ConfidenceLevel,
    PipelineStage,
    LLMMetadata,
    StructuredExplanation,
    DecisionTrace,
    RankedResult,
    AgentResponse,
    compute_confidence,
)

__version__ = "3.0.0"

__all__ = [
    # States
    "ValidationMode", "DataDomain", "DataFormat", "LicenseType",
    "StageStatus", "DataSource", "QualityTier", "VALID_SOURCES",
    "get_validation_mode", "normalize_domain", "normalize_format",
    "normalize_license", "is_commercial_allowed", "license_display_name",
    "compute_quality_tier",
    # Task types
    "TaskType", "Modality", "TaskCompatibility", "TASK_MODALITY_MAP",
    "normalize_task_type", "normalize_modality", "compute_task_compatibility",
    # Errors
    "ErrorCode", "ErrorMeta", "ErrorSeverity", "ERROR_REGISTRY",
    "is_retryable", "get_http_status", "get_retry_after", "get_severity", "get_category",
    "DataScoutError", "ValidationError", "FingerprintInvalidError",
    "LineageMissingError", "EmbeddingDimensionMismatchError",
    "SchemaVersionMismatchError", "InvalidSourceError", "QueryTooLongError",
    "AdapterError", "AdapterTimeoutError", "AdapterRateLimitError",
    "AdapterAuthError", "AdapterNotFoundError", "AdapterMalformedResponseError",
    "AdapterServerError", "AdapterConnectionError",
    "PipelineError", "InsufficientAdaptersError", "DedupFailureError",
    "ScoringFailureError", "PipelineTimeoutError",
    "InfrastructureError", "CircuitBreakerOpenError", "RetryExhaustedError",
    "LLMError", "LLMTimeoutError", "LLMContextOverflowError",
    "LLMFallbackExhaustedError", "LLMRateLimitError",
    # Models
    "EmbeddingModelConfig", "ScoreDimension", "RawDataset", "EvaluatedDataset",
    "CURRENT_SCHEMA_VERSION", "CURRENT_PIPELINE_VERSION",
    "compute_fingerprint", "truncate_to_sentence_boundary",
    "truncate_to_word_boundary", "compute_primary_tags", "compute_completeness",
    # Requests
    "SearchFilters", "SearchPreferences", "SearchQuery", "AgentQuery",
    # Responses
    "ConfidenceLevel", "PipelineStage", "LLMMetadata",
    "StructuredExplanation", "DecisionTrace", "RankedResult", "AgentResponse",
    "compute_confidence",
    # Version
    "__version__",
]