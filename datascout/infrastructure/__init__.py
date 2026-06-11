"""
datascout.infrastructure
─────────────────────────────────────────────────────
Public surface for the infrastructure package.
All agents and adapters import from here.
"""

from .logging import (
    configure_logging,
    get_logger,
    log_performance,
    set_request_context,
    get_request_context,
    request_id_var,
    user_id_var,
    session_id_var,
    agent_id_var,
)
from .monitoring import (
    metrics,
    Metrics,
    timer,
    start_metrics_server,
    LATENCY_BUCKETS,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_registry,
)
from .retry import (
    RetryConfig,
    retry_async,
    with_retry,
    compute_backoff,
    ADAPTER_RETRY_CONFIG,
    LLM_RETRY_CONFIG,
    HEALTH_CHECK_RETRY_CONFIG,
    RETRYABLE_HTTP_STATUS_CODES,
    NON_RETRYABLE_HTTP_STATUS_CODES,
)
from .health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    HealthResponse,
    health_checker,
    make_circuit_breaker_check,
)

__all__ = [
    # Logging
    "configure_logging", "get_logger", "log_performance",
    "set_request_context", "get_request_context",
    "request_id_var", "user_id_var", "session_id_var", "agent_id_var",
    # Monitoring
    "metrics", "Metrics", "timer", "start_metrics_server", "LATENCY_BUCKETS",
    # Circuit Breaker
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerRegistry",
    "CircuitState", "circuit_registry",
    # Retry
    "RetryConfig", "retry_async", "with_retry", "compute_backoff",
    "ADAPTER_RETRY_CONFIG", "LLM_RETRY_CONFIG", "HEALTH_CHECK_RETRY_CONFIG",
    "RETRYABLE_HTTP_STATUS_CODES", "NON_RETRYABLE_HTTP_STATUS_CODES",
    # Health
    "HealthChecker", "HealthStatus", "ComponentHealth", "HealthResponse",
    "health_checker", "make_circuit_breaker_check",
]