"""
datascout.contracts.errors.handlers
-----------------------------------
Utility functions for exception handling and recovery.

Key functions:
- exception_to_response: Convert any exception to AgentResponse
- is_transient: Classify if exception is retryable
- should_retry: Decision logic for retry attempts
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from datascout.contracts.errors.base import DataScoutError
from datascout.contracts.errors.codes import ErrorCode
from datascout.contracts.states import AgentState

if TYPE_CHECKING:
    from datascout.contracts.responses import AgentResponse


def exception_to_response(
    exc: Exception,
    query_id: str,
    context_id: str | None = None,
) -> "AgentResponse":
    """
    Convert any exception to a terminal FAILED AgentResponse.

    This is the bridge between exception-based error handling
    and the contract-based response model.

    Used by:
    - Agent controller's top-level try/except
    - Orchestrators catching sub-agent failures
    - API handlers catching endpoint failures

    Args:
        exc: The exception to convert
        query_id: AgentQuery.query_id for traceability
        context_id: Optional ExecutionContext.context_id for full trace

    Returns:
        AgentResponse in FAILED state with structured error info
    """
    from datascout.contracts.responses import AgentResponse

    # Extract structured info if it's a DataScoutError
    if isinstance(exc, DataScoutError):
        error_code = exc.error_code
        error_summary = exc.message
        # Include context dict in summary if present
        if exc.context:
            error_summary += f" | Context: {exc.context}"
    else:
        # Generic Python exception — fallback to UNKNOWN_ERROR
        error_code = ErrorCode.UNKNOWN_ERROR
        error_summary = f"{exc.__class__.__name__}: {str(exc)}"

    return AgentResponse(
        query_id=query_id,
        terminal_state=AgentState.FAILED,
        error_code=error_code,
        error_summary=error_summary,
        results=[],
        metrics=None,  # Metrics not computable if we failed
    )


def is_transient(exc: Exception) -> bool:
    """
    Classify if exception represents a transient failure.

    Transient failures are candidates for retry (with backoff).
    Permanent failures should NOT be retried — they require
    input correction or config change.

    Returns:
        True if exception is marked as transient, False otherwise
    """
    if isinstance(exc, DataScoutError):
        return exc.is_transient
    # Unknown exceptions are assumed permanent to avoid infinite retry loops
    return False


def should_retry(
    exc: Exception,
    attempt: int,
    max_attempts: int,
) -> bool:
    """
    Retry decision logic.

    Combines transient classification with attempt budget.

    Args:
        exc: The exception to evaluate
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum allowed attempts

    Returns:
        True if caller should retry, False otherwise
    """
    if attempt >= max_attempts:
        return False

    return is_transient(exc)


def format_exception_for_logging(exc: Exception) -> dict:
    """
    Serialize exception to structured dict for logging.

    Includes:
    - Exception class and message
    - Error code (if DataScoutError)
    - Context dict (if DataScoutError)
    - Full traceback

    Returns:
        Dict suitable for structured logger
    """
    result = {
        "exception_class": exc.__class__.__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
    }

    if isinstance(exc, DataScoutError):
        result.update({
            "error_code": exc.error_code,
            "context": exc.context,
            "is_transient": exc.is_transient,
        })

    return result


def wrap_external_exception(
    exc: Exception,
    layer: str,
    message_override: str | None = None,
) -> DataScoutError:
    """
    Wrap a non-DataScoutError exception into the hierarchy.

    Use when catching exceptions from external libraries (requests, httpx, etc.)
    to preserve stack trace while adding DATASCOUT structure.

    Args:
        exc: The external exception
        layer: Which layer caught it ("tool", "llm", "infra")
        message_override: Optional custom message (defaults to str(exc))

    Returns:
        DataScoutError subclass appropriate for the layer

    Example:
        try:
            response = requests.get(url, timeout=10)
        except requests.Timeout as e:
            raise wrap_external_exception(e, "tool") from e
    """
    from datascout.contracts.errors.exceptions import (
        InfrastructureError,
        LLMAPIError,
        ToolAPIError,
    )

    message = message_override or f"{exc.__class__.__name__}: {str(exc)}"
    context = {
        "original_exception": exc.__class__.__name__,
        "original_message": str(exc),
    }

    if layer == "tool":
        return ToolAPIError(message=message, context=context)
    elif layer == "llm":
        return LLMAPIError(message=message, context=context)
    elif layer == "infra":
        return InfrastructureError(
            message=message,
            error_code=ErrorCode.INFRA_DEPENDENCY_UNAVAILABLE,
            context=context,
        )
    else:
        # Fallback — should not happen if layer param is validated
        from datascout.contracts.errors.base import DataScoutError
        return DataScoutError(
            message=message,
            error_code=ErrorCode.UNKNOWN_ERROR,
            context=context,
        )