"""
datascout.infrastructure.retry
─────────────────────────────────────────────────────
Async retry logic with exponential backoff, jitter, and per-operation config.

FIX: This file had the wrong content (it contained adapter registry code).
     Replaced with the actual retry implementation that infrastructure/__init__.py
     imports from.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

logger = logging.getLogger("datascout.infrastructure.retry")


# ─────────────────────────────────────────────────────────────────────────────
# RETRY CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """
    Configuration for retry behaviour on a single operation.

    max_attempts:     Total attempts including the first (not just retries)
    base_delay_s:     Initial backoff delay in seconds
    max_delay_s:      Cap on exponential backoff
    exponential_base: Multiplier per attempt (2.0 = double each time)
    jitter:           Add random noise to avoid thundering herd
    retryable_exceptions: Exception types that trigger a retry
    """
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,)


# ─────────────────────────────────────────────────────────────────────────────
# PRESET CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

ADAPTER_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay_s=1.0,
    max_delay_s=10.0,
    exponential_base=2.0,
    jitter=True,
)

LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay_s=2.0,
    max_delay_s=15.0,
    exponential_base=2.0,
    jitter=True,
)

HEALTH_CHECK_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay_s=0.5,
    max_delay_s=3.0,
    exponential_base=2.0,
    jitter=False,
)

# HTTP status codes that should trigger a retry
RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# HTTP status codes that should NOT be retried (client errors)
NON_RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 422})


# ─────────────────────────────────────────────────────────────────────────────
# BACKOFF COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Compute backoff delay for a given attempt number (0-indexed).

    Formula: min(base * exponential_base^attempt, max_delay) + optional jitter
    """
    delay = min(
        config.base_delay_s * (config.exponential_base ** attempt),
        config.max_delay_s,
    )
    if config.jitter:
        # Add up to 25% random jitter to prevent thundering herd
        delay += random.uniform(0, delay * 0.25)
    return delay


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC RETRY DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig = ADAPTER_RETRY_CONFIG,
    operation: str = "unknown",
    **kwargs: Any,
) -> Any:
    """
    Execute an async callable with retry logic.

    Args:
        func:      Async callable to retry
        *args:     Positional args passed to func
        config:    RetryConfig controlling backoff and attempts
        operation: Name for logging
        **kwargs:  Keyword args passed to func

    Raises:
        Last exception if all attempts exhausted.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            # Never retry CancelledError — propagate immediately
            raise
        except config.retryable_exceptions as exc:
            last_exc = exc
            if attempt == config.max_attempts - 1:
                logger.warning(
                    "retry_exhausted",
                    extra={
                        "operation": operation,
                        "max_attempts": config.max_attempts,
                        "error": str(exc)[:100],
                        "error_type": type(exc).__name__,
                    },
                )
                break

            delay = compute_backoff(attempt, config)
            logger.debug(
                "retry_attempt",
                extra={
                    "operation": operation,
                    "attempt": attempt + 1,
                    "max_attempts": config.max_attempts,
                    "delay_s": round(delay, 2),
                    "error_type": type(exc).__name__,
                },
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"retry_async: no result after {config.max_attempts} attempts")


# ─────────────────────────────────────────────────────────────────────────────
# DECORATOR FORM
# ─────────────────────────────────────────────────────────────────────────────

def with_retry(
    config: RetryConfig = ADAPTER_RETRY_CONFIG,
    operation: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator that wraps an async function with retry logic.

    Usage:
        @with_retry(config=ADAPTER_RETRY_CONFIG, operation="kaggle.search")
        async def search(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_async(
                func,
                *args,
                config=config,
                operation=operation or func.__qualname__,
                **kwargs,
            )
        return wrapper
    return decorator