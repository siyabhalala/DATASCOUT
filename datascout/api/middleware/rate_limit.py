"""
datascout.api.middleware.rate_limit
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Token bucket rate limiting middleware with per-client
tracking, bypass for health endpoints, and graceful degradation.

ALGORITHM: Token Bucket
  - Each client starts with `burst` tokens
  - Tokens refill at `rate` per second (fractional)
  - Each request costs 1 token
  - Request rejected if bucket is empty (0 tokens)

WHY Token Bucket (not sliding window, not fixed window)?
  - Fixed window: 100 req/min — 100 at 00:59, 100 at 01:00 = 200 in 2s burst
  - Sliding window: O(requests) memory per client — expensive at scale
  - Token bucket: O(1) memory per client — only stores count + last refill time
  - Token bucket handles bursts gracefully (burst tokens allow short spikes)

DESIGN DECISIONS:

  1. WHY in-memory state (not Redis)?
     - Redis adds network latency (1–5ms) to every request — expensive
     - In-memory is O(1) and sub-microsecond
     - For multi-instance deployments: use Redis-backed rate limiter
     - For single-instance (current): in-memory is correct

  2. WHY keyed by client IP (not API key)?
     - API key rotation would reset limits unexpectedly
     - IP is stable within a session
     - Shared IPs (NAT, CDN): acceptable — rate limits are generous enough

  3. WHY Retry-After header on 429?
     - Clients can back off intelligently instead of hammering
     - Standard HTTP spec — all HTTP clients understand it
     - Reduces retry storm load

  4. WHY module-level _rate_limit_state dict?
     - Shared state across requests in the same process
     - Admin clear endpoint can reset it (admin.py imports _rate_limit_state)
     - Alternative (class instance): same behavior, more boilerplate

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from threading import Lock
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)

# ── Module-level state — shared across all requests in this process ────────────
# Structure: {client_key: {"tokens": float, "last_refill": float}}
_rate_limit_state: dict[str, dict[str, float]] = defaultdict(
    lambda: {"tokens": 0.0, "last_refill": 0.0}
)
_state_lock = Lock()

# ── Default limits ─────────────────────────────────────────────────────────────
_DEFAULT_RATE: float = 10.0   # Requests per second (sustained)
_DEFAULT_BURST: int = 30      # Max burst tokens (short spike allowance)

# ── Paths exempt from rate limiting ───────────────────────────────────────────
_EXEMPT_PREFIXES: tuple[str, ...] = ("/health/", "/metrics")
_EXEMPT_PATHS: frozenset[str] = frozenset({"/health/live", "/health/ready"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.

    Per-client (IP) token bucket. Thread-safe via Lock.
    Health endpoints are exempt to ensure Kubernetes probes always succeed.

    Complexity: O(1) per request — dictionary lookup + arithmetic.
    """

    def __init__(
        self,
        app: Any,
        rate: float = _DEFAULT_RATE,
        burst: int = _DEFAULT_BURST,
    ) -> None:
        super().__init__(app)
        self.rate = rate
        self.burst = burst

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limit and either pass the request or return 429.

        Flow:
          1. Exempt check — health/metrics bypass
          2. Get client key (IP)
          3. Refill tokens since last request
          4. Check token count → allow or reject
        """
        path = request.url.path

        # ── Exempt paths bypass rate limiting ─────────────────────────────────
        if path in _EXEMPT_PATHS or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        client_key = _get_client_key(request)
        now = time.monotonic()

        # ── Token bucket logic (thread-safe) ───────────────────────────────────
        with _state_lock:
            state = _rate_limit_state[client_key]

            # Initialize on first request
            if state["last_refill"] == 0.0:
                state["tokens"] = float(self.burst)
                state["last_refill"] = now
            else:
                # Refill: add tokens proportional to elapsed time
                elapsed = now - state["last_refill"]
                new_tokens = elapsed * self.rate
                state["tokens"] = min(float(self.burst), state["tokens"] + new_tokens)
                state["last_refill"] = now

            # Check if request can proceed
            if state["tokens"] >= 1.0:
                state["tokens"] -= 1.0
                remaining = int(state["tokens"])
                allowed = True
            else:
                # Time until 1 token refills
                wait_seconds = (1.0 - state["tokens"]) / self.rate
                remaining = 0
                allowed = False

        if allowed:
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.burst)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(now + (1.0 / self.rate)))
            return response

        # ── Rate limit exceeded ────────────────────────────────────────────────
        request_id = getattr(request.state, "request_id", "unknown")
        retry_after = math.ceil(wait_seconds)  # type: ignore[possibly-undefined]

        logger.info(
            "rate_limit_exceeded",
            extra={
                "request_id": request_id,
                "client_key": client_key,
                "path": path,
                "retry_after_seconds": retry_after,
            },
        )

        return JSONResponse(
            status_code=429,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(self.burst),
                "X-RateLimit-Remaining": "0",
            },
            content={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Too many requests. Retry after {retry_after}s.",
                "retry_after_seconds": retry_after,
                "request_id": request_id,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_client_key(request: Request) -> str:
    """
    Derive the rate limit bucket key for a client.

    Priority: X-Forwarded-For → real client IP → "unknown"
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# Fix missing Any import
from typing import Any  # noqa: E402
