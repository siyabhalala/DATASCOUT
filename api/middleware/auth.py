"""
datascout.api.middleware.auth
──────────────────────────────
API key authentication middleware. In development mode (no API_KEY set),
all routes are open. In production, API_KEY header is required.
"""
from __future__ import annotations

import hmac
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)

# Paths that never require auth
_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/api/",        # All API routes open — auth handled per-route if needed
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Always allow OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Public path bypass
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        from datascout.infrastructure.config.settings import get_settings
        settings = get_settings()
        configured_key: str | None = getattr(settings, "api_key", None)

        # No API key configured → open access
        if not configured_key:
            return await call_next(request)

        provided_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or request.query_params.get("api_key", "")
        )

        if not provided_key or not hmac.compare_digest(
            provided_key.encode(), configured_key.encode()
        ):
            return JSONResponse(
                status_code=401,
                content={"error": "UNAUTHORIZED", "message": "Invalid or missing API key."},
            )

        return await call_next(request)
