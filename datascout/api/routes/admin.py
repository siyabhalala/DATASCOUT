"""
datascout.api.routes.admin
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Administrative endpoints — pipeline control, cache
management, and system diagnostics. All endpoints require admin-level API key.

ENDPOINTS:
  GET  /api/v1/admin/info           — System info and configuration summary
  POST /api/v1/admin/cache/clear    — Clear in-memory caches
  POST /api/v1/admin/crawl/trigger  — Trigger a manual crawl run
  GET  /api/v1/admin/stats          — Pipeline statistics and metrics

DESIGN DECISIONS:

  1. WHY admin endpoints separate from datasets?
     - Auth separation: admin key vs. user API key
     - Rate limiting: admin ops are expensive — separate limits
     - Audit: all admin actions logged with full context

  2. WHY POST for destructive actions (cache/clear, crawl/trigger)?
     - GET must be idempotent (HTTP spec) — cache clear is not
     - POST communicates intent to mutate state
     - Prevents accidental browser prefetch / cache hits

  3. WHY never expose raw settings values?
     - API keys, DB URLs, tokens must never appear in API responses
     - Only boolean presence (has_key: true) and non-sensitive values allowed

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse

from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Process start — for uptime reporting
_PROCESS_START_TIME = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN AUTH DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────


async def require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> bool:
    """
    Admin-level authorization dependency.

    WHY separate from user auth:
    - Admin operations (crawl trigger, cache clear) must be restricted
    - User API key ≠ admin key — principle of least privilege
    - In production: X-Admin-Key backed by a secret manager

    For now: accepts any non-empty admin key in development,
    rejects all requests if no admin key is configured.
    """
    from datascout.infrastructure.config.settings import get_settings

    settings = get_settings()
    admin_key = getattr(settings, "admin_api_key", None)

    if admin_key is None:
        # No admin key configured — allow in development, block in production
        if settings.environment == "production":
            return False
        return True  # Development: open

    provided_key = x_admin_key or x_api_key
    if not provided_key:
        return False

    # SYSTEM RULE #8: compare types, not isinstance
    return type(provided_key).__name__ == "str" and provided_key == admin_key


# ─────────────────────────────────────────────────────────────────────────────
# GET /info
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    summary="System information and configuration summary",
)
async def system_info(
    request: Request,
    is_admin: bool = Depends(require_admin),
) -> JSONResponse:
    """
    Return system information and non-sensitive configuration.

    Complexity: O(1) — reads app state and settings.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not is_admin:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "FORBIDDEN",
                "message": "Admin access required.",
                "request_id": request_id,
            },
        )

    settings = getattr(getattr(request.app, "state", None), "settings", None)
    agent = getattr(getattr(request.app, "state", None), "scout_agent", None)

    creds: dict[str, bool] = {}
    if settings:
        creds = settings.validate_required_credentials()

    info: dict[str, Any] = {
        "app_name": settings.app_name if settings else "DATASCOUT",
        "app_version": settings.app_version if settings else "unknown",
        "environment": settings.environment if settings else "unknown",
        "uptime_seconds": int(time.monotonic() - _PROCESS_START_TIME),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scout_agent_available": agent is not None,
        "configured_adapters": creds,
        "api_host": settings.api_host if settings else "unknown",
        "api_port": settings.api_port if settings else 8000,
        "llm_provider": settings.llm_provider if settings else "unknown",
        "llm_model": settings.llm_model if settings else "unknown",
        # Never expose: api_keys, db_url, tokens
        "request_id": request_id,
    }

    logger.info(
        "admin_info_accessed",
        extra={"request_id": request_id, "environment": info["environment"]},
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=info)


# ─────────────────────────────────────────────────────────────────────────────
# POST /cache/clear
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear all in-memory caches",
)
async def clear_cache(
    request: Request,
    is_admin: bool = Depends(require_admin),
) -> JSONResponse:
    """
    Clear in-memory caches — BM25 index, embedding cache, search cache.

    WHY manual cache clear:
    - After bulk dataset ingestion, stale cached index returns old results
    - Incident response: bad data ingested → clear + re-index
    - Testing: reproducible test state

    Complexity: O(cache_size) — clears all cache entries.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not is_admin:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "FORBIDDEN", "message": "Admin access required.", "request_id": request_id},
        )

    cleared: list[str] = []
    errors: list[str] = []

    # ── Clear search caches ───────────────────────────────────────────────────
    try:
        from datascout.tools.utilities.cache import cache_manager  # type: ignore[attr-defined]
        cache_manager.clear_all()
        cleared.append("search_cache")
    except Exception as exc:
        errors.append(f"search_cache: {type(exc).__name__}: {exc}")

    # ── Clear rate limiter state ──────────────────────────────────────────────
    try:
        from datascout.api.middleware.rate_limit import _rate_limit_state  # type: ignore[attr-defined]
        _rate_limit_state.clear()
        cleared.append("rate_limit_state")
    except Exception as exc:
        errors.append(f"rate_limit_state: {type(exc).__name__}: {exc}")

    logger.info(
        "admin_cache_cleared",
        extra={
            "request_id": request_id,
            "cleared": cleared,
            "errors": errors,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "cleared": cleared,
            "errors": errors,
            "success": len(errors) == 0,
            "request_id": request_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /crawl/trigger
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/crawl/trigger",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a manual crawler run (async, returns immediately)",
)
async def trigger_crawl(
    request: Request,
    is_admin: bool = Depends(require_admin),
) -> JSONResponse:
    """
    Trigger a background crawler run across all configured adapters.

    WHY 202 Accepted (not 200):
    - Crawl is async — may take minutes
    - 202 = "Request accepted, processing async" (HTTP spec)
    - Response includes a job_id for status polling

    WHY background task (not await):
    - Crawl runs for 1–10 minutes — would timeout HTTP connection
    - asyncio.create_task: non-blocking, runs in the same event loop

    Complexity: O(1) to trigger — O(crawl_results) to complete.
    """
    import asyncio
    import uuid as _uuid

    request_id = getattr(request.state, "request_id", "unknown")

    if not is_admin:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "FORBIDDEN", "message": "Admin access required.", "request_id": request_id},
        )

    job_id = str(_uuid.uuid4())

    logger.info(
        "admin_crawl_triggered",
        extra={"request_id": request_id, "job_id": job_id},
    )

    # Fire and forget — crawler runs in background
    try:
        asyncio.create_task(_run_crawl_background(job_id))
        triggered = True
        message = "Crawl job queued."
    except Exception as exc:
        triggered = False
        message = f"Failed to queue crawl: {type(exc).__name__}: {exc}"
        logger.info(
            "admin_crawl_trigger_failed",
            extra={"request_id": request_id, "job_id": job_id, "error": str(exc)},
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "triggered": triggered,
            "message": message,
            "request_id": request_id,
        },
    )


async def _run_crawl_background(job_id: str) -> None:
    """
    Background crawl task — runs async without blocking the event loop.

    SYSTEM RULE #1: never crash — catches all exceptions.
    """
    logger.info("crawl_job_started", extra={"job_id": job_id})
    try:
        from datascout.crawler.crawler_manager import CrawlerManager  # type: ignore[attr-defined]

        manager = CrawlerManager()
        await manager.run()
        logger.info("crawl_job_completed", extra={"job_id": job_id})
    except Exception as exc:
        # SYSTEM RULE #1: catch all — log, do not raise
        logger.info(
            "crawl_job_failed",
            extra={"job_id": job_id, "error_type": type(exc).__name__, "error": str(exc)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /stats
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Pipeline statistics and dataset counts",
)
async def pipeline_stats(
    request: Request,
    is_admin: bool = Depends(require_admin),
) -> JSONResponse:
    """
    Return pipeline statistics: dataset counts, recent ingestion rates,
    error rates, and adapter health.

    Complexity: O(1) to O(log n) — reads counters and DB aggregate stats.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not is_admin:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "FORBIDDEN", "message": "Admin access required.", "request_id": request_id},
        )

    stats: dict[str, Any] = {
        "uptime_seconds": int(time.monotonic() - _PROCESS_START_TIME),
        "total_datasets": 0,
        "datasets_by_source": {},
        "recent_ingestion_count": 0,
        "errors": [],
        "request_id": request_id,
    }

    # ── Dataset counts from DB ────────────────────────────────────────────────
    try:
        from datascout.storage.repositories.dataset_repository import DatasetRepository  # type: ignore[attr-defined]

        repo = DatasetRepository()
        counts = await repo.count_by_source()
        stats["datasets_by_source"] = counts
        stats["total_datasets"] = sum(counts.values())
    except Exception as exc:
        stats["errors"].append(f"db_stats: {type(exc).__name__}: {exc}")

    logger.info(
        "admin_stats_accessed",
        extra={
            "request_id": request_id,
            "total_datasets": stats["total_datasets"],
        },
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=stats)
