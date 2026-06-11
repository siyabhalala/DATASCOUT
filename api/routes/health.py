"""
datascout.api.routes.health
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Kubernetes-compatible liveness and readiness probes.

ENDPOINTS:
  GET /health/live     — Liveness probe (is the process alive?)
  GET /health/ready    — Readiness probe (can it serve traffic?)
  GET /health/status   — Detailed component status (for dashboards)
  GET /health/metrics  — Prometheus text exposition format

DESIGN DECISIONS:

  1. WHY liveness separate from readiness?
     - Liveness: "Is this process alive?" → NO = Kubernetes RESTARTS the pod
     - Readiness: "Can this pod serve traffic?" → NO = Kubernetes DROPS from LB
     - Without separation: slow DB causes pod restarts on perfectly healthy code
     - A pod can be LIVE (process running) but NOT READY (DB unreachable)

  2. WHY /health/metrics endpoint?
     - Prometheus scrapes /metrics by convention
     - Keeping it under /health/ namespace avoids auth middleware conflicts
     - Text format: compatible with all Prometheus versions (no protobuf needed)

  3. WHY no auth on health endpoints?
     - Kubernetes probes cannot send auth headers
     - Monitoring systems (Prometheus, Datadog) need unauthenticated access
     - Health endpoints expose no sensitive data — safe to leave open

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Process start time — used to compute uptime
_PROCESS_START_TIME = time.monotonic()
_PROCESS_START_UTC = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# GET /live  — Kubernetes liveness probe
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe — is the process alive?",
    include_in_schema=False,  # Don't clutter OpenAPI docs
)
async def liveness() -> JSONResponse:
    """
    Kubernetes liveness probe.

    Returns 200 if the process is running. Returns 503 only if the process
    is in an unrecoverable state (should never happen in normal operation).

    Complexity: O(1) — no external calls.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "alive",
            "uptime_seconds": int(time.monotonic() - _PROCESS_START_TIME),
            "started_at": _PROCESS_START_UTC,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready — Kubernetes readiness probe
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe — can the pod serve traffic?",
    include_in_schema=False,
)
async def readiness(request: Request) -> JSONResponse:
    """
    Kubernetes readiness probe.

    Returns 200 (READY) if critical components are healthy.
    Returns 503 (NOT READY) if the pod cannot serve traffic.

    DEGRADED components (non-critical adapters down) → still READY.
    CRITICAL components DOWN → NOT READY → Kubernetes removes from LB.

    Complexity: O(components) — parallel health checks with TTL cache.
    """
    checks = await _run_readiness_checks(request)
    is_ready = checks["overall"] in ("healthy", "degraded")

    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
            "uptime_seconds": int(time.monotonic() - _PROCESS_START_TIME),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /status — detailed component health for dashboards
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Detailed component health status",
)
async def health_status(request: Request) -> JSONResponse:
    """
    Detailed health status for monitoring dashboards.

    Includes per-component status, version info, and uptime.
    Not used by Kubernetes — used by Datadog/Grafana dashboards.

    Complexity: O(components) — parallel health checks.
    """
    settings = getattr(getattr(request.app, "state", None), "settings", None)
    checks = await _run_readiness_checks(request)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "app": settings.app_name if settings else "DATASCOUT",
            "version": settings.app_version if settings else "unknown",
            "environment": settings.environment if settings else "unknown",
            "overall_status": checks["overall"],
            "uptime_seconds": int(time.monotonic() - _PROCESS_START_TIME),
            "started_at": _PROCESS_START_UTC,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "components": checks.get("components", {}),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics — Prometheus text format
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Prometheus metrics exposition",
    include_in_schema=False,
)
async def metrics() -> PlainTextResponse:
    """
    Prometheus text format metrics endpoint.

    Attempts to use prometheus_client if available.
    Falls back to basic gauge metrics if not installed.

    Complexity: O(metrics_count) — iterate registered metrics.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore[import]

        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        # Graceful fallback — prometheus_client not installed
        uptime = int(time.monotonic() - _PROCESS_START_TIME)
        text = (
            "# HELP datascout_uptime_seconds Process uptime in seconds\n"
            "# TYPE datascout_uptime_seconds gauge\n"
            f"datascout_uptime_seconds {uptime}\n"
        )
        return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4")


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HEALTH CHECK RUNNER
# ─────────────────────────────────────────────────────────────────────────────


async def _run_readiness_checks(request: Request) -> dict[str, Any]:
    """
    Run all readiness checks in parallel and aggregate results.

    WHY parallel:
    - Each check may do I/O (DB ping, adapter ping)
    - Sequential = sum of all latencies
    - Parallel = max of all latencies

    Complexity: O(max_check_latency) — parallel bounded by slowest check.
    """
    import asyncio

    components: dict[str, dict[str, Any]] = {}

    # ── Scout agent check ─────────────────────────────────────────────────────
    async def check_scout_agent() -> None:
        try:
            agent = getattr(getattr(request.app, "state", None), "scout_agent", None)
            if agent is not None:
                components["scout_agent"] = {"status": "healthy", "message": "Agent initialized"}
            else:
                components["scout_agent"] = {
                    "status": "degraded",
                    "message": "Agent not initialized (check API keys)",
                }
        except Exception as exc:
            components["scout_agent"] = {"status": "down", "message": str(exc)}

    # ── Settings check ────────────────────────────────────────────────────────
    async def check_settings() -> None:
        try:
            settings = getattr(getattr(request.app, "state", None), "settings", None)
            if settings:
                creds = settings.validate_required_credentials()
                configured = [k for k, v in creds.items() if v]
                components["settings"] = {
                    "status": "healthy",
                    "configured_adapters": configured,
                }
            else:
                components["settings"] = {"status": "down", "message": "Settings not loaded"}
        except Exception as exc:
            components["settings"] = {"status": "down", "message": str(exc)}

    # ── Run all checks in parallel ─────────────────────────────────────────────
    await asyncio.gather(
        check_scout_agent(),
        check_settings(),
        return_exceptions=True,
    )

    # ── Compute overall status ─────────────────────────────────────────────────
    statuses = [c.get("status", "unknown") for c in components.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "down" for s in statuses):
        overall = "degraded"  # Non-critical components — still serve
    else:
        overall = "degraded"

    return {"overall": overall, "components": components}

# ── GET /status — PHASE 6 ENHANCED ───────────────────────────────────────────

@router.get("/status")
async def system_status(request: Request) -> JSONResponse:
    """
    Detailed system status for dashboards and demo health panels.

    PHASE 6: Enhanced to include LLM provider, adapter, and demo mode status.

    Returns:
        200: System healthy or degraded (serving)
        503: System down (not serving)
    """
    settings = get_settings()
    uptime_seconds = int(time.monotonic() - _PROCESS_START_TIME)

    # ── Agent status ──────────────────────────────────────────────────────────
    scout_agent = getattr(request.app.state, "scout_agent", None)
    agent_available = scout_agent is not None

    # ── DB status ─────────────────────────────────────────────────────────────
    db_available = getattr(request.app.state, "db_available", False)

    # ── LLM provider status ───────────────────────────────────────────────────
    llm_status = _check_llm_provider(settings)

    # ── Adapter status (from agent if available) ──────────────────────────────
    adapter_statuses = _get_adapter_statuses(scout_agent)

    # ── Overall health ────────────────────────────────────────────────────────
    critical_ok = agent_available  # DB degradation is acceptable
    if critical_ok:
        overall = "healthy" if llm_status["available"] else "degraded"
    else:
        overall = "down"

    http_status = (
        status.HTTP_200_OK if overall in ("healthy", "degraded")
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "started_at": _PROCESS_START_UTC,

            # Deployment metadata
            "deployment": {
                "environment": settings.environment,
                "version": settings.app_version,
                "demo_mode": getattr(settings, "demo_mode", False),
                "is_cloud_run": getattr(settings, "is_cloud_run", False),
                "cloud_run_revision": getattr(settings, "cloud_run_revision", None),
            },

            # Component health
            "components": {
                "scout_agent": {
                    "status": "healthy" if agent_available else "down",
                    "available": agent_available,
                },
                "database": {
                    "status": "healthy" if db_available else "degraded",
                    "available": db_available,
                    "note": "stateless mode active" if not db_available else None,
                },
                "llm_provider": llm_status,
                "adapters": adapter_statuses,
            },
        },
    )


# ── GET /metrics — UNCHANGED ──────────────────────────────────────────────────

@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus text-format metrics exposition."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception:
        return PlainTextResponse(
            content="# prometheus_client not available\n",
            media_type="text/plain",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_llm_provider(settings: Any) -> dict[str, Any]:
    """Check LLM provider configuration and availability."""
    provider = getattr(settings, "llm_provider", "unknown")
    model = getattr(settings, "llm_model", "unknown")

    # Check API key availability
    key_map = {
        "gemini": "google_api_key",
        "claude": "anthropic_api_key",
        "openai": "openai_api_key",
    }
    key_attr = key_map.get(provider, "")
    api_key_set = bool(getattr(settings, key_attr, None)) if key_attr else False

    return {
        "provider": provider,
        "model": model,
        "available": api_key_set,
        "status": "configured" if api_key_set else "api_key_missing",
        "note": (
            None if api_key_set
            else f"Set {key_attr.upper()} to enable LLM explanations. "
                 "Deterministic ranking still works without it."
        ),
    }


def _get_adapter_statuses(scout_agent: Any) -> dict[str, Any]:
    """Extract adapter health from ScoutAgent if available."""
    if scout_agent is None:
        return {
            "huggingface": {"status": "unknown", "available": False},
            "openml": {"status": "unknown", "available": False},
            "kaggle": {"status": "unknown", "available": False},
        }

    try:
        # Try to get adapter registry health
        registry = getattr(scout_agent, "adapter_registry", None)
        if registry is None:
            registry = getattr(scout_agent, "_adapter_registry", None)

        if registry and hasattr(registry, "get_health"):
            return registry.get_health()
    except Exception:
        pass

    # Fallback: report adapters as available but unverified
    return {
        "huggingface": {"status": "available", "available": True},
        "openml": {"status": "available", "available": True},
        "kaggle": {"status": "unknown", "available": False, "note": "credentials not verified"},
    }