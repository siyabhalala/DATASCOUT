"""
datascout.infrastructure.health
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Kubernetes-compatible liveness and readiness probes
with per-component health tracking and structured reporting.

AGENT-0 CONTEXT:
  Infrastructure layer — polled by Kubernetes every 30s.
  Incorrect health reporting causes pod restarts or dropped traffic.

SYSTEM DESIGN DECISIONS:

  1. WHY liveness separate from readiness?
     - Liveness: "Is this process alive?" → NO = Kubernetes RESTARTS the pod
     - Readiness: "Can this pod serve traffic?" → NO = Kubernetes STOPS sending requests
     - Without separation: slow DB makes Kubernetes restart perfectly healthy pods
     - A pod can be LIVE (process running) but NOT READY (DB unreachable)

  2. WHY per-component health checks (not one boolean)?
     - "Service unhealthy" is useless for ops
     - "kaggle_adapter: DEGRADED — rate limited for 45s" is actionable
     - Dashboard shows exactly which component is failing
     - Partial degradation: 2 of 3 adapters healthy = DEGRADED not DOWN

  3. WHY HealthStatus.DEGRADED as a middle state?
     - HEALTHY: all components nominal
     - DEGRADED: some non-critical components failing — still serving
     - DOWN: critical components failing — cannot serve
     - Without DEGRADED: a single adapter timeout = DOWN = Kubernetes kills the pod
     - With DEGRADED: one adapter timing out = acceptable degradation

  4. WHY async health checks?
     - Health checks call external APIs (Kaggle ping, DB ping)
     - Synchronous → blocks event loop → health check itself causes latency
     - Async + timeout → check completes in parallel, bounded by timeout

  5. WHY cache health results with TTL?
     - Kubernetes polls every 30s
     - Re-checking every 30s is fine; re-checking every request is wasteful
     - TTL=15s: fresh enough for Kubernetes, cheap enough for production

FAILURE SCENARIOS HANDLED:
  - Health check function raises exception → component marked DOWN + error captured
  - Health check times out → component marked DEGRADED + timeout logged
  - All critical components DOWN → overall status DOWN → readiness returns 503
  - Non-critical components DOWN → overall status DEGRADED → readiness returns 200

PERFORMANCE ANALYSIS:
  - Cached result: O(1) ≈ 0.01ms
  - Full health check (parallel): O(max_component_check_time) ≈ 2-5s
  - Kubernetes probe timeout should be set to 10s minimum

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add startup probe (separate from liveness)
  Breaking: v4.0.0 — change health response schema

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("datascout.infrastructure.health")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH STATUS
# ─────────────────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    HEALTHY  = "healthy"    # All checks passing
    DEGRADED = "degraded"   # Some non-critical checks failing — still serving
    DOWN     = "down"       # Critical checks failing — cannot serve


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT HEALTH RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentHealth:
    """Health result for a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_critical: bool = True            # Critical = DOWN causes overall DOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "error": self.error,
            "checked_at": self.checked_at.isoformat(),
            "is_critical": self.is_critical,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OVERALL HEALTH RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthResponse:
    """Aggregated health response for the entire service."""
    status: HealthStatus
    components: list[ComponentHealth]
    version: str = "3.0.0"
    uptime_seconds: Optional[float] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def http_status_code(self) -> int:
        """
        HTTP status for Kubernetes probes.
        HEALTHY/DEGRADED = 200 (still serving)
        DOWN = 503 (stop sending traffic)
        """
        return 503 if self.status == HealthStatus.DOWN else 200

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "http_status": self.http_status_code,
            "components": [c.to_dict() for c in self.components],
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checked_at": self.checked_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# REGISTERED HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegisteredCheck:
    name: str
    check_fn: Callable[[], Any]         # async () -> ComponentHealth
    is_critical: bool
    timeout_s: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class HealthChecker:
    """
    Collects and runs health checks for all registered components.

    Registration:
        checker.register("kaggle_adapter", kaggle_check_fn, critical=False)
        checker.register("embedding_model", embedding_check_fn, critical=True)

    Probes:
        liveness_probe()  → is process alive?
        readiness_probe() → can it serve traffic?
    """

    def __init__(self, cache_ttl_s: float = 15.0) -> None:
        self._checks: list[RegisteredCheck] = []
        self._cache_ttl_s = cache_ttl_s
        self._cached_result: Optional[HealthResponse] = None
        self._cached_at: Optional[float] = None
        self._start_time: float = time.monotonic()

    def register(
        self,
        name: str,
        check_fn: Callable[[], Any],
        critical: bool = True,
        timeout_s: float = 5.0,
    ) -> None:
        """
        Register a health check function.

        check_fn must be async and return ComponentHealth.
        If it raises, it's caught and marked as DOWN.

        Args:
            name: Component name (shown in health dashboard)
            check_fn: Async function returning ComponentHealth
            critical: If True, DOWN status causes overall DOWN
            timeout_s: Max time before marking as DEGRADED
        """
        self._checks.append(
            RegisteredCheck(
                name=name,
                check_fn=check_fn,
                is_critical=critical,
                timeout_s=timeout_s,
            )
        )
        logger.debug("health_check_registered", extra={"component": name, "critical": critical})

    def _is_cache_valid(self) -> bool:
        if self._cached_result is None or self._cached_at is None:
            return False
        return (time.monotonic() - self._cached_at) < self._cache_ttl_s

    async def _run_single_check(self, check: RegisteredCheck) -> ComponentHealth:
        """Run a single check with timeout protection."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(check.check_fn(), timeout=check.timeout_s)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if isinstance(result, ComponentHealth):
                result.latency_ms = elapsed_ms
                return result
            # check_fn returned a non-ComponentHealth value — treat as healthy
            return ComponentHealth(
                name=check.name,
                status=HealthStatus.HEALTHY,
                latency_ms=elapsed_ms,
                is_critical=check.is_critical,
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "health_check_timeout",
                extra={"component": check.name, "timeout_s": check.timeout_s},
            )
            return ComponentHealth(
                name=check.name,
                status=HealthStatus.DEGRADED,
                latency_ms=elapsed_ms,
                error=f"Health check timed out after {check.timeout_s}s",
                is_critical=check.is_critical,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "health_check_failed",
                extra={"component": check.name, "error": str(exc)},
                exc_info=True,
            )
            return ComponentHealth(
                name=check.name,
                status=HealthStatus.DOWN,
                latency_ms=elapsed_ms,
                error=str(exc),
                is_critical=check.is_critical,
            )

    def _aggregate_status(self, components: list[ComponentHealth]) -> HealthStatus:
        """
        Aggregate individual component statuses into overall service status.

        Rules:
        - Any CRITICAL component DOWN → overall DOWN
        - Any component DEGRADED (or non-critical DOWN) → overall DEGRADED
        - All HEALTHY → overall HEALTHY
        """
        has_degraded = False
        for c in components:
            if c.status == HealthStatus.DOWN and c.is_critical:
                return HealthStatus.DOWN
            if c.status in (HealthStatus.DEGRADED, HealthStatus.DOWN):
                has_degraded = True
        return HealthStatus.DEGRADED if has_degraded else HealthStatus.HEALTHY

    async def check_all(self, force: bool = False) -> HealthResponse:
        """
        Run all registered health checks in parallel.

        Uses cached result if within TTL unless force=True.
        All checks run concurrently — total time = max(individual times).
        """
        if not force and self._is_cache_valid():
            return self._cached_result  # type: ignore[return-value]

        # Run all checks in parallel
        if self._checks:
            tasks = [self._run_single_check(check) for check in self._checks]
            components = list(await asyncio.gather(*tasks, return_exceptions=False))
        else:
            components = []

        overall = self._aggregate_status(components)
        uptime = time.monotonic() - self._start_time

        response = HealthResponse(
            status=overall,
            components=components,
            uptime_seconds=round(uptime, 1),
        )

        # Cache the result
        self._cached_result = response
        self._cached_at = time.monotonic()

        logger.info(
            "health_check_complete",
            extra={
                "overall_status": overall.value,
                "component_count": len(components),
                "critical_down": sum(
                    1 for c in components
                    if c.status == HealthStatus.DOWN and c.is_critical
                ),
            },
        )
        return response

    async def liveness_probe(self) -> HealthResponse:
        """
        Kubernetes liveness probe.
        Returns DOWN only if the process itself is broken (not external deps).
        Kubernetes restarts the pod on DOWN.

        This always returns HEALTHY unless the process is truly broken.
        We never report liveness as DOWN due to external API failures.
        """
        uptime = time.monotonic() - self._start_time
        return HealthResponse(
            status=HealthStatus.HEALTHY,
            components=[
                ComponentHealth(
                    name="process",
                    status=HealthStatus.HEALTHY,
                    message="Process is alive",
                    is_critical=True,
                )
            ],
            uptime_seconds=round(uptime, 1),
        )

    async def readiness_probe(self) -> HealthResponse:
        """
        Kubernetes readiness probe.
        Returns DOWN if critical dependencies are unavailable.
        Kubernetes stops sending traffic to this pod on DOWN.

        Runs full health check (or returns cached result).
        """
        return await self.check_all()


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN CHECK FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def make_circuit_breaker_check(adapter_name: str) -> Callable[[], Any]:
    """
    Health check that reports circuit breaker state as component health.
    Non-critical: one open circuit = DEGRADED, not DOWN.
    """
    async def check() -> ComponentHealth:
        from infrastructure.circuit_breaker import circuit_registry
        status_map = circuit_registry.get_all_status()
        cb_status = status_map.get(adapter_name)
        if cb_status is None:
            return ComponentHealth(
                name=f"circuit_breaker.{adapter_name}",
                status=HealthStatus.HEALTHY,
                message="Circuit breaker not initialized (no traffic yet)",
                is_critical=False,
            )
        state = cb_status.get("state", "closed")
        if state == "open":
            return ComponentHealth(
                name=f"circuit_breaker.{adapter_name}",
                status=HealthStatus.DEGRADED,
                message=f"Circuit OPEN. Will retry at: {cb_status.get('will_retry_at')}",
                is_critical=False,
            )
        return ComponentHealth(
            name=f"circuit_breaker.{adapter_name}",
            status=HealthStatus.HEALTHY,
            message=f"Circuit {state}. Failure count: {cb_status.get('failure_count', 0)}",
            is_critical=False,
        )
    return check


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

health_checker = HealthChecker(cache_ttl_s=15.0)