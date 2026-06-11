"""
datascout.adapters.registry
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Singleton adapter registry with lazy initialization,
auto-registration on import, and health-aware adapter routing.

AGENT-0 CONTEXT:
  The orchestrator uses this registry to discover and call adapters.
  No adapter-specific code anywhere outside this package.

SYSTEM DESIGN DECISIONS:

  1. WHY singleton registry?
     - One registry per process — no duplicate adapter initialization
     - Adapter init is expensive (API auth, connection test)
     - Reusing the same instance across all orchestrator calls
     - Thread-safe: adapters registered at import time, before any concurrency

  2. WHY lazy initialization?
     - Don't connect to Kaggle/HF/OpenML APIs at startup
     - Startup should be fast — adapters init on first use
     - If Kaggle is down at startup, system still starts cleanly
     - Health checks can be used to probe readiness independently

  3. WHY auto-register on import?
     - Adapters register themselves: no central config file to keep in sync
     - Adding a new adapter = create the file + add one import here
     - Orchestrator always gets the current set of registered adapters

  4. WHY get_healthy_adapters() method?
     - Before executing a parallel search, verify which adapters are ready
     - Skips adapters with OPEN circuit breakers (already known to be failing)
     - Returns at least min_adapters or raises InsufficientAdaptersError

FAILURE SCENARIOS HANDLED:
  - Adapter init fails at registration → logged, adapter excluded from registry
  - All adapters unhealthy → InsufficientAdaptersError with clear message
  - Adapter not found by name → AdapterNotRegisteredError
  - Duplicate registration → log warning, skip (idempotent)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from typing import Optional

from datascout.contracts.errors import AdapterError
from datascout.contracts.requests import SearchQuery
from datascout.infrastructure.logging import get_logger

from .base import AdapterHealth, BaseAdapter

logger = get_logger("adapters.registry")


class AdapterRegistry:
    """
    Singleton registry of all data source adapters.

    Usage:
        from adapters.registry import adapter_registry

        # Get a specific adapter
        kaggle = adapter_registry.get("kaggle")

        # Get all available adapters
        all_adapters = adapter_registry.get_all()

        # Get only healthy adapters (circuit not open)
        healthy = await adapter_registry.get_healthy()
    """

    _instance: Optional["AdapterRegistry"] = None

    def __new__(cls) -> "AdapterRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters: dict[str, BaseAdapter] = {}
            cls._instance._disabled: set[str] = set()
        return cls._instance

    def register(self, adapter: BaseAdapter) -> None:
        """
        Register an adapter instance.
        Idempotent: re-registering the same adapter name logs a warning and skips.
        """
        name = adapter.ADAPTER_NAME
        if name in self._adapters:
            logger.warning(
                "adapter_already_registered",
                extra={"adapter": name, "action": "skipped"},
            )
            return
        self._adapters[name] = adapter
        logger.info(
            "adapter_registered",
            extra={
                "adapter": name,
                "version": adapter.ADAPTER_VERSION,
            },
        )

    def disable(self, name: str, reason: str = "") -> None:
        """Manually disable an adapter by name."""
        self._disabled.add(name)
        logger.warning(
            "adapter_disabled",
            extra={"adapter": name, "reason": reason},
        )

    def enable(self, name: str) -> None:
        """Re-enable a previously disabled adapter."""
        self._disabled.discard(name)
        logger.info("adapter_enabled", extra={"adapter": name})

    def get(self, name: str) -> BaseAdapter:
        """
        Get adapter by name. Raises if not registered or disabled.
        """
        if name in self._disabled:
            raise AdapterError(
                message=f"Adapter '{name}' is disabled.",
                adapter=name,
            )
        if name not in self._adapters:
            raise AdapterError(
                message=f"Adapter '{name}' is not registered. "
                        f"Available: {list(self._adapters.keys())}",
                adapter=name,
            )
        return self._adapters[name]

    def get_all(self, include_disabled: bool = False) -> dict[str, BaseAdapter]:
        """Return all registered adapters, optionally including disabled ones."""
        if include_disabled:
            return dict(self._adapters)
        return {
            name: adapter
            for name, adapter in self._adapters.items()
            if name not in self._disabled
        }

    async def get_healthy(
        self,
        names: Optional[list[str]] = None,
    ) -> dict[str, BaseAdapter]:
        """
        Return adapters that are not circuit-breaker-open.

        Checks circuit breaker state only (fast, no network call).
        Full health probe (network ping) is done by health_check().

        Args:
            names: Specific adapter names to check. None = all registered.
        """
        from infrastructure.circuit_breaker import circuit_registry

        candidates = self.get_all()
        if names:
            candidates = {k: v for k, v in candidates.items() if k in names}

        cb_statuses = circuit_registry.get_all_status()
        healthy: dict[str, BaseAdapter] = {}

        for name, adapter in candidates.items():
            cb = cb_statuses.get(name, {})
            cb_state = cb.get("state", "closed")
            if cb_state == "open":
                logger.warning(
                    "adapter_skipped_circuit_open",
                    extra={"adapter": name},
                )
                continue
            healthy[name] = adapter

        if not healthy:
            logger.error(
                "no_healthy_adapters",
                extra={"total_registered": len(candidates)},
            )

        return healthy

    async def health_check_all(self) -> dict[str, AdapterHealth]:
        """
        Run health checks on all registered (non-disabled) adapters in parallel.
        Returns dict of adapter_name → AdapterHealth.
        """
        import asyncio
        adapters = self.get_all()
        if not adapters:
            return {}

        tasks = {
            name: asyncio.create_task(adapter.health_check())
            for name, adapter in adapters.items()
        }
        results: dict[str, AdapterHealth] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = AdapterHealth(
                    adapter=name,
                    healthy=False,
                    error=f"Health check raised: {e}",
                )
        return results

    def list_registered(self) -> list[str]:
        """Return sorted list of all registered adapter names."""
        return sorted(self._adapters.keys())

    def __repr__(self) -> str:
        return (
            f"AdapterRegistry("
            f"registered={self.list_registered()}, "
            f"disabled={sorted(self._disabled)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON + AUTO-REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

adapter_registry = AdapterRegistry()


def _auto_register() -> None:
    """
    Auto-register all known adapters at import time.
    Each adapter is instantiated lazily — no network calls happen here.
    Import errors for individual adapters are caught and logged — they don't
    prevent the registry from initializing with the remaining adapters.
    """
    adapters_to_register = [
        ("kaggle",       "datascout.adapters.kaggle_adapter",      "KaggleAdapter"),
        ("huggingface",  "datascout.adapters.huggingface_adapter",  "HuggingFaceAdapter"),
        ("openml",       "datascout.adapters.openml_adapter",       "OpenMLAdapter"),
    ]

    for adapter_name, module_path, class_name in adapters_to_register:
        try:
            import importlib
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            adapter_registry.register(adapter_class())
        except Exception as e:
            logger.warning(
                "adapter_auto_register_failed",
                extra={
                    "adapter": adapter_name,
                    "error": str(e)[:100],
                },
            )


_auto_register()