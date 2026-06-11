"""
datascout/agents/base.py

Abstract base class for all DATASCOUT agents.
Defines the contract every agent must fulfil.

Complexity: O(1) per method call (abstract stubs).
"""

from __future__ import annotations

import abc
import time
from typing import Any

from datascout.contracts.schemas import SearchQuery, SearchResult
from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AgentError(Exception):
    """Raised when an agent encounters an unrecoverable error."""


class BaseAgent(abc.ABC):
    """
    Abstract agent contract.

    Every agent must implement:
        - run(query) → SearchResult
        - name property
        - version property
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version string e.g. '1.0.0'."""

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (optional override)                                  #
    # ------------------------------------------------------------------ #

    def on_start(self, query: SearchQuery) -> None:
        """Called once before the main run loop. Override to add setup logic."""
        logger.info(
            "agent_start",
            extra={
                "agent": self.name,
                "version": self.version,
                "query": query.raw_query,
            },
        )

    def on_finish(self, result: SearchResult, elapsed_ms: float) -> None:
        """Called once after the main run loop finishes."""
        logger.info(
            "agent_finish",
            extra={
                "agent": self.name,
                "version": self.version,
                "datasets_returned": len(result.datasets),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

    def on_error(self, exc: Exception) -> None:
        """Called when an unhandled exception bubbles up from run()."""
        logger.error(
            "agent_error",
            extra={
                "agent": self.name,
                "version": self.version,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
            },
        )

    # ------------------------------------------------------------------ #
    # Core contract                                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def _run_impl(self, query: SearchQuery) -> SearchResult:
        """
        Concrete agent logic.

        Must NEVER raise — catch all exceptions internally and return a
        degraded SearchResult instead.
        """

    def run(self, query: SearchQuery) -> SearchResult:
        """
        Public entry point.  Wraps _run_impl with timing + lifecycle hooks.

        Complexity: O(cost of _run_impl).
        """
        self.on_start(query)
        t0 = time.perf_counter()
        try:
            result = self._run_impl(query)
        except Exception as exc:  # noqa: BLE001
            self.on_error(exc)
            result = SearchResult(
                query=query,
                datasets=[],
                metadata={"agent_error": str(exc)},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self.on_finish(result, elapsed_ms)
        return result

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def describe(self) -> dict[str, Any]:
        """Return agent metadata dict (useful for health endpoints)."""
        return {
            "name": self.name,
            "version": self.version,
            "class": type(self).__name__,
        }