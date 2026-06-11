"""
datascout.api.dependencies
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: FastAPI dependency injection functions.

WHY dependency injection (not global state)?
  - Testing: inject mock agent without patching module globals
  - Flexibility: swap implementation per endpoint if needed
  - Testability: pytest can override dependencies with test_app.dependency_overrides
  - Explicitness: dependencies appear in route signatures → self-documenting

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

from typing import Any

from fastapi import Request


async def get_scout_agent(request: Request) -> Any | None:
    """
    Dependency: return the ScoutAgent from app state.

    Returns None if the agent was not initialized (degraded mode).
    Route handlers must check for None and return 503.

    Complexity: O(1) — attribute lookup.
    """
    return getattr(request.app.state, "scout_agent", None)


async def get_settings(request: Request) -> Any:
    """
    Dependency: return application settings from app state.

    Complexity: O(1) — attribute lookup.
    """
    from datascout.infrastructure.config.settings import get_settings as _get_settings

    return getattr(request.app.state, "settings", None) or _get_settings()
