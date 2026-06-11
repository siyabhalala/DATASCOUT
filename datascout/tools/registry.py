"""
datascout.tools.registry
------------------------
Global registry for tool adapter instances.

Adapters register themselves at import time or via explicit registration.
The tool orchestrator looks up adapters by name from this registry.

Design rationale:
    Decouples adapter selection from orchestrator logic.
    Adding a new adapter = implement BaseTool + register.
    No changes needed in orchestrator or agent controller.

Usage:
    # Register an adapter
    from datascout.tools.registry import register_tool
    from datascout.tools.dataset_sources.kaggle_tool import KaggleTool

    kaggle = KaggleTool(api_key="...")
    register_tool(kaggle)

    # Retrieve an adapter
    tool = get_tool("kaggle")
    results = await tool.search(query)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datascout.contracts.errors import ToolNotFoundError

if TYPE_CHECKING:
    from datascout.tools.base import BaseTool

# Global registry mapping adapter names to instances
_TOOL_REGISTRY: dict[str, "BaseTool"] = {}


def register_tool(tool: "BaseTool") -> None:
    """
    Register a tool adapter in the global registry.

    Args:
        tool: Adapter instance implementing BaseTool

    Raises:
        ValueError: If an adapter with this name is already registered

    Example:
        kaggle = KaggleTool(api_key=os.environ["KAGGLE_API_KEY"])
        register_tool(kaggle)
    """
    if tool.name in _TOOL_REGISTRY:
        raise ValueError(
            f"Tool '{tool.name}' is already registered. "
            f"Use unregister_tool() first if you need to replace it."
        )

    _TOOL_REGISTRY[tool.name] = tool


def unregister_tool(name: str) -> None:
    """
    Remove a tool adapter from the registry.

    Args:
        name: Adapter name to remove

    Raises:
        ToolNotFoundError: If adapter is not registered
    """
    if name not in _TOOL_REGISTRY:
        raise ToolNotFoundError(
            f"Cannot unregister tool '{name}' — not found in registry",
            context={"requested_tool": name, "available_tools": list(_TOOL_REGISTRY.keys())},
        )

    del _TOOL_REGISTRY[name]


def get_tool(name: str) -> "BaseTool":
    """
    Retrieve a registered tool adapter by name.

    Args:
        name: Adapter name (must match SearchQuery.source_adapter)

    Returns:
        Registered adapter instance

    Raises:
        ToolNotFoundError: If adapter is not registered

    Example:
        tool = get_tool("kaggle")
        results = await tool.search(query)
    """
    if name not in _TOOL_REGISTRY:
        raise ToolNotFoundError(
            f"Tool '{name}' not found in registry",
            context={
                "requested_tool": name,
                "available_tools": list(_TOOL_REGISTRY.keys()),
            },
        )

    return _TOOL_REGISTRY[name]


def list_tools() -> list[str]:
    """
    List all registered tool adapter names.

    Returns:
        List of adapter names

    Example:
        >>> list_tools()
        ['kaggle', 'huggingface', 'openml']
    """
    return list(_TOOL_REGISTRY.keys())


def is_tool_registered(name: str) -> bool:
    """
    Check if a tool adapter is registered.

    Args:
        name: Adapter name to check

    Returns:
        True if registered, False otherwise
    """
    return name in _TOOL_REGISTRY


def clear_registry() -> None:
    """
    Clear all registered tools.

    Primarily used in testing to reset state between tests.
    """
    _TOOL_REGISTRY.clear()


async def health_check_all() -> dict[str, bool]:
    """
    Run health checks on all registered adapters.

    Returns:
        Dict mapping adapter names to health status (True = healthy)

    Example:
        statuses = await health_check_all()
        unhealthy = [name for name, healthy in statuses.items() if not healthy]
        if unhealthy:
            logger.warning("Unhealthy adapters", extra={"adapters": unhealthy})
    """
    import asyncio

    tasks = {
        name: tool.health_check()
        for name, tool in _TOOL_REGISTRY.items()
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    health_status = {}
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            health_status[name] = False
        else:
            health_status[name] = result

    return health_status