"""
datascout.orchestration
─────────────────────────────────────────────────────
Public surface for the orchestration package.
Entry point for all DATASCOUT search requests.

Usage:
    from orchestration import ToolOrchestrator, production_config

    orchestrator = ToolOrchestrator(production_config())
    response = await orchestrator.execute(search_query)
    # response: AgentResponse — always, never raises
"""

from .tool_orchestrator import (
    ToolOrchestrator,
    OrchestratorResult,
    OrchestrationConfig,
    development_config,
    production_config,
    testing_config,
)

__all__ = [
    "ToolOrchestrator",
    "OrchestratorResult",
    "OrchestrationConfig",
    "development_config",
    "production_config",
    "testing_config",
]