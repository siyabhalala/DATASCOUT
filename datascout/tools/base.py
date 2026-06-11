"""
datascout.tools.base
-------------------
Base interface for all DATASCOUT tool adapters.

Every tool (Kaggle, HuggingFace, OpenML, etc.) implements this interface.
The tool layer orchestrator depends only on this abstraction — individual
adapters are pluggable and independently replaceable.

Design principles:
1. Async-first — all tool methods are async for parallel execution
2. Type-safe — input/output contracts via Pydantic models
3. Fail-safe — tools return errors as data, not just raise exceptions
4. Observable — automatic logging and metrics via decorators
5. Testable — adapters can be mocked at the interface boundary
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datascout.contracts import RawDataset, SearchQuery


class BaseTool(ABC):
    """
    Abstract base for all tool adapters.

    Subclasses implement search() to query their respective data sources.
    The tool orchestrator calls search() on all registered adapters in parallel.

    Attributes:
        name: Unique adapter identifier (e.g., "kaggle", "huggingface")
        timeout_seconds: Default timeout for this adapter's operations
    """

    def __init__(self, name: str, timeout_seconds: float = 10.0):
        """
        Initialize tool adapter.

        Args:
            name: Adapter name (must match SearchQuery.source_adapter)
            timeout_seconds: Default operation timeout
        """
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def search(self, query: "SearchQuery") -> list["RawDataset"]:
        """
        Execute search against the tool's data source.

        This is the primary method called by the tool orchestrator.
        Implementations should:
        1. Validate the query
        2. Make the external API call(s)
        3. Parse responses into RawDataset instances
        4. Handle errors gracefully (log + raise DataScoutError subclass)

        Args:
            query: Validated search instruction

        Returns:
            List of raw dataset records (empty list if no results)

        Raises:
            ToolTimeoutError: Operation exceeded timeout
            ToolAPIError: External API returned error
            ToolRateLimitError: Rate limit exceeded
            ToolAuthError: Authentication failed
            ToolNetworkError: Network-level failure
            ToolParseError: Response parsing failed
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """
        Verify adapter is operational.

        Optional health check method. Adapters can override to implement
        a lightweight ping to verify API connectivity, credentials, etc.

        Returns:
            True if adapter is healthy, False otherwise

        Example:
            async def health_check(self) -> bool:
                try:
                    response = await self._http_client.get("/health")
                    return response.status_code == 200
                except Exception:
                    return False
        """
        return True  # Default: assume healthy

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, timeout={self.timeout_seconds}s)"


class ToolResult:
    """
    Wrapper for tool execution results.

    Captures both successful results and failures without raising exceptions.
    The tool orchestrator uses this to aggregate partial results.

    Design rationale:
        If 3 adapters are queried and 1 times out, we want to return
        results from the 2 that succeeded + capture the failure context.
        Raising an exception would abort the entire pipeline.
    """

    def __init__(
        self,
        adapter_name: str,
        datasets: list["RawDataset"] | None = None,
        error: Exception | None = None,
        duration_ms: float = 0.0,
    ):
        """
        Initialize tool result.

        Args:
            adapter_name: Name of the adapter that produced this result
            datasets: List of datasets (None if error occurred)
            error: Exception if the tool failed (None if successful)
            duration_ms: Execution time in milliseconds
        """
        self.adapter_name = adapter_name
        self.datasets = datasets or []
        self.error = error
        self.duration_ms = duration_ms

    @property
    def succeeded(self) -> bool:
        """Returns True if the tool execution succeeded."""
        return self.error is None

    @property
    def failed(self) -> bool:
        """Returns True if the tool execution failed."""
        return self.error is not None

    def __repr__(self) -> str:
        if self.succeeded:
            return (
                f"ToolResult(adapter={self.adapter_name!r}, "
                f"datasets={len(self.datasets)}, duration={self.duration_ms:.1f}ms)"
            )
        else:
            return (
                f"ToolResult(adapter={self.adapter_name!r}, "
                f"error={self.error.__class__.__name__}, duration={self.duration_ms:.1f}ms)"
            )