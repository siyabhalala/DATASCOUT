"""
datascout.tools.orchestrator
----------------------------
Parallel tool execution orchestrator.

Coordinates execution of multiple SearchQuery objects across registered
tool adapters. Handles timeouts, retries, partial failures, and result
aggregation.

The SEARCHING stage of the agent calls execute_searches() with a list
of SearchQuery instances. The orchestrator:
1. Looks up each adapter from the registry
2. Dispatches all searches in parallel
3. Enforces per-tool timeouts
4. Retries transient failures with exponential backoff
5. Aggregates results and failures
6. Emits metrics and logs

Design principles:
- Fail-safe: Partial failures don't abort the pipeline
- Observable: Every execution emits metrics and structured logs
- Concurrent: All tools execute in parallel via asyncio.gather()
- Bounded: Timeouts prevent runaway operations
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from datascout.contracts.errors import (
    ToolTimeoutError,
    is_transient,
    should_retry,
    wrap_external_exception,
)
from datascout.infrastructure.logging import get_contextual_logger
from datascout.infrastructure.monitoring import MetricNames, metrics
from datascout.tools.base import ToolResult
from datascout.tools.registry import get_tool

if TYPE_CHECKING:
    from datascout.contracts import RawDataset, SearchQuery


class ToolOrchestrator:
    """
    Orchestrates parallel execution of tool searches.

    Usage:
        orchestrator = ToolOrchestrator(
            query_id="query-123",
            context_id="ctx-456",
            max_retries=2,
        )

        results = await orchestrator.execute_searches([
            SearchQuery(source_adapter="kaggle", query_string="climate data"),
            SearchQuery(source_adapter="huggingface", query_string="climate data"),
        ])

        # results is a list of ToolResult objects
        successful = [r for r in results if r.succeeded]
        failed = [r for r in results if r.failed]
    """

    def __init__(
        self,
        query_id: str,
        context_id: str,
        max_retries: int = 2,
    ):
        """
        Initialize orchestrator.

        Args:
            query_id: AgentQuery.query_id for logging/metrics
            context_id: ExecutionContext.context_id for logging/metrics
            max_retries: Maximum retry attempts for transient failures
        """
        self.query_id = query_id
        self.context_id = context_id
        self.max_retries = max_retries
        self.logger = get_contextual_logger(
            __name__,
            query_id=query_id,
            context_id=context_id,
        )

    async def execute_searches(
        self,
        queries: list["SearchQuery"],
    ) -> list[ToolResult]:
        """
        Execute all search queries in parallel.

        Args:
            queries: List of search instructions

        Returns:
            List of ToolResult objects (one per query)

        Example:
            results = await orchestrator.execute_searches([query1, query2, query3])
            all_datasets = []
            for result in results:
                if result.succeeded:
                    all_datasets.extend(result.datasets)
                else:
                    logger.warning("Tool failed", extra={
                        "adapter": result.adapter_name,
                        "error": str(result.error),
                    })
        """
        if not queries:
            self.logger.info("No search queries to execute")
            return []

        self.logger.info(
            "Starting parallel tool execution",
            extra={
                "query_count": len(queries),
                "adapters": [q.source_adapter for q in queries],
            },
        )

        # Create task for each query
        tasks = [
            self._execute_single_search(query)
            for query in queries
        ]

        # Execute all in parallel
        start = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_duration_ms = (time.perf_counter() - start) * 1000.0

        # Aggregate metrics
        successful_count = sum(1 for r in results if r.succeeded)
        failed_count = sum(1 for r in results if r.failed)
        total_datasets = sum(len(r.datasets) for r in results)

        self.logger.info(
            "Parallel tool execution complete",
            extra={
                "total_duration_ms": total_duration_ms,
                "successful_adapters": successful_count,
                "failed_adapters": failed_count,
                "total_datasets": total_datasets,
            },
        )

        metrics.histogram(
            MetricNames.TOOL_SEARCH_DURATION,
            total_duration_ms,
            labels={"query_id": self.query_id},
        )

        return results

    async def _execute_single_search(
        self,
        query: "SearchQuery",
    ) -> ToolResult:
        """
        Execute a single search with timeout and retry logic.

        Args:
            query: Search instruction

        Returns:
            ToolResult (success or failure)
        """
        adapter_name = query.source_adapter

        # Attempt with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._execute_with_timeout(query, attempt)
            except Exception as exc:
                # Check if we should retry
                if should_retry(exc, attempt=attempt, max_attempts=self.max_retries):
                    backoff_seconds = self._calculate_backoff(attempt)
                    self.logger.warning(
                        "Tool execution failed, retrying",
                        extra={
                            "adapter": adapter_name,
                            "attempt": attempt,
                            "error": str(exc),
                            "backoff_seconds": backoff_seconds,
                        },
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue
                else:
                    # Final failure — return error result
                    self.logger.error(
                        "Tool execution failed permanently",
                        extra={
                            "adapter": adapter_name,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                        exc_info=True,
                    )
                    metrics.increment(
                        MetricNames.TOOL_ERRORS,
                        labels={"adapter": adapter_name},
                    )
                    return ToolResult(
                        adapter_name=adapter_name,
                        error=exc,
                        duration_ms=0.0,
                    )

        # Should never reach here
        raise RuntimeError(f"Retry loop exited without return for adapter {adapter_name}")

    async def _execute_with_timeout(
        self,
        query: "SearchQuery",
        attempt: int,
    ) -> ToolResult:
        """
        Execute search with hard timeout enforcement.

        Args:
            query: Search instruction
            attempt: Current attempt number (for logging)

        Returns:
            ToolResult on success

        Raises:
            ToolTimeoutError: If timeout exceeded
            Other DataScoutError subclasses: On adapter failures
        """
        adapter_name = query.source_adapter

        try:
            # Look up adapter from registry
            tool = get_tool(adapter_name)
        except Exception as exc:
            # ToolNotFoundError or registry error
            raise wrap_external_exception(exc, layer="tool") from exc

        self.logger.debug(
            "Executing tool search",
            extra={
                "adapter": adapter_name,
                "query": query.query_string,
                "timeout": query.timeout_seconds,
                "attempt": attempt,
            },
        )

        start = time.perf_counter()

        try:
            # Execute with timeout
            datasets = await asyncio.wait_for(
                tool.search(query),
                timeout=query.timeout_seconds,
            )

            duration_ms = (time.perf_counter() - start) * 1000.0

            self.logger.info(
                "Tool search succeeded",
                extra={
                    "adapter": adapter_name,
                    "result_count": len(datasets),
                    "duration_ms": duration_ms,
                },
            )

            metrics.histogram(
                MetricNames.TOOL_SEARCH_DURATION,
                duration_ms,
                labels={"adapter": adapter_name},
            )
            metrics.increment(
                MetricNames.TOOL_SEARCH_RESULTS,
                len(datasets),
                labels={"adapter": adapter_name},
            )

            return ToolResult(
                adapter_name=adapter_name,
                datasets=datasets,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0

            self.logger.warning(
                "Tool search timeout",
                extra={
                    "adapter": adapter_name,
                    "timeout_seconds": query.timeout_seconds,
                    "duration_ms": duration_ms,
                },
            )

            metrics.increment(
                MetricNames.TOOL_TIMEOUTS,
                labels={"adapter": adapter_name},
            )

            raise ToolTimeoutError(
                f"Tool '{adapter_name}' exceeded timeout of {query.timeout_seconds}s",
                context={
                    "adapter": adapter_name,
                    "timeout_seconds": query.timeout_seconds,
                    "duration_ms": duration_ms,
                },
            ) from exc

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Backoff delay in seconds

        Formula: min(2^(attempt-1), 60)
        - Attempt 1: 1s
        - Attempt 2: 2s
        - Attempt 3: 4s
        - Capped at 60s
        """
        base_delay = 2 ** (attempt - 1)
        return min(base_delay, 60.0)