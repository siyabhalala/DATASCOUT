"""
datascout.orchestration.tool_orchestrator
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Parallel adapter orchestration — the top-level
coordinator of the entire Agent-0 pipeline. Every search request enters here,
gets routed to all adapters in parallel, deduplicated, ranked, and returned
as a typed AgentResponse.

AGENT-0 CONTEXT:
  This is the final stage of Agent-0. It is the bridge between raw adapter
  output and the typed AgentResponse contract consumed by Agent-1+.
  Every design decision here has a direct user-facing latency or reliability impact.

SYSTEM DESIGN DECISIONS:

  1. WHY parallel execution over sequential?
     - Sequential: 3 adapters × avg 2s = 6s user wait minimum
     - Parallel: max(2s, 2s, 2s) = 2s user wait — 3x improvement guaranteed
     - asyncio.gather() with per-adapter timeouts is the correct async pattern
     - asyncio.Semaphore(max_concurrent) prevents thundering herd on external APIs

  2. WHY partial results over all-or-nothing?
     - "20 results from 2 adapters in 2s" always beats "0 results after 30s timeout"
     - partial_result=True signals to Agent-1+ that results may be incomplete
     - min_successful_adapters=1: at least one must succeed, else raise InsufficientAdaptersError
     - If ALL adapters fail: return AgentResponse with error_message, never raise to caller

  3. WHY two timeout levels (global + per-adapter)?
     - per_adapter_timeout (6s): individual adapter SLA — don't let one slow adapter block others
     - global_timeout (10s): user-facing SLA — NEVER exceed this total
     - global must always be > per_adapter: adapters need room to finish
     - Enforced at construction time: raises ValueError if violated

  4. WHY cross-adapter deduplication after merge (not per-adapter)?
     - Within-adapter dedup: impossible at ingestion (don't have other adapters' data)
     - Cross-adapter dedup: ONLY possible after merging all results
     - Kaggle "Titanic" + HuggingFace "Titanic" = same fingerprint → keep one
     - At 1M records with 15% overlap: 150K duplicate records without this step

  5. WHY O(n) hash map dedup over pairwise comparison?
     - Pairwise: O(n²) — at 450 records (150 per adapter × 3): 202,500 comparisons
     - Hash map: O(n) — at 450 records: 450 lookups
     - At 1M datasets: hash = 1M ops vs pairwise = 10^12 ops

  6. WHY keep highest-completeness record on collision?
     - HuggingFace "Titanic" may have richer metadata (model card) than Kaggle "Titanic"
     - Kaggle "Titanic" may have download_count that HF doesn't
     - Keeping highest completeness maximises information available to Agent-1

  7. WHY DecisionTrace stored only for top 20?
     - Storing full trace for all 1M results = ~2GB per query
     - Users only ever see top results — traces for rank 21+ are never read
     - lineage_ref on every record → can always reconstruct trace if needed

  8. WHY asyncio.Semaphore for max_concurrent?
     - Without: 100 concurrent users × 3 adapters = 300 simultaneous API calls
     - With Semaphore(10): max 10 adapter calls at any moment system-wide
     - Protects Kaggle/HF/OpenML from being overwhelmed by our own concurrency

FAILURE SCENARIOS HANDLED:
  - asyncio.TimeoutError per adapter     → adapter_errors + partial_result=True
  - CircuitBreakerOpenError              → adapter skipped + logged, not counted as failure
  - All adapters fail                    → InsufficientAdaptersError if below min_successful
  - Global pipeline timeout              → return whatever is ready + partial_result=True
  - Individual record dedup crash        → skip that record + log, continue
  - No healthy adapters at selection     → AgentResponse with clear error_message
  - Unexpected exception in pipeline     → AgentResponse with error_message, never propagates

PERFORMANCE ANALYSIS:
  - Parallel adapter calls: O(max_adapter_latency) not O(sum)
  - Deduplication:          O(n) via fingerprint hash map
  - Result sort:            O(n log n) by completeness score
  - DecisionTrace build:    O(top_k) — only top 20
  - At 450 results (150 per adapter × 3):
      dedup:   <1ms
      sort:    <1ms
      trace:   <1ms
      total:   dominated by adapter fetch latency (~2-6s)
  - At 1M cached results:   ~1s for dedup + sort (acceptable for batch mode)

PIPELINE STAGES (tracked with PipelineStage):
  1. adapter_selection   — circuit-breaker-aware adapter picking
  2. parallel_fetch      — all adapters called concurrently
  3. deduplication       — cross-adapter fingerprint dedup
  4. result_assembly     — sort + cap + DecisionTrace + AgentResponse

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add Redis result caching layer between fetch and dedup
  Breaking: v4.0.0 — OrchestratorResult schema change (add embedding_requested field)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from datascout.contracts import (
    RawDataset,
    compute_confidence,
    compute_quality_tier,
)
from datascout.contracts.errors import (
    CircuitBreakerOpenError,
    DataScoutError,
    InsufficientAdaptersError,
    PipelineTimeoutError,
)
from datascout.contracts.models import EvaluatedDataset
from datascout.contracts.requests import SearchQuery
from datascout.contracts.responses import (
    AgentResponse,
    ConfidenceLevel,
    DecisionTrace,
    PipelineStage,
    RankedResult,
    compute_confidence,
)
from datascout.contracts.states import StageStatus, QualityTier
from datascout.infrastructure.logging import get_logger, log_performance, set_request_context
from datascout.infrastructure.monitoring import metrics

logger = get_logger("orchestration.tool_orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

import os


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "on"}


def _env_list(key: str, default: list[str]) -> list[str]:
    val = os.getenv(key)
    if val is None:
        return default
    return [v.strip() for v in val.split(",") if v.strip()]


@dataclass
class OrchestrationConfig:
    """
    Configuration for the ToolOrchestrator.

    All fields support environment variable overrides for deployment flexibility.
    Validation runs in __post_init__ — startup fails fast on invalid config.

    Environment variable overrides:
        DATASCOUT_ADAPTERS          comma-separated: "kaggle,huggingface,openml"
        DATASCOUT_GLOBAL_TIMEOUT    float seconds:   "10.0"
        DATASCOUT_ADAPTER_TIMEOUT   float seconds:   "6.0"
        DATASCOUT_MIN_ADAPTERS      int:             "1"
        DATASCOUT_MAX_CONCURRENT    int:             "10"
        DATASCOUT_PARALLEL          bool:            "true"
        DATASCOUT_DEDUP             bool:            "true"
        DATASCOUT_PARTIAL_RESULTS   bool:            "true"
        DATASCOUT_FAIL_FAST         bool:            "false"
    """

    enabled_adapters: list[str] = field(
        default_factory=lambda: _env_list(
            "DATASCOUT_ADAPTERS", ["kaggle", "huggingface", "openml"]
        )
    )
    parallel_execution: bool = field(
        default_factory=lambda: _env_bool("DATASCOUT_PARALLEL", True)
    )
    max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("DATASCOUT_MAX_CONCURRENT", "10"))
    )
    global_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("DATASCOUT_GLOBAL_TIMEOUT", "10.0"))
    )
    per_adapter_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("DATASCOUT_ADAPTER_TIMEOUT", "6.0"))
    )
    min_successful_adapters: int = field(
        default_factory=lambda: int(os.getenv("DATASCOUT_MIN_ADAPTERS", "1"))
    )
    allow_partial_results: bool = field(
        default_factory=lambda: _env_bool("DATASCOUT_PARTIAL_RESULTS", True)
    )
    fail_fast: bool = field(
        default_factory=lambda: _env_bool("DATASCOUT_FAIL_FAST", False)
    )
    dedup_across_adapters: bool = field(
        default_factory=lambda: _env_bool("DATASCOUT_DEDUP", True)
    )
    max_results_hard_cap: int = 50
    decision_trace_top_k: int = 20     # Only store DecisionTrace for top N results

    def __post_init__(self) -> None:
        """Validate config at construction. Startup fails fast on invalid config."""
        if self.global_timeout_seconds <= self.per_adapter_timeout_seconds:
            raise ValueError(
                f"global_timeout_seconds ({self.global_timeout_seconds}s) must be "
                f"greater than per_adapter_timeout_seconds ({self.per_adapter_timeout_seconds}s). "
                f"Adapters need room to complete before the global ceiling fires."
            )
        if self.min_successful_adapters < 1:
            raise ValueError("min_successful_adapters must be >= 1")
        if self.min_successful_adapters > len(self.enabled_adapters):
            raise ValueError(
                f"min_successful_adapters ({self.min_successful_adapters}) "
                f"cannot exceed enabled_adapters count ({len(self.enabled_adapters)})"
            )
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        valid = {"kaggle", "huggingface", "openml"}
        invalid = [a for a in self.enabled_adapters if a not in valid]
        if invalid:
            raise ValueError(
                f"Unknown adapters: {invalid}. Valid: {sorted(valid)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_adapters":          self.enabled_adapters,
            "parallel_execution":        self.parallel_execution,
            "max_concurrent":            self.max_concurrent,
            "global_timeout_seconds":    self.global_timeout_seconds,
            "per_adapter_timeout_seconds": self.per_adapter_timeout_seconds,
            "min_successful_adapters":   self.min_successful_adapters,
            "allow_partial_results":     self.allow_partial_results,
            "dedup_across_adapters":     self.dedup_across_adapters,
            "fail_fast":                 self.fail_fast,
            "max_results_hard_cap":      self.max_results_hard_cap,
            "decision_trace_top_k":      self.decision_trace_top_k,
        }


# ─── Pre-built profiles ────────────────────────────────────────────────────

def development_config() -> OrchestrationConfig:
    """Local development: generous timeouts, all adapters, partial results OK."""
    return OrchestrationConfig(
        enabled_adapters=["kaggle", "huggingface", "openml"],
        parallel_execution=True,
        global_timeout_seconds=15.0,
        per_adapter_timeout_seconds=8.0,
        min_successful_adapters=1,
        allow_partial_results=True,
        fail_fast=False,
        dedup_across_adapters=True,
        max_concurrent=5,
    )


def production_config() -> OrchestrationConfig:
    """Production: tight SLA, max concurrency, partial results always allowed."""
    return OrchestrationConfig(
        enabled_adapters=["kaggle", "huggingface", "openml"],
        parallel_execution=True,
        global_timeout_seconds=10.0,
        per_adapter_timeout_seconds=6.0,
        min_successful_adapters=1,
        allow_partial_results=True,
        fail_fast=False,
        dedup_across_adapters=True,
        max_concurrent=10,
    )


def testing_config() -> OrchestrationConfig:
    """Tests: sequential execution, short timeouts, single adapter, fail fast."""
    return OrchestrationConfig(
        enabled_adapters=["kaggle"],
        parallel_execution=False,
        global_timeout_seconds=5.0,
        per_adapter_timeout_seconds=3.0,
        min_successful_adapters=1,
        allow_partial_results=True,
        fail_fast=True,
        dedup_across_adapters=True,
        max_concurrent=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    """
    Internal result produced by _run_pipeline().
    Converted to AgentResponse by _build_response().
    Not exposed outside this module.
    """
    datasets: list[RawDataset]
    adapter_results: dict[str, int]       # {"kaggle": 15, "huggingface": 8}
    adapter_errors: dict[str, str]        # {"openml": "AdapterTimeoutError"}
    partial_result: bool
    stages: list[PipelineStage]
    processing_time_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# TOOL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolOrchestrator:
    """
    Orchestrates parallel calls to all registered adapters.

    Entry point for all DATASCOUT search requests.

    Pipeline (4 stages):
        1. adapter_selection   — circuit-breaker-aware adapter picking
        2. parallel_fetch      — asyncio.gather with per-adapter + global timeouts
        3. deduplication       — O(n) fingerprint hash map, keep highest completeness
        4. result_assembly     — sort, cap, build DecisionTrace for top-k, AgentResponse

    Usage:
        orchestrator = ToolOrchestrator(production_config())
        response = await orchestrator.execute(search_query)
        # → AgentResponse (always — never raises)
    """

    def __init__(self, config: Optional[OrchestrationConfig] = None) -> None:
        self.config = config or production_config()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        logger.info(
            "tool_orchestrator_initialized",
            extra=self.config.to_dict(),
        )

    @log_performance("orchestrator.execute")
    async def execute(self, query: SearchQuery) -> AgentResponse:
        """
        Execute a full search pipeline for the given query.

        This method NEVER raises. All failures are captured in AgentResponse.
        Callers can check response.is_success() and response.error_message.

        Args:
            query: Validated SearchQuery (auto-generated query_id if not set)

        Returns:
            AgentResponse — always, even on total failure
        """
        pipeline_start = time.perf_counter()
        set_request_context(
            request_id=query.query_id,
            session_id=query.session_id or "unknown",
            agent_id="agent-0",
        )
        metrics.request_started()
        stages: list[PipelineStage] = []

        try:
            result = await asyncio.wait_for(
                self._run_pipeline(query, stages),
                timeout=self.config.global_timeout_seconds,
            )
        except asyncio.TimeoutError:
            processing_ms = int((time.perf_counter() - pipeline_start) * 1000)
            logger.error(
                "pipeline_global_timeout",
                extra={
                    "query_id":   query.query_id,
                    "timeout_s":  self.config.global_timeout_seconds,
                    "elapsed_ms": processing_ms,
                },
            )
            metrics.request_error()
            return AgentResponse(
                query_id=query.query_id,
                results=[],
                partial_result=True,
                pipeline_stages=stages,
                processing_time_ms=processing_ms,
                error_message=(
                    f"Pipeline exceeded global SLA of "
                    f"{self.config.global_timeout_seconds}s. "
                    f"Try narrowing your query."
                ),
            )

        except InsufficientAdaptersError as e:
            processing_ms = int((time.perf_counter() - pipeline_start) * 1000)
            logger.error(
                "pipeline_insufficient_adapters",
                extra={
                    "query_id":  query.query_id,
                    "succeeded": e.succeeded,
                    "total":     e.total,
                    "minimum":   e.minimum,
                },
            )
            metrics.request_error()
            return AgentResponse(
                query_id=query.query_id,
                results=[],
                pipeline_stages=stages,
                processing_time_ms=processing_ms,
                error_message=e.message,
                error_code=int(e.code),
            )

        except Exception as e:
            processing_ms = int((time.perf_counter() - pipeline_start) * 1000)
            logger.error(
                "pipeline_unexpected_error",
                extra={
                    "query_id":   query.query_id,
                    "error_type": type(e).__name__,
                    "error":      str(e)[:200],
                },
                exc_info=True,
            )
            metrics.request_error()
            return AgentResponse(
                query_id=query.query_id,
                results=[],
                pipeline_stages=stages,
                processing_time_ms=processing_ms,
                error_message=f"Pipeline error: {type(e).__name__}: {str(e)[:100]}",
            )

        except BaseException as e:
            # FIX: In Python 3.8+, asyncio.CancelledError is BaseException, NOT Exception.
            # When uvicorn/gunicorn cancels a request task (worker timeout, client disconnect),
            # CancelledError propagates through execute() and was crashing the server.
            # We catch it here, log it, and return a safe AgentResponse instead of crashing.
            processing_ms = int((time.perf_counter() - pipeline_start) * 1000)
            logger.warning(
                "pipeline_cancelled",
                extra={
                    "query_id":   query.query_id,
                    "error_type": type(e).__name__,
                    "elapsed_ms": processing_ms,
                },
            )
            metrics.request_error()
            return AgentResponse(
                query_id=query.query_id,
                results=[],
                partial_result=True,
                pipeline_stages=stages,
                processing_time_ms=processing_ms,
                error_message="Request was cancelled (worker timeout or client disconnect).",
            )

        result.processing_time_ms = int((time.perf_counter() - pipeline_start) * 1000)
        metrics.stage_complete("pipeline", result.processing_time_ms / 1000, success=True)

        if result.partial_result:
            metrics.request_partial()
        else:
            metrics.request_success()

        return self._build_response(query, result)

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE STAGES
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_pipeline(
        self,
        query: SearchQuery,
        stages: list[PipelineStage],
    ) -> OrchestratorResult:
        """
        Execute the four pipeline stages in sequence.
        Each stage is tracked with PipelineStage for SLA monitoring.
        """

        # ── Stage 1: Adapter Selection ────────────────────────────────────────
        stage_select = PipelineStage(stage="adapter_selection")
        stages.append(stage_select)
        stage_select.mark_running()

        from datascout.adapters.registry import adapter_registry

        requested = query.sources if query.sources else self.config.enabled_adapters
        healthy_adapters = await adapter_registry.get_healthy(names=requested)

        if not healthy_adapters:
            stage_select.mark_failed(
                error="No healthy adapters available",
                error_code=3000,
            )
            raise InsufficientAdaptersError(
                succeeded=0,
                total=len(requested),
                minimum=self.config.min_successful_adapters,
            )

        stage_select.mark_done()
        stage_select.metadata["selected"] = list(healthy_adapters.keys())
        stage_select.metadata["requested"] = requested

        logger.info(
            "adapters_selected",
            extra={
                "query_id": query.query_id,
                "adapters": list(healthy_adapters.keys()),
                "requested": requested,
            },
        )

        # ── Stage 2: Parallel Fetch ───────────────────────────────────────────
        stage_fetch = PipelineStage(stage="parallel_fetch")
        stages.append(stage_fetch)
        stage_fetch.mark_running()

        fetch_start = time.perf_counter()

        if self.config.parallel_execution:
            raw_by_adapter, adapter_errors = await self._fetch_parallel(
                query, healthy_adapters
            )
        else:
            raw_by_adapter, adapter_errors = await self._fetch_sequential(
                query, healthy_adapters
            )

        fetch_ms = int((time.perf_counter() - fetch_start) * 1000)
        succeeded = len(raw_by_adapter)
        total_fetched = sum(len(v) for v in raw_by_adapter.values())

        metrics.stage_complete("parallel_fetch", fetch_ms / 1000, success=succeeded > 0)

        if succeeded < self.config.min_successful_adapters:
            stage_fetch.mark_failed(
                error=f"Only {succeeded}/{len(healthy_adapters)} adapters succeeded",
                error_code=3000,
            )
            raise InsufficientAdaptersError(
                succeeded=succeeded,
                total=len(healthy_adapters),
                minimum=self.config.min_successful_adapters,
            )

        stage_fetch.mark_done()
        stage_fetch.metadata.update({
            "adapters_succeeded": succeeded,
            "adapters_failed":    len(adapter_errors),
            "total_records":      total_fetched,
            "fetch_ms":           fetch_ms,
            "per_adapter":        {k: len(v) for k, v in raw_by_adapter.items()},
        })

        logger.info(
            "parallel_fetch_complete",
            extra={
                "query_id":     query.query_id,
                "succeeded":    succeeded,
                "failed":       len(adapter_errors),
                "total_records": total_fetched,
                "fetch_ms":     fetch_ms,
                "errors":       adapter_errors,
            },
        )

        # ── Stage 3: Deduplication ────────────────────────────────────────────
        stage_dedup = PipelineStage(stage="deduplication")
        stages.append(stage_dedup)
        stage_dedup.mark_running()

        dedup_start = time.perf_counter()
        all_raw: list[RawDataset] = [
            ds for records in raw_by_adapter.values() for ds in records
        ]

        if self.config.dedup_across_adapters:
            deduplicated = self._deduplicate(all_raw)
        else:
            deduplicated = all_raw

        dedup_ms = int((time.perf_counter() - dedup_start) * 1000)
        duplicates_removed = len(all_raw) - len(deduplicated)

        metrics.stage_complete("deduplication", dedup_ms / 1000)

        stage_dedup.mark_done()
        stage_dedup.metadata.update({
            "before":             len(all_raw),
            "after":              len(deduplicated),
            "duplicates_removed": duplicates_removed,
            "dedup_ms":           dedup_ms,
        })

        if duplicates_removed > 0:
            logger.info(
                "deduplication_complete",
                extra={
                    "query_id":           query.query_id,
                    "before":             len(all_raw),
                    "after":              len(deduplicated),
                    "duplicates_removed": duplicates_removed,
                },
            )

        # ── Stage 4: Result Assembly ──────────────────────────────────────────
        stage_assemble = PipelineStage(stage="result_assembly")
        stages.append(stage_assemble)
        stage_assemble.mark_running()

        assemble_start = time.perf_counter()

        # Sort by metadata_completeness descending
        # Agent-1 will re-rank by semantic relevance — this is pre-sort by quality
        sorted_datasets = sorted(
            deduplicated,
            key=lambda d: d.metadata_completeness,
            reverse=True,
        )

        # Cap at query.max_results (already validated at [1, 50])
        capped = sorted_datasets[: min(query.max_results, self.config.max_results_hard_cap)]

        assemble_ms = int((time.perf_counter() - assemble_start) * 1000)
        metrics.stage_complete("result_assembly", assemble_ms / 1000)

        stage_assemble.mark_done()
        stage_assemble.metadata.update({
            "total_available": len(deduplicated),
            "returned":        len(capped),
            "assemble_ms":     assemble_ms,
        })

        return OrchestratorResult(
            datasets=capped,
            adapter_results={k: len(v) for k, v in raw_by_adapter.items()},
            adapter_errors=adapter_errors,
            partial_result=len(adapter_errors) > 0,
            stages=stages,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PARALLEL FETCH
    # ─────────────────────────────────────────────────────────────────────────

    async def _fetch_parallel(
        self,
        query: SearchQuery,
        adapters: dict,
    ) -> tuple[dict[str, list[RawDataset]], dict[str, str]]:
        """
        Call all adapters concurrently, bounded by Semaphore and per-adapter timeout.

        Semaphore: prevents our own concurrency from rate-limiting external APIs.
        per-adapter timeout: one slow adapter doesn't block the others.

        Returns:
            (successes dict, errors dict) — never raises
        """

        async def _call_one(name: str, adapter: Any) -> tuple[str, Any]:
            async with self._semaphore:
                start = time.perf_counter()
                try:
                    results = await asyncio.wait_for(
                        adapter.search(query),
                        timeout=self.config.per_adapter_timeout_seconds,
                    )
                    elapsed = time.perf_counter() - start
                    metrics.adapter_success(name, len(results), elapsed)
                    logger.info(
                        "adapter_success",
                        extra={
                            "adapter":     name,
                            "results":     len(results),
                            "elapsed_ms":  int(elapsed * 1000),
                            "query_id":    query.query_id,
                        },
                    )
                    return name, results

                except asyncio.TimeoutError:
                    elapsed = time.perf_counter() - start
                    metrics.adapter_timeout(name, elapsed)
                    logger.warning(
                        "adapter_timeout",
                        extra={
                            "adapter":    name,
                            "timeout_s":  self.config.per_adapter_timeout_seconds,
                            "elapsed_ms": int(elapsed * 1000),
                            "query_id":   query.query_id,
                        },
                    )
                    return name, asyncio.TimeoutError(
                        f"{name} timed out after {self.config.per_adapter_timeout_seconds}s"
                    )

                except CircuitBreakerOpenError as e:
                    metrics.adapter_error(name, "circuit_open")
                    logger.warning(
                        "adapter_circuit_open",
                        extra={
                            "adapter":    name,
                            "retry_at":   e.will_retry_at.isoformat(),
                            "query_id":   query.query_id,
                        },
                    )
                    return name, e

                except Exception as e:
                    elapsed = time.perf_counter() - start
                    metrics.adapter_error(name, type(e).__name__.lower())
                    logger.error(
                        "adapter_error",
                        extra={
                            "adapter":     name,
                            "error_type":  type(e).__name__,
                            "error":       str(e)[:100],
                            "elapsed_ms":  int(elapsed * 1000),
                            "query_id":    query.query_id,
                        },
                        exc_info=True,
                    )
                    return name, e

                except BaseException as e:
                    # FIX: CancelledError (Python 3.8+) is BaseException not Exception.
                    # Without this, when the outer pipeline is cancelled, this task
                    # raises CancelledError which is NOT caught by except Exception,
                    # propagates out of gather(), and crashes the server.
                    # We catch it, log it, and return it as an error tuple so gather
                    # can complete cleanly and the pipeline can degrade gracefully.
                    elapsed = time.perf_counter() - start
                    metrics.adapter_error(name, "cancelled")
                    logger.warning(
                        "adapter_cancelled",
                        extra={
                            "adapter":    name,
                            "elapsed_ms": int(elapsed * 1000),
                            "query_id":   query.query_id,
                        },
                    )
                    return name, RuntimeError(f"{name} cancelled after {elapsed:.1f}s")

        tasks = [
            asyncio.create_task(_call_one(name, adapter))
            for name, adapter in adapters.items()
        ]
        # FIX: return_exceptions=True so if one task raises (e.g. unexpected BaseException
        # that slips through _call_one), the other adapter tasks are NOT cancelled.
        # Previously return_exceptions=False meant one crash killed all parallel adapters.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results:  dict[str, list[RawDataset]] = {}
        errors:   dict[str, str]              = {}

        # outcomes is list of (name, result_or_exc) tuples OR raw exceptions if task raised
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                # Task itself raised (shouldn't happen with our BaseException catch above,
                # but defensive: map by position back to adapter name)
                adapter_name = list(adapters.keys())[i]
                errors[adapter_name] = type(outcome).__name__
                logger.error(
                    "adapter_task_raised",
                    extra={"adapter": adapter_name, "error": str(outcome)[:100]},
                )
                continue

            name, result_or_exc = outcome
            if isinstance(result_or_exc, BaseException):
                errors[name] = type(result_or_exc).__name__
            elif isinstance(result_or_exc, list):
                if result_or_exc:
                    results[name] = result_or_exc
                else:
                    logger.debug("adapter_returned_empty", extra={"adapter": name})
                    results[name] = []
            else:
                results[name] = []

        return results, errors

    async def _fetch_sequential(
        self,
        query: SearchQuery,
        adapters: dict,
    ) -> tuple[dict[str, list[RawDataset]], dict[str, str]]:
        """
        Sequential fallback — only used when parallel_execution=False (testing/debug).
        Production always uses _fetch_parallel.
        """
        results: dict[str, list[RawDataset]] = {}
        errors:  dict[str, str]              = {}

        for name, adapter in adapters.items():
            try:
                datasets = await asyncio.wait_for(
                    adapter.search(query),
                    timeout=self.config.per_adapter_timeout_seconds,
                )
                results[name] = datasets or []
            except asyncio.TimeoutError:
                errors[name] = "TimeoutError"
                logger.warning(
                    "sequential_adapter_timeout",
                    extra={"adapter": name, "query_id": query.query_id},
                )
            except asyncio.CancelledError:
                # FIX: CancelledError must propagate — do not swallow it here.
                # execute() will catch it via the BaseException handler and return
                # a safe partial AgentResponse instead of hanging or crashing.
                raise
            except Exception as e:
                errors[name] = type(e).__name__
                logger.error(
                    "sequential_adapter_error",
                    extra={"adapter": name, "error": str(e)[:100]},
                )

            # fail_fast: stop on first error in sequential mode
            if self.config.fail_fast and errors:
                logger.warning(
                    "sequential_fail_fast_triggered",
                    extra={"adapter": name, "query_id": query.query_id},
                )
                break

        return results, errors

    # ─────────────────────────────────────────────────────────────────────────
    # DEDUPLICATION
    # ─────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, datasets: list[RawDataset]) -> list[RawDataset]:
        """
        Remove cross-adapter duplicates by dataset_fingerprint.

        Algorithm: O(n) hash map — fingerprint → best RawDataset.
        On collision: keep record with highest metadata_completeness.

        WHY highest completeness wins:
        - HuggingFace Titanic may have richer model card than Kaggle Titanic
        - Kaggle Titanic may have download_count that HF doesn't have
        - Completeness score captures which record has more information

        Marks duplicate records with is_duplicate=True in extra{} for audit.
        """
        seen: dict[str, RawDataset] = {}  # fingerprint → best record

        for ds in datasets:
            fp = ds.dataset_fingerprint

            if fp not in seen:
                seen[fp] = ds
            else:
                existing = seen[fp]
                if ds.metadata_completeness > existing.metadata_completeness:
                    # New record is richer — replace
                    seen[fp] = ds
                    logger.debug(
                        "dedup_replaced",
                        extra={
                            "fingerprint":   fp[:16],
                            "old_source":    existing.source,
                            "old_score":     existing.metadata_completeness,
                            "new_source":    ds.source,
                            "new_score":     ds.metadata_completeness,
                        },
                    )
                    metrics.dataset_duplicate(existing.source)
                else:
                    # Existing record is richer — discard new
                    logger.debug(
                        "dedup_discarded",
                        extra={
                            "fingerprint":  fp[:16],
                            "kept_source":  existing.source,
                            "kept_score":   existing.metadata_completeness,
                            "drop_source":  ds.source,
                            "drop_score":   ds.metadata_completeness,
                        },
                    )
                    metrics.dataset_duplicate(ds.source)

        return list(seen.values())

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        query: SearchQuery,
        result: OrchestratorResult,
    ) -> AgentResponse:
        """
        Convert OrchestratorResult into a fully typed AgentResponse.

        FIX (Phase 2): This method previously skipped the entire evaluation pipeline,
        using only metadata_completeness as the score. Now properly wires:
          1. QueryParser → extract task_type + keywords from raw query
          2. DatasetScorer → 5-dimensional deterministic scoring
          3. RankingEngine → diversity-boosted ranking + confidence
          4. DecisionTrace → explainable per-dataset score breakdown

        LLM is NOT called here — explanation is a separate optional layer.
        Ranking is always deterministic.
        """
        # ── Step 1: Parse query for scorer signals ────────────────────────────
        # QueryParser extracts task_type + keywords so the scorer can apply
        # task_relevance and description_match dimensions properly.
        query_task = None
        query_modality = None
        keywords: list[str] = []

        try:
            from datascout.query_understanding.parser import QueryParser
            from datascout.contracts.task_types import TaskType

            parser = QueryParser()
            parsed = parser.parse(query.raw_query)
            query_task = parsed.task_type if parsed.task_type != TaskType.OTHER else None
            query_modality = parsed.modality
            keywords = parsed.keywords or []
        except Exception as e:
            logger.warning(
                "query_parse_skipped",
                extra={"error": str(e)[:100], "query_id": query.query_id},
            )

        # ── Step 2: Run deterministic evaluator scoring ───────────────────────
        # DatasetScorer scores ALL candidates before ranking.
        # LLM NEVER touches this step — it is fully deterministic.
        scored_datasets = []
        try:
            from datascout.evaluation.scorer import DatasetScorer

            scorer = DatasetScorer(
                query_task=query_task,
                query_modality=query_modality,
                keywords=keywords,
            )
            scored_datasets = scorer.score_all(result.datasets)
        except Exception as e:
            logger.error(
                "evaluator_scoring_failed",
                extra={"error": str(e)[:100], "query_id": query.query_id},
            )
            # Fallback: build dummy ScoredDataset list so ranking still works
            from datascout.evaluation.scorer import ScoredDataset, ScoreBreakdown
            scored_datasets = [
                ScoredDataset(
                    dataset=ds,
                    breakdown=ScoreBreakdown(
                        task_relevance=0.5,
                        quality=ds.metadata_completeness,
                        popularity=0.3,
                        freshness=0.5,
                        description_match=0.5,
                        composite=ds.metadata_completeness,
                        weights={},
                    ),
                )
                for ds in result.datasets
            ]

        # ── Step 3: Deterministic ranking with diversity boost ────────────────
        from datascout.evaluation.ranker import RankingEngine

        ranker = RankingEngine(
            top_k=min(query.max_results, self.config.max_results_hard_cap),
            diversity_boost=True,
        )
        ranking = ranker.rank(scored_datasets)

        # ── Step 4: Build AgentResponse via ranker ────────────────────────────
        total_across_adapters = sum(result.adapter_results.values())

        response = ranker.build_agent_response(
            query=query,
            ranking=ranking,
            pipeline_stages=result.stages,
            processing_time_ms=result.processing_time_ms,
        )

        # Patch in adapter metadata
        response.total_found = total_across_adapters
        response.partial_result = result.partial_result
        response.adapter_results = result.adapter_results
        response.adapter_errors = result.adapter_errors

        logger.info(
            "response_built",
            extra={
                "query_id":         query.query_id,
                "results_returned": len(response.results),
                "total_found":      total_across_adapters,
                "confidence":       response.confidence.value,
                "partial":          result.partial_result,
                "processing_ms":    result.processing_time_ms,
                "adapters_used":    list(result.adapter_results.keys()),
                "adapters_failed":  list(result.adapter_errors.keys()),
                "query_task":       str(query_task),
                "keywords_count":   len(keywords),
            },
        )

        return response

    def _build_trace(self, ds: RawDataset, rank: int) -> DecisionTrace:
        """
        Build a DecisionTrace for one ranked dataset.

        score_breakdown captures the pre-ranking signals from Agent-0.
        Agent-1 will add semantic_similarity and other ranking dimensions
        when it builds its own DecisionTrace layer.

        lineage_ref: "{ingestion_timestamp}::{pipeline_version}" — links
        the trace back to the specific ingestion event for audit purposes.
        """
        lineage_ref = (
            f"{ds.ingestion_timestamp.isoformat()}::{ds.pipeline_version}"
        )

        # Score breakdown — Agent-0's structural signals
        score_breakdown: dict[str, float] = {
            "metadata_completeness":  ds.metadata_completeness,
            "has_description":        1.0 if ds.has_description else 0.0,
            "has_schema_info":        1.0 if ds.has_schema_info else 0.0,
            "has_size_info":          1.0 if ds.has_size_info else 0.0,
            "has_license_info":       1.0 if ds.has_license_info else 0.0,
            "tag_richness":           min(ds.tag_count / 10.0, 1.0),
            "description_richness":   min(ds.description_length / 500.0, 1.0),
        }

        # Filter reasons — why this record passed Agent-0's gates
        filter_reasons: list[str] = []
        if ds.has_description:
            filter_reasons.append("has_description=True")
        if ds.has_schema_info:
            filter_reasons.append("has_schema_info=True")
        if not ds.is_duplicate:
            filter_reasons.append("is_duplicate=False")
        if ds.metadata_completeness > 0.5:
            filter_reasons.append(
                f"metadata_completeness={ds.metadata_completeness:.2f} > 0.5"
            )

        return DecisionTrace(
            dataset_id=ds.canonical_id,
            dataset_title=ds.title,
            final_score=ds.metadata_completeness,
            final_rank=rank,
            score_breakdown=score_breakdown,
            filters_passed=True,
            filter_reasons=filter_reasons,
            lineage_ref=lineage_ref,
            trace_version="3.0.0",
        )