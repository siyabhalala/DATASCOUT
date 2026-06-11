"""
datascout.agents.scout_agent
─────────────────────────────
Level-0 ScoutAgent — fully async-native.

WHY THIS EXISTS:
    The Level-0 ReActLoop (level_0/react_loop.py) is synchronous by design.
    Every attempt to bridge sync→async via threads deadlocks on Windows+Python
    3.10 because asyncio does not allow nested event loops in the same thread.

    This file re-implements the ReAct logic (Thought→Action→Observe→Refine)
    natively in async, calling adapters with proper await. The level_0/ files
    remain for architecture reference but execution is driven here.

GUARANTEE:
    Never returns a blank page. On total failure → returns ([], trace) which
    causes search_v2 to call _no_results_response → Gemini friendly message.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["ScoutAgent", "AgentTrace", "AdapterFailure"]

MAX_REFINEMENTS: int = 2
# Quality gate: pass if we have ANY candidates (≥1)
# Real scoring happens in EvaluatorPipeline — not here
MIN_CANDIDATES: int = 1


@dataclass
class AdapterFailure:
    source: str
    reason: str
    error_message: str = ""
    attempted_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass
class AgentTrace:
    query_original: str = ""
    query_used: str = ""
    iterations: int = 0
    broadening_applied: bool = False
    refinements_used: int = 0
    adapter_failures: List[AdapterFailure] = field(default_factory=list)
    sources_succeeded: List[str] = field(default_factory=list)
    react_trace: Optional[Any] = None
    state_history: Optional[List[Any]] = None
    elapsed_ms: float = 0.0

    def to_user_message(self) -> str:
        if self.adapter_failures and not self.sources_succeeded:
            issues, seen = [], set()
            for f in self.adapter_failures:
                if f.source in seen:
                    continue
                seen.add(f.source)
                label = {
                    "auth_failed":  "credentials not configured",
                    "timeout":      "timed out",
                    "rate_limited": "rate limited",
                }.get(f.reason, "unavailable")
                issues.append(f"{f.source.title()}: {label}")
            return f"Search failed — {'; '.join(issues)}."
        if self.refinements_used > 0:
            return (
                f"Query refined {self.refinements_used} time(s). "
                f"Showing best matches for '{self.query_used}'."
            )
        return ""


class ScoutAgent:
    """
    Async-native Level-0 agent.
    Called by search_v2: raw_datasets, trace = await agent.search(...)
    """

    async def search(
        self,
        query: str,
        sources: List[str],
        max_results: int,
    ) -> Tuple[List[Any], AgentTrace]:
        t0 = time.perf_counter()
        trace = AgentTrace(query_original=query, query_used=query)
        try:
            raw_datasets, trace = await self._react_cycle(query, sources, max_results)
            trace.elapsed_ms = (time.perf_counter() - t0) * 1_000
            logger.info(
                "level0_complete",
                extra={
                    "query":    query[:60],
                    "datasets": len(raw_datasets),
                    "sources":  trace.sources_succeeded,
                    "refines":  trace.refinements_used,
                    "ms":       round(trace.elapsed_ms, 1),
                },
            )
            return raw_datasets, trace
        except Exception as exc:
            logger.error(
                "scout_agent_fatal",
                extra={"query": query[:60], "error": str(exc)[:200]},
                exc_info=True,
            )
            trace.elapsed_ms = (time.perf_counter() - t0) * 1_000
            return await self._fallback(query, sources, max_results, trace)

    async def _react_cycle(
        self,
        query: str,
        sources: List[str],
        max_results: int,
    ) -> Tuple[List[Any], AgentTrace]:
        from datascout.contracts.requests import SearchQuery  # noqa

        original_expanded = query   # preserve the enriched query for all SearchQuery instances
        current_query    = query
        refinements      = 0
        iterations       = 0
        all_failures: List[AdapterFailure] = []
        all_succeeded: List[str] = []
        raw_datasets: List[Any] = []

        while True:
            iterations += 1

            # THOUGHT
            logger.info(
                "react_thought",
                extra={"iteration": iterations, "query": current_query[:60]},
            )

            # ACTION: search
            # FIX: populate expanded_query so all 3 adapters use the enriched query
            # even when current_query is a broadened/shortened fallback on retry iterations
            sq = SearchQuery(raw_query=current_query, expanded_query=original_expanded)
            datasets, failures, succeeded = await self._fan_out(
                sq, sources, max_results
            )

            for f in failures:
                if not any(x.source == f.source for x in all_failures):
                    all_failures.append(f)
            for s in succeeded:
                if s not in all_succeeded:
                    all_succeeded.append(s)

            # OBSERVE: did we get anything?
            got_results = len(datasets) >= MIN_CANDIDATES

            logger.info(
                "react_observation",
                extra={
                    "iteration":   iterations,
                    "candidates":  len(datasets),
                    "got_results": got_results,
                    "query":       current_query[:60],
                },
            )

            if got_results or refinements >= MAX_REFINEMENTS:
                raw_datasets = datasets
                if not got_results:
                    logger.warning(
                        "react_max_refinements_reached",
                        extra={"candidates": len(datasets), "query": current_query[:60]},
                    )
                break

            # REFINE
            current_query = self._refine(current_query, len(datasets))
            refinements  += 1
            logger.info(
                "react_refined",
                extra={"new_query": current_query[:60], "refinement": refinements},
            )

        trace = AgentTrace(
            query_original=query,
            query_used=current_query,
            iterations=iterations,
            broadening_applied=refinements > 0,
            refinements_used=refinements,
            adapter_failures=all_failures,
            sources_succeeded=all_succeeded,
        )
        return raw_datasets, trace

    async def _fan_out(
        self,
        sq: Any,
        sources: List[str],
        max_results: int,
    ) -> Tuple[List[Any], List[AdapterFailure], List[str]]:
        """
        Multi-query fan-out — sends 2-3 short focused queries per adapter.

        WHY: Kaggle/HuggingFace return perfect results for short queries like
        "crop disease detection" but garbage for long enriched sentences like
        "i m building a crop disease detector for indian farms plant pathology..."
        We generate short queries, run all in parallel, deduplicate by id.
        """
        import importlib
        from datascout.contracts.requests import SearchQuery  # noqa

        # Build short focused queries from enriched text
        short_queries = self._build_short_queries(sq.raw_query)
        logger.info(
            "multi_query_fan_out",
            extra={"queries": short_queries, "sources": sources},
        )

        adapters: dict = {}
        for name, mod_path, cls_name in [
            ("kaggle",      "datascout.adapters.kaggle_adapter",     "KaggleAdapter"),
            ("huggingface", "datascout.adapters.huggingface_adapter", "HuggingFaceAdapter"),
            ("openml",      "datascout.adapters.openml_adapter",      "OpenMLAdapter"),
        ]:
            if name not in sources:
                continue
            # NOTE: OpenML is now included — timeout reduced to 15s in OpenMLAdapter._do_search
            # to fail fast without blocking. It participates like any other adapter.
            try:
                mod = importlib.import_module(mod_path)
                adapters[name] = getattr(mod, cls_name)()
            except Exception as exc:
                logger.warning(
                    "adapter_init_failed",
                    extra={"source": name, "error": str(exc)[:80]},
                )

        if not adapters:
            return [], [], []

        async def _call(name: str, adapter: Any, query: str):
            try:
                sq_short = SearchQuery(raw_query=query, expanded_query=sq.expanded_query, max_results=max_results)  # FIX: pass enriched query through to adapters
                results = await adapter.search(sq_short)
                return name, results or [], None
            except Exception as exc:
                return name, [], _classify_error(exc)

        # Run all adapter × query combinations in parallel
        tasks = [
            _call(name, adapter, q)
            for name, adapter in adapters.items()
            for q in short_queries
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        failures:  List[AdapterFailure] = []
        succeeded_set: set = set()
        seen_ids: set = set()
        raw_datasets: List[Any] = []

        for item in gathered:
            if isinstance(item, Exception):
                continue
            name, datasets, reason = item
            if reason:
                if not any(f.source == name for f in failures):
                    failures.append(AdapterFailure(
                        source=name, reason=reason, error_message=reason
                    ))
            elif datasets:
                succeeded_set.add(name)
                for ds in datasets:
                    # Deduplicate by canonical_id or title
                    cid = getattr(ds, "canonical_id", None) or getattr(ds, "title", "")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        raw_datasets.append(ds)

        return raw_datasets, failures, list(succeeded_set)

    @staticmethod
    def _build_short_queries(enriched_query: str) -> List[str]:
        """
        Extract 2-3 short focused search queries from the enriched query.

        "i m building a crop disease detector for indian farms plant pathology..."
        → ["crop disease detection", "plant disease indian", "leaf disease agriculture"]

        "speech recognition hindi gujarati asr indic..."
        → ["speech recognition hindi", "asr indic language", "hindi audio"]
        """
        STOP = {
            "i", "m", "a", "an", "the", "for", "in", "on", "with", "and",
            "or", "is", "are", "to", "that", "this", "of", "my", "we", "our",
            "im", "building", "want", "need", "using", "use", "can", "how",
            "what", "get", "make", "create", "build", "train", "model",
            "please", "help", "find", "give", "show", "looking",
            # Pure structural noise — safe to strip
            "datasets", "dataset", "data", "csv", "file", "files",
            "machine", "learning", "deep", "neural", "network",
            # FIX v3.5.0: REMOVED "classification", "prediction", "analysis",
            # "detection" — these are load-bearing domain discriminators.
            # "plant disease detection" → strips to "plant disease" and returns
            # agricultural research datasets instead of CV detection benchmarks.
            # "fraud detection tabular" → strips to "fraud tabular" (nonsense).
            # Kaggle and HuggingFace rank on exact keyword overlap with dataset
            # titles/tags; removing these terms destroys query specificity.
        }
        tokens = [
            t.lower() for t in enriched_query.replace("-", " ").split()
            if len(t) >= 3 and t.lower() not in STOP
        ]
        # Deduplicate preserving order
        seen, unique = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        if not unique:
            return [enriched_query]

        queries = []
        # Q1: first 3 tokens — core concept
        queries.append(" ".join(unique[:3]) if len(unique) >= 3 else " ".join(unique))
        # Q2: tokens 2-5 — secondary angle
        if len(unique) >= 5:
            queries.append(" ".join(unique[2:5]))
        # Q3: first + last two tokens — specificity combo
        if len(unique) >= 6:
            queries.append(" ".join([unique[0], unique[-2], unique[-1]]))

        return list(dict.fromkeys(queries))[:3]  # dedupe, max 3

    @staticmethod
    def _refine(query: str, result_count: int) -> str:
        """Broaden query for retry if no results found."""
        STOP = frozenset({
            "the","a","an","of","for","in","on","with",
            "and","or","is","are","to","that","this",
        })
        if result_count == 0:
            tokens = [t for t in query.lower().split() if t not in STOP]
            return " ".join(tokens[:4]) if tokens else query
        return f"{query} dataset" if "dataset" not in query.lower() else query

    async def _fallback(
        self,
        query: str,
        sources: List[str],
        max_results: int,
        trace: AgentTrace,
    ) -> Tuple[List[Any], AgentTrace]:
        logger.warning("falling_back_to_legacy_engine", extra={"query": query[:60]})
        try:
            from datascout.agents.scout_engine import ScoutAgent as Legacy  # noqa
            return await Legacy().search(
                query=query, sources=sources, max_results=max_results
            )
        except Exception as exc:
            logger.error("fallback_also_failed", extra={"error": str(exc)[:200]})
            return [], trace


def _classify_error(exc: Exception) -> str:
    try:
        from datascout.contracts.errors.exceptions import (  # noqa
            AdapterAuthError, AdapterTimeoutError, AdapterRateLimitError,
        )
        if isinstance(exc, AdapterAuthError):      return "auth_failed"
        if isinstance(exc, AdapterTimeoutError):   return "timeout"
        if isinstance(exc, AdapterRateLimitError): return "rate_limited"
    except Exception:
        pass
    name = type(exc).__name__.lower()
    if "timeout"  in name: return "timeout"
    if "auth"     in name: return "auth_failed"
    if "connect"  in name: return "connection_error"
    return "unknown"