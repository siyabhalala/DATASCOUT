"""
datascout/agents/level_0/scout_agent.py

Level-0 ScoutAgent: the top-level orchestrator for DATASCOUT.

Wires together:
    HybridEngine (Phase 10) → semantic + BM25 search
    Evaluator    (Phase  9) → deterministic scoring
    LLMProvider  (Phase 11) → natural-language explanation ONLY (never ranking)
    ReActLoop               → Thought → Action → Observation cycle
    AgentStateMachine       → enforces legal state transitions

Design constraints:
    - LLM NEVER scores or ranks — only explains (SYSTEM RULE #2)
    - ALL scoring is deterministic math (SYSTEM RULE #3)
    - Max 2 query refinements (SYSTEM RULE per Phase 12 spec)
    - Never raises — always returns a SearchResult (SYSTEM RULE #1)

Complexity: O(I × (H + E)) where
    I = iterations ≤ 3,
    H = hybrid search cost,
    E = evaluation cost.
"""

from __future__ import annotations

import time
from typing import Any

from datascout.agents.base import BaseAgent
from datascout.agents.level_0.react_loop import (
    MAX_REFINEMENTS,
    QUALITY_THRESHOLD,
    Observation,
    ReActLoop,
    ReActTrace,
    Thought,
)
from datascout.agents.level_0.state_machine import AgentState, AgentStateMachine
from datascout.contracts.schemas import DatasetCandidate, SearchQuery, SearchResult
from datascout.infrastructure.logging import get_logger
from datascout.infrastructure.metrics import record_metric

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Query Refiner (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────────────


def _default_refine(query: SearchQuery, obs: Observation) -> SearchQuery:
    """
    Heuristic query refinement when quality gate fails.

    Strategy (in order):
        1. If top_score == 0.0 → strip stop-words and retry bare terms
        2. Otherwise → append "dataset" suffix to broaden adapter hits

    LLM is deliberately NOT used here (SYSTEM RULE #2 / #3).
    Complexity: O(|query|).
    """
    raw = query.raw_query.strip()

    STOP_WORDS = frozenset(
        {
            "the", "a", "an", "of", "for", "in", "on", "with",
            "and", "or", "is", "are", "to", "that", "this",
        }
    )

    if obs.top_score == 0.0:
        tokens = [t for t in raw.lower().split() if t not in STOP_WORDS]
        refined_raw = " ".join(tokens) if tokens else raw
    else:
        refined_raw = f"{raw} dataset" if "dataset" not in raw.lower() else raw

    logger.info(
        "query_refined",
        extra={
            "original": raw,
            "refined": refined_raw,
            "top_score": obs.top_score,
        },
    )

    # Build a new SearchQuery — preserving all original fields where possible.
    # We construct via dict to avoid depending on exact constructor signature
    # (forward-compatibility with Phase 9 contracts).
    try:
        refined = SearchQuery(
            raw_query=refined_raw,
            task_type=query.task_type,
            domain=getattr(query, "domain", None),
            min_samples=getattr(query, "min_samples", None),
            max_samples=getattr(query, "max_samples", None),
            required_features=getattr(query, "required_features", []),
            preferred_licenses=getattr(query, "preferred_licenses", []),
        )
    except Exception:  # noqa: BLE001
        # Fallback: clone with only raw_query changed
        refined = SearchQuery(raw_query=refined_raw)

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# ScoutAgent
# ─────────────────────────────────────────────────────────────────────────────


class ScoutAgent(BaseAgent):
    """
    Level-0 DATASCOUT agent.

    Parameters
    ----------
    hybrid_engine : HybridEngine-like
        Must expose .search(query: SearchQuery) → list[DatasetCandidate].
        Accepts any object satisfying the protocol (duck typing).
    evaluator : Evaluator-like
        Must expose .evaluate(candidates, query) → list[DatasetCandidate].
    llm_provider : BaseLLMProvider-like | None
        Must expose .explain(candidates, query) → str.
        If None, explanation step is skipped gracefully.
    refine_fn : callable | None
        Custom query refiner.  Defaults to _default_refine.
    max_results : int
        Maximum datasets to return in final SearchResult.
    """

    _NAME = "ScoutAgent"
    _VERSION = "1.0.0"

    def __init__(
        self,
        hybrid_engine: Any,
        evaluator: Any,
        llm_provider: Any | None = None,
        refine_fn: Any | None = None,
        max_results: int = 20,
    ) -> None:
        self._hybrid = hybrid_engine
        self._evaluator = evaluator
        self._llm = llm_provider
        self._refine_fn = refine_fn or _default_refine
        self._max_results = max_results

    # ------------------------------------------------------------------ #
    # BaseAgent identity                                                   #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    # ------------------------------------------------------------------ #
    # Internal search / evaluate / explain hooks                          #
    # ------------------------------------------------------------------ #

    def _search(self, query: SearchQuery) -> list[DatasetCandidate]:
        """
        Delegate to HybridEngine.
        Complexity: O(cost of hybrid_engine.search).
        """
        try:
            candidates = self._hybrid.search(query)
            record_metric("scout_agent.search.candidates", len(candidates))
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "scout_search_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            return []

    def _evaluate(
        self, candidates: list[DatasetCandidate], query: SearchQuery
    ) -> list[DatasetCandidate]:
        """
        Delegate to Evaluator — deterministic scoring only.
        Complexity: O(N log N) where N = len(candidates).
        """
        if not candidates:
            return []
        try:
            ranked = self._evaluator.evaluate(candidates, query)
            return ranked[: self._max_results]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "scout_evaluate_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            # Degrade: sort by whatever score field exists
            try:
                return sorted(
                    candidates,
                    key=lambda c: c.final_score,
                    reverse=True,
                )[: self._max_results]
            except Exception:  # noqa: BLE001
                return candidates[: self._max_results]

    def _explain(
        self, candidates: list[DatasetCandidate], query: SearchQuery
    ) -> str:
        """
        LLM explains the top results — NEVER ranks or scores.
        Falls back to template string if LLM unavailable.
        Complexity: O(cost of LLM call) or O(N) for template.
        """
        if self._llm is None:
            return self._template_explanation(candidates, query)

        try:
            explanation = self._llm.explain(candidates, query)
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("Empty explanation from LLM")
            return explanation
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scout_llm_explain_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            return self._template_explanation(candidates, query)

    @staticmethod
    def _template_explanation(
        candidates: list[DatasetCandidate], query: SearchQuery
    ) -> str:
        """
        Deterministic fallback explanation.
        No LLM required — safe for all environments.
        Complexity: O(N).
        """
        if not candidates:
            return (
                f"No datasets found for query '{query.raw_query}'. "
                "Try broadening your search terms."
            )

        top = candidates[:3]
        names = ", ".join(
            f"'{c.title}' (score: {c.final_score:.2f})" for c in top
        )
        return (
            f"Found {len(candidates)} dataset(s) for '{query.raw_query}'. "
            f"Top results: {names}."
        )

    # ------------------------------------------------------------------ #
    # State machine driven orchestration                                   #
    # ------------------------------------------------------------------ #

    def _run_impl(self, query: SearchQuery) -> SearchResult:
        """
        Full ReAct pipeline with state machine enforcement.

        Stages:
            OBSERVING  → validate query
            PLANNING   → configure ReActLoop
            SEARCHING  → ReActLoop.run()  (handles SEARCHING + EVALUATING internally)
            EVALUATING → already done inside ReActLoop
            RANKING    → final sort + truncate
            EXPLAINING → LLM explain
            COMPLETE   → assemble SearchResult

        Complexity: O(I × (H + E) + L) where L = LLM cost.
        """
        sm = AgentStateMachine()
        t0 = time.perf_counter()

        # ── OBSERVING ─────────────────────────────────────────────────
        try:
            self._observe(query)
        except Exception as exc:  # noqa: BLE001
            sm.to_error(str(exc))
            return self._error_result(query, str(exc))

        # ── PLANNING ──────────────────────────────────────────────────
        try:
            sm.transition(AgentState.PLANNING)
            loop = self._plan(query)
        except Exception as exc:  # noqa: BLE001
            sm.to_error(str(exc))
            return self._error_result(query, str(exc))

        # ── SEARCHING (ReAct loop handles SEARCHING + EVALUATING) ─────
        try:
            sm.transition(AgentState.SEARCHING)
            trace: ReActTrace = loop.run(query)
        except Exception as exc:  # noqa: BLE001
            sm.to_error(str(exc))
            return self._error_result(query, str(exc))

        # ── EVALUATING already embedded in loop; transition for record ─
        try:
            sm.transition(AgentState.EVALUATING)
        except Exception:  # noqa: BLE001
            pass  # Non-critical; do not abort

        # ── RANKING ───────────────────────────────────────────────────
        try:
            sm.transition(AgentState.RANKING)
            ranked = self._rank(trace.final_candidates)
        except Exception as exc:  # noqa: BLE001
            sm.to_error(str(exc))
            ranked = trace.final_candidates  # degrade

        # ── EXPLAINING ────────────────────────────────────────────────
        try:
            sm.transition(AgentState.EXPLAINING)
            explanation = self._explain(ranked, query)
        except Exception as exc:  # noqa: BLE001
            sm.to_error(str(exc))
            explanation = self._template_explanation(ranked, query)

        # ── COMPLETE ──────────────────────────────────────────────────
        try:
            sm.transition(AgentState.COMPLETE)
        except Exception:  # noqa: BLE001
            pass

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        record_metric("scout_agent.total_elapsed_ms", elapsed_ms)
        record_metric("scout_agent.final_results", len(ranked))

        return SearchResult(
            query=query,
            datasets=ranked,
            metadata={
                "agent": self.name,
                "version": self.version,
                "react_trace": trace.to_dict(),
                "state_history": sm.snapshot()["history"],
                "explanation": explanation,
                "elapsed_ms": round(elapsed_ms, 2),
                "refinements_used": trace.refinements_used,
                "quality_threshold": QUALITY_THRESHOLD,
                "max_refinements": MAX_REFINEMENTS,
            },
        )

    def _observe(self, query: SearchQuery) -> None:
        """Validate the query in OBSERVING state. Raises on hard failures."""
        if not query.raw_query or not query.raw_query.strip():
            raise ValueError("Empty query supplied to ScoutAgent")
        logger.info(
            "scout_observing",
            extra={"query": query.raw_query},
        )

    def _plan(self, query: SearchQuery) -> ReActLoop:
        """Configure and return the ReActLoop. Complexity: O(1)."""
        logger.info(
            "scout_planning",
            extra={"query": query.raw_query},
        )
        return ReActLoop(
            search_fn=self._search,
            evaluate_fn=self._evaluate,
            refine_fn=self._refine_fn,
        )

    def _rank(self, candidates: list[DatasetCandidate]) -> list[DatasetCandidate]:
        """
        Final deterministic sort by final_score DESC, title ASC as tiebreak.
        Complexity: O(N log N).
        """
        return sorted(
            candidates,
            key=lambda c: (-c.final_score, c.title.lower()),
        )[: self._max_results]

    # ------------------------------------------------------------------ #
    # Error helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _error_result(query: SearchQuery, reason: str) -> SearchResult:
        return SearchResult(
            query=query,
            datasets=[],
            metadata={"agent_error": reason},
        )