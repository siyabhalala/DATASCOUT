"""
datascout/agents/level_0/react_loop.py

ReAct (Reason + Act) execution loop for ScoutAgent.

Pattern per iteration:
    Thought   → reason about current observations
    Action    → call a tool / subsystem
    Observation → process the tool result

Loop contract:
    - Maximum 2 refinement iterations (SYSTEM RULE)
    - Quality gate: if top_score < 0.4 → refine query
    - All exceptions are caught and logged; loop degrades gracefully

Complexity:
    O(I × (S + E)) where:
        I = iterations (≤ 3 total = 1 initial + 2 refinements)
        S = search cost
        E = evaluation cost
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Protocol

from datascout.contracts.schemas import DatasetCandidate, SearchQuery, SearchResult
from datascout.infrastructure.logging import get_logger
from datascout.infrastructure.metrics import record_metric

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_REFINEMENTS: int = 2
QUALITY_THRESHOLD: float = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Protocols — dependency injection seams
# ─────────────────────────────────────────────────────────────────────────────


class SearchCallable(Protocol):
    """Any callable that takes a SearchQuery and returns list[DatasetCandidate]."""

    def __call__(self, query: SearchQuery) -> list[DatasetCandidate]: ...


class EvaluateCallable(Protocol):
    """Any callable that scores a list of candidates and returns them sorted."""

    def __call__(
        self, candidates: list[DatasetCandidate], query: SearchQuery
    ) -> list[DatasetCandidate]: ...


class RefineCallable(Protocol):
    """Any callable that refines a SearchQuery based on failure observations."""

    def __call__(
        self, query: SearchQuery, observation: "Observation"
    ) -> SearchQuery: ...


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Thought:
    """Reasoning step produced before each action."""

    iteration: int
    reasoning: str
    planned_action: str
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "reasoning": self.reasoning,
            "planned_action": self.planned_action,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass(frozen=True)
class Action:
    """Concrete action taken in one ReAct step."""

    iteration: int
    action_type: str          # "search" | "evaluate" | "refine"
    query: SearchQuery
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action_type": self.action_type,
            "query": self.query.raw_query,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass
class Observation:
    """Result observed after an action."""

    iteration: int
    candidates: list[DatasetCandidate]
    top_score: float
    quality_pass: bool          # True if top_score >= QUALITY_THRESHOLD
    action_type: str
    error: str | None = None
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "num_candidates": len(self.candidates),
            "top_score": self.top_score,
            "quality_pass": self.quality_pass,
            "action_type": self.action_type,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass
class ReActTrace:
    """Full execution trace of one ReAct run."""

    steps: list[tuple[Thought, Action, Observation]] = dataclasses.field(
        default_factory=list
    )
    refinements_used: int = 0
    final_candidates: list[DatasetCandidate] = dataclasses.field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def add_step(
        self, thought: Thought, action: Action, observation: Observation
    ) -> None:
        self.steps.append((thought, action, observation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "thought": t.to_dict(),
                    "action": a.to_dict(),
                    "observation": o.to_dict(),
                }
                for t, a, o in self.steps
            ],
            "refinements_used": self.refinements_used,
            "final_candidates": len(self.final_candidates),
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Loop
# ─────────────────────────────────────────────────────────────────────────────


class ReActLoop:
    """
    Executes the Thought → Action → Observation cycle.

    Parameters
    ----------
    search_fn      : SearchCallable
        Performs dataset retrieval for a query.
    evaluate_fn    : EvaluateCallable
        Scores and sorts candidates.
    refine_fn      : RefineCallable
        Rewrites a failing query.
    on_thought     : optional callback invoked after each Thought
    on_observation : optional callback invoked after each Observation
    """

    def __init__(
        self,
        search_fn: SearchCallable,
        evaluate_fn: EvaluateCallable,
        refine_fn: RefineCallable,
        on_thought: Callable[[Thought], None] | None = None,
        on_observation: Callable[[Observation], None] | None = None,
    ) -> None:
        self._search = search_fn
        self._evaluate = evaluate_fn
        self._refine = refine_fn
        self._on_thought = on_thought
        self._on_observation = on_observation

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _think(self, iteration: int, query: SearchQuery, prev_obs: Observation | None) -> Thought:
        """
        Produce a reasoning step.
        Complexity: O(1).
        """
        if prev_obs is None:
            reasoning = (
                f"Initial search for '{query.raw_query}'. "
                "No prior observations. Will execute broad search."
            )
            planned = "search"
        elif not prev_obs.quality_pass:
            reasoning = (
                f"Iteration {iteration}: top_score={prev_obs.top_score:.3f} < "
                f"{QUALITY_THRESHOLD}. Quality gate failed. "
                "Will refine query and retry."
            )
            planned = "refine+search"
        else:
            reasoning = (
                f"Iteration {iteration}: top_score={prev_obs.top_score:.3f} ≥ "
                f"{QUALITY_THRESHOLD}. Quality gate passed. "
                "Will evaluate and rank."
            )
            planned = "evaluate"

        thought = Thought(
            iteration=iteration,
            reasoning=reasoning,
            planned_action=planned,
        )
        logger.info(
            "react_thought",
            extra=thought.to_dict(),
        )
        if self._on_thought:
            try:
                self._on_thought(thought)
            except Exception:  # noqa: BLE001
                pass
        return thought

    def _act_search(self, iteration: int, query: SearchQuery) -> tuple[Action, Observation]:
        """
        Execute a search action and wrap result in Observation.
        Complexity: O(cost of search_fn).
        """
        action = Action(iteration=iteration, action_type="search", query=query)
        logger.info("react_action", extra=action.to_dict())

        try:
            candidates = self._search(query)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "react_search_error",
                extra={"iteration": iteration, "error": str(exc)},
            )
            obs = Observation(
                iteration=iteration,
                candidates=[],
                top_score=0.0,
                quality_pass=False,
                action_type="search",
                error=str(exc),
            )
            return action, obs

        top_score = candidates[0].final_score if candidates else 0.0
        obs = Observation(
            iteration=iteration,
            candidates=candidates,
            top_score=top_score,
            quality_pass=top_score >= QUALITY_THRESHOLD,
            action_type="search",
        )
        return action, obs

    def _act_evaluate(
        self, iteration: int, query: SearchQuery, candidates: list[DatasetCandidate]
    ) -> tuple[Action, Observation]:
        """
        Score and sort candidates.
        Complexity: O(cost of evaluate_fn).
        """
        action = Action(iteration=iteration, action_type="evaluate", query=query)
        logger.info("react_action", extra=action.to_dict())

        try:
            ranked = self._evaluate(candidates, query)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "react_evaluate_error",
                extra={"iteration": iteration, "error": str(exc)},
            )
            ranked = candidates  # degraded: return unranked

        top_score = ranked[0].final_score if ranked else 0.0
        obs = Observation(
            iteration=iteration,
            candidates=ranked,
            top_score=top_score,
            quality_pass=top_score >= QUALITY_THRESHOLD,
            action_type="evaluate",
        )
        return action, obs

    def _emit_observation(self, obs: Observation) -> None:
        logger.info("react_observation", extra=obs.to_dict())
        record_metric("react_loop.top_score", obs.top_score)
        record_metric("react_loop.candidates", len(obs.candidates))
        if self._on_observation:
            try:
                self._on_observation(obs)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self, query: SearchQuery) -> ReActTrace:
        """
        Execute the full ReAct loop.

        Returns ReActTrace regardless of errors (never raises).
        Complexity: O(I × (S + E)) where I ≤ 3.
        """
        t0 = time.perf_counter()
        trace = ReActTrace()
        current_query = query
        prev_obs: Observation | None = None
        iteration = 0

        while True:
            iteration += 1

            # ── THINK ──────────────────────────────────────────────────
            thought = self._think(iteration, current_query, prev_obs)

            # ── ACT: SEARCH ────────────────────────────────────────────
            action, obs = self._act_search(iteration, current_query)
            self._emit_observation(obs)
            trace.add_step(thought, action, obs)
            prev_obs = obs

            # ── OBSERVE: quality gate ──────────────────────────────────
            if obs.quality_pass or trace.refinements_used >= MAX_REFINEMENTS:
                # Gate passed OR we've exhausted refinements → move to evaluate
                if not obs.quality_pass:
                    logger.warning(
                        "react_max_refinements_reached",
                        extra={
                            "max_refinements": MAX_REFINEMENTS,
                            "top_score": obs.top_score,
                        },
                    )
                break

            # ── REFINE query and loop ──────────────────────────────────
            try:
                current_query = self._refine(current_query, obs)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "react_refine_error",
                    extra={"iteration": iteration, "error": str(exc)},
                )
                break

            trace.refinements_used += 1
            record_metric("react_loop.refinements", trace.refinements_used)

        # ── EVALUATE final candidates ──────────────────────────────────
        eval_iter = iteration + 1
        eval_thought = self._think(eval_iter, current_query, prev_obs)
        eval_action, eval_obs = self._act_evaluate(
            eval_iter, current_query, prev_obs.candidates if prev_obs else []
        )
        self._emit_observation(eval_obs)
        trace.add_step(eval_thought, eval_action, eval_obs)

        trace.final_candidates = eval_obs.candidates
        trace.total_elapsed_ms = (time.perf_counter() - t0) * 1_000

        logger.info(
            "react_loop_complete",
            extra={
                "iterations": iteration,
                "refinements": trace.refinements_used,
                "final_candidates": len(trace.final_candidates),
                "elapsed_ms": round(trace.total_elapsed_ms, 2),
            },
        )
        return trace