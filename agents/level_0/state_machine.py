"""
datascout/agents/level_0/state_machine.py

Deterministic finite-state machine for the ScoutAgent ReAct loop.

States (in order):
    OBSERVING   → parse and validate the incoming query
    PLANNING    → decide which adapters / indices to use
    SEARCHING   → execute the search across all sources
    EVALUATING  → score + quality-gate results
    RANKING     → sort final candidate list
    EXPLAINING  → LLM generates natural-language explanation
    COMPLETE    → terminal; results are ready

    ERROR       → terminal error state (non-crashing)

Transitions are validated; illegal moves raise StateMachineError.
Complexity: O(1) per transition.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Any

from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AgentState(str, Enum):
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    SEARCHING = "SEARCHING"
    EVALUATING = "EVALUATING"
    RANKING = "RANKING"
    EXPLAINING = "EXPLAINING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Legal forward transitions (source → set of allowed targets)
_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.OBSERVING: frozenset({AgentState.PLANNING, AgentState.ERROR}),
    AgentState.PLANNING: frozenset({AgentState.SEARCHING, AgentState.ERROR}),
    AgentState.SEARCHING: frozenset({AgentState.EVALUATING, AgentState.ERROR}),
    AgentState.EVALUATING: frozenset(
        {AgentState.RANKING, AgentState.SEARCHING, AgentState.ERROR}
    ),  # SEARCHING allowed for query refinement loop
    AgentState.RANKING: frozenset({AgentState.EXPLAINING, AgentState.ERROR}),
    AgentState.EXPLAINING: frozenset({AgentState.COMPLETE, AgentState.ERROR}),
    AgentState.COMPLETE: frozenset(),
    AgentState.ERROR: frozenset(),
}

_TERMINAL = frozenset({AgentState.COMPLETE, AgentState.ERROR})


class StateMachineError(Exception):
    """Raised on illegal state transitions."""


class AgentStateMachine:
    """
    Tracks agent state and enforces legal transitions.

    Attributes
    ----------
    current : AgentState
        The current state.
    history : list[tuple[AgentState, float]]
        Ordered list of (state, unix_timestamp) pairs.
    """

    def __init__(self) -> None:
        self._state = AgentState.OBSERVING
        self._history: list[tuple[AgentState, float]] = [
            (AgentState.OBSERVING, time.time())
        ]
        logger.info(
            "state_machine_init",
            extra={"initial_state": AgentState.OBSERVING.value},
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def current(self) -> AgentState:
        return self._state

    @property
    def history(self) -> list[tuple[AgentState, float]]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state in _TERMINAL

    # ------------------------------------------------------------------ #
    # Transition                                                           #
    # ------------------------------------------------------------------ #

    def transition(self, target: AgentState) -> None:
        """
        Move to *target* state.

        Raises StateMachineError on illegal transition.
        Complexity: O(1).
        """
        if self._state in _TERMINAL:
            raise StateMachineError(
                f"Cannot transition from terminal state {self._state.value!r}"
            )

        allowed = _TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            raise StateMachineError(
                f"Illegal transition {self._state.value!r} → {target.value!r}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        prev = self._state
        self._state = target
        self._history.append((target, time.time()))

        logger.info(
            "state_transition",
            extra={"from": prev.value, "to": target.value},
        )

    def to_error(self, reason: str = "") -> None:
        """
        Force-transition to ERROR without legality check.
        This is the emergency escape hatch — always safe to call.
        Complexity: O(1).
        """
        prev = self._state
        self._state = AgentState.ERROR
        self._history.append((AgentState.ERROR, time.time()))
        logger.error(
            "state_machine_error",
            extra={"from": prev.value, "reason": reason},
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """Return serialisable state snapshot. Complexity: O(H) where H = history length."""
        return {
            "current": self._state.value,
            "is_terminal": self.is_terminal,
            "history": [
                {"state": s.value, "ts": ts} for s, ts in self._history
            ],
        }