"""
datascout.evaluation.base
──────────────────────────
Abstract base types shared across all evaluation components.

Every evaluator in the DataScout pipeline (scorer, ranker, explainability,
diagnostics) derives from ``BaseEvaluator`` and returns its results wrapped
in an ``EvaluatorResult``.  This contract enables uniform error handling,
timing instrumentation, and pluggable evaluator composition in
``EvaluatorPipeline``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from datascout.contracts.models import RawDataset

__all__ = [
    "BaseEvaluator",
    "EvaluatorResult",
]


class BaseEvaluator(ABC):
    """Abstract base class for all DataScout evaluator components.

    Subclasses must implement :meth:`evaluate` and :attr:`name`.  The
    ``version`` class attribute should be overridden per evaluator when a
    breaking change in scoring logic is shipped so that cached results can be
    invalidated.

    Usage
    -----
    >>> class MyScorer(BaseEvaluator):
    ...     version = "2.1.0"
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_scorer"
    ...
    ...     def evaluate(self, datasets: list[RawDataset]) -> list:
    ...         return [score(d) for d in datasets]
    """

    #: Semantic version of this evaluator's scoring logic.  Increment the
    #: minor version for non-breaking improvements, the major version for
    #: changes that alter result ordering.
    version: str = "1.0.0"

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable identifier for this evaluator.

        Used in logging, metrics labels, and ``EvaluatorResult.evaluator_name``.
        Must be a valid Python identifier (lowercase, underscores).
        """

    @abstractmethod
    def evaluate(self, datasets: list[RawDataset]) -> list:
        """Run the evaluation over a batch of datasets.

        Parameters
        ----------
        datasets:
            The raw datasets to evaluate.  May be an empty list; the
            implementation must handle that gracefully and return ``[]``.

        Returns
        -------
        list:
            Evaluated/annotated datasets.  The concrete type depends on the
            subclass (e.g. ``list[ScoredDataset]``, ``list[RankedDataset]``).

        Notes
        -----
        Implementations **must not raise**.  Catch all exceptions internally,
        log them, and return a partial or empty result.
        """

    def timed_evaluate(self, datasets: list[RawDataset]) -> "EvaluatorResult":
        """Convenience wrapper that times :meth:`evaluate` and wraps the
        output in an :class:`EvaluatorResult`.

        Parameters
        ----------
        datasets:
            Passed through to :meth:`evaluate`.

        Returns
        -------
        EvaluatorResult:
            Always returned, even when :meth:`evaluate` encounters an error
            internally (the ``error`` field will be set in that case).
        """
        t0 = time.perf_counter()
        error: Optional[str] = None
        result: list = []
        try:
            result = self.evaluate(datasets)
        except Exception as exc:  # pragma: no cover — safety net
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        return EvaluatorResult(
            datasets=result,
            evaluator_name=self.name,
            evaluation_time_ms=elapsed_ms,
            error=error,
        )


@dataclass
class EvaluatorResult:
    """Container for the output of a single evaluator pass.

    Attributes
    ----------
    datasets:
        The evaluated/annotated dataset objects returned by the evaluator.
        May be empty if an error occurred or the input was empty.
    evaluator_name:
        The :attr:`BaseEvaluator.name` of the evaluator that produced this
        result.
    evaluation_time_ms:
        Wall-clock time taken by the :meth:`BaseEvaluator.evaluate` call, in
        milliseconds.  Useful for pipeline latency breakdown.
    error:
        If the evaluator raised an unexpected exception, its string
        representation is stored here.  ``None`` means success.
    """

    datasets: list = field(default_factory=list)
    evaluator_name: str = ""
    evaluation_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """``True`` when no error was recorded."""
        return self.error is None

    def __len__(self) -> int:
        return len(self.datasets)