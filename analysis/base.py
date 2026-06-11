"""
datascout.analysis.base
────────────────────────
Abstract base types for all dataset analysis components.

The analysis layer sits *after* ingestion and *before* evaluation.  Its job
is to derive additional signals from a ``RawDataset`` (quality scores,
summaries, target-column candidates) that the evaluation pipeline then
uses for scoring and explanation.

Every analysis component derives from ``AnalysisBase`` and receives its
runtime parameters via ``AnalysisConfig``.  Errors are wrapped in
``AnalysisError`` so the pipeline can distinguish analysis failures from
unexpected programming errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "AnalysisBase",
    "AnalysisConfig",
    "AnalysisError",
]


class AnalysisBase(ABC):
    """Abstract base class for all DataScout analysis components.

    Subclasses implement :meth:`analyze` to accept an arbitrary input (a
    ``RawDataset``, a list of column names, a file path, etc.) and return
    an analysis-specific result object.

    All implementations **must not raise** to their callers.  Catch internal
    errors, log them, and return a safe default (an empty dataclass, zeros,
    ``None``).

    Example
    -------
    >>> class ColumnProfiler(AnalysisBase):
    ...     @property
    ...     def name(self) -> str:
    ...         return "column_profiler"
    ...
    ...     def analyze(self, data: list[str]) -> dict:
    ...         return {"columns": data, "count": len(data)}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier for this analysis component.

        Used in log messages and for keying results in the analysis manifest.
        """

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """Run analysis on *data* and return the result.

        Parameters
        ----------
        data:
            Component-specific input.  Type is declared by each concrete
            subclass in its own docstring.

        Returns
        -------
        Any:
            Component-specific result object.  Must never be ``None``; return
            an empty/default instance of the result type instead.
        """


@dataclass
class AnalysisConfig:
    """Runtime configuration shared across all analysis components.

    Pass a single ``AnalysisConfig`` instance into every component that needs
    it so that global limits (max sample size, timeout) are applied
    consistently across the pipeline.

    Attributes
    ----------
    max_sample_rows:
        Maximum number of rows to sample when profiling large datasets.
        Keeps analysis bounded for datasets with millions of rows.
    timeout_seconds:
        Per-component wall-clock timeout.  If a component exceeds this, it
        should abort and return a partial/default result.
    enable_expensive_checks:
        When ``False`` (default), skip computationally expensive checks such
        as full duplicate detection or deep schema inference.  Set to ``True``
        in batch/offline pipelines where latency is less critical.
    """

    max_sample_rows: int = 10_000
    timeout_seconds: float = 30.0
    enable_expensive_checks: bool = False


class AnalysisError(Exception):
    """Raised (internally) when an analysis component encounters a fatal error.

    Should be caught at the pipeline boundary and converted to a log warning
    plus safe default result — never propagated to the API layer.

    Attributes
    ----------
    component:
        The :attr:`AnalysisBase.name` of the component that raised the error.
    cause:
        The original exception that triggered this error, if available.
        Stored for debugging but not re-raised automatically.
    """

    def __init__(
        self,
        message: str,
        component: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialise the analysis error.

        Parameters
        ----------
        message:
            Human-readable description of what went wrong.
        component:
            Name of the analysis component that failed.
        cause:
            Original exception for debug context.
        """
        super().__init__(message)
        self.component: str = component
        self.cause: Optional[Exception] = cause

    def __str__(self) -> str:
        base = f"[{self.component}] {super().__str__()}"
        if self.cause:
            return f"{base} — caused by {type(self.cause).__name__}: {self.cause}"
        return base