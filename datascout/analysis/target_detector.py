"""
datascout.analysis.target_detector
────────────────────────────────────
Heuristic detection of likely target / label columns in a dataset.

``TargetDetector`` scans a list of column names and identifies which column
is most likely to be the prediction target (label, class, output) based on
naming conventions widely used in ML datasets.  No data values are inspected
— this is a name-only heuristic that runs in O(n) where n is the number of
columns.

Detection strategy
──────────────────
1. **Exact match** — columns named exactly ``target``, ``label``, ``class``,
   ``y``, ``output``, ``outcome``, ``result`` → confidence 0.95.
2. **Suffix match** — columns whose name *contains* ``_label``, ``_class``, or
   ``_target`` (e.g. ``sentiment_label``) → confidence 0.75.
3. **No match** — method returns with empty candidates and
   ``detection_method="none"``.

Multiple candidates are returned in descending confidence order.  The
``best_candidate`` field always points to the highest-confidence candidate,
or ``None`` if no candidates were found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "TargetCandidate",
    "TargetDetectionResult",
    "TargetDetector",
]

# Exact column names that strongly indicate the target variable
_EXACT_TARGET_NAMES: frozenset[str] = frozenset(
    {"target", "label", "class", "y", "output", "outcome", "result"}
)

# Substrings that indicate a target/label column (checked as lower-case suffix)
_SUFFIX_SIGNALS: tuple[str, ...] = ("_label", "_class", "_target")

_CONFIDENCE_EXACT: float = 0.95
_CONFIDENCE_SUFFIX: float = 0.75


@dataclass
class TargetCandidate:
    """A single column that is a candidate target variable.

    Attributes
    ----------
    column_name:
        The original column name (preserves casing).
    confidence:
        Heuristic confidence in [0, 1] that this column is the target.
    reason:
        Short human-readable explanation of why this column was selected
        (e.g. ``"Exact match: 'label'"``).
    """

    column_name: str
    confidence: float
    reason: str


@dataclass
class TargetDetectionResult:
    """The complete result of a target-detection run.

    Attributes
    ----------
    candidates:
        All columns identified as possible targets, sorted by confidence
        descending.
    best_candidate:
        The highest-confidence candidate, or ``None`` if none were found.
    detection_method:
        One of:
        - ``"column_name_heuristic"`` — exact or suffix name match found.
        - ``"none"`` — no heuristic matched.
    """

    candidates: List[TargetCandidate] = field(default_factory=list)
    best_candidate: Optional[TargetCandidate] = None
    detection_method: str = "none"


class TargetDetector:
    """Detect likely target columns from a list of column names.

    The detector is stateless; a single instance can be reused across the
    entire pipeline without thread-safety concerns.

    Example
    -------
    >>> detector = TargetDetector()
    >>> result = detector.detect(["age", "fare", "survived"])
    >>> result.best_candidate
    None  # "survived" not in exact list — no match

    >>> result = detector.detect(["age", "fare", "label"])
    >>> result.best_candidate.column_name
    'label'
    >>> result.best_candidate.confidence
    0.95
    """

    def detect(self, column_names: list[str]) -> TargetDetectionResult:
        """Detect likely target columns from *column_names*.

        This method **never raises**.  If *column_names* is empty or ``None``,
        an empty result with ``detection_method="none"`` is returned.

        Parameters
        ----------
        column_names:
            List of raw column name strings.  Case is preserved in output but
            matching is case-insensitive.

        Returns
        -------
        TargetDetectionResult:
            Populated result; ``candidates`` is sorted by confidence
            descending.
        """
        if not column_names:
            return TargetDetectionResult()

        candidates: list[TargetCandidate] = []

        for col in column_names:
            if not isinstance(col, str) or not col.strip():
                continue

            lower = col.strip().lower()
            candidate = self._check_column(col, lower)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return TargetDetectionResult(
                candidates=[],
                best_candidate=None,
                detection_method="none",
            )

        # Sort by confidence descending; for ties, prefer shorter names
        candidates.sort(key=lambda c: (-c.confidence, len(c.column_name)))
        best = candidates[0]

        return TargetDetectionResult(
            candidates=candidates,
            best_candidate=best,
            detection_method="column_name_heuristic",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_column(
        self, original: str, lower: str
    ) -> Optional[TargetCandidate]:
        """Check a single column name and return a candidate or ``None``.

        Parameters
        ----------
        original:
            The original column name (casing preserved for output).
        lower:
            ``original.strip().lower()`` — pre-computed for efficiency.
        """
        # 1. Exact match
        if lower in _EXACT_TARGET_NAMES:
            return TargetCandidate(
                column_name=original,
                confidence=_CONFIDENCE_EXACT,
                reason=f"Exact match: '{lower}'",
            )

        # 2. Suffix match
        for suffix in _SUFFIX_SIGNALS:
            if lower.endswith(suffix) and len(lower) > len(suffix):
                return TargetCandidate(
                    column_name=original,
                    confidence=_CONFIDENCE_SUFFIX,
                    reason=f"Contains suffix '{suffix}': '{lower}'",
                )

        return None