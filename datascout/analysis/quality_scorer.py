"""
datascout.analysis.quality_scorer
───────────────────────────────────
Deterministic quality scoring for ``RawDataset`` objects.

``QualityScorer`` derives three orthogonal sub-scores from the metadata
already present on a ``RawDataset`` — completeness, consistency, and
validity — then combines them into a single composite score.  No external
APIs or file I/O are required; the scorer is intentionally deterministic and
fast (<1 ms per dataset).

Score ranges
────────────
All scores are floats in [0.0, 1.0].

Composite formula
─────────────────
    composite = completeness × 0.4 + consistency × 0.3 + validity × 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from datascout.contracts.models import RawDataset

logger = logging.getLogger(__name__)

__all__ = [
    "QualityScore",
    "QualityScorer",
]

# Composite weight constants — must sum to 1.0
_W_COMPLETENESS = 0.4
_W_CONSISTENCY = 0.3
_W_VALIDITY = 0.3


@dataclass
class QualityScore:
    """The result of scoring a single ``RawDataset`` for data quality.

    Attributes
    ----------
    completeness:
        Fraction of expected metadata fields that are populated.  Derived
        from :attr:`RawDataset.metadata_completeness` which the ingestion
        pipeline already computes.
    consistency:
        How internally consistent the metadata is.  Currently checks whether
        ``column_count`` matches ``len(column_names)`` when both are present.
    validity:
        How well the populated fields pass basic sanity checks: meaningful
        description length, at least one tag, license information present,
        schema (column names) present.
    composite:
        Weighted combination: completeness×0.4 + consistency×0.3 +
        validity×0.3.
    issues:
        Human-readable strings describing each quality problem found.  Empty
        list means no issues detected.
    computed_at:
        UTC timestamp at which this score was computed.
    """

    completeness: float
    consistency: float
    validity: float
    composite: float
    issues: List[str] = field(default_factory=list)
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def __post_init__(self) -> None:
        """Clamp all float scores into [0.0, 1.0] defensively."""
        self.completeness = max(0.0, min(1.0, self.completeness))
        self.consistency = max(0.0, min(1.0, self.consistency))
        self.validity = max(0.0, min(1.0, self.validity))
        self.composite = max(0.0, min(1.0, self.composite))


class QualityScorer:
    """Compute a ``QualityScore`` for any ``RawDataset``.

    The scorer is stateless; a single instance can be reused across the entire
    pipeline without thread-safety concerns.

    Example
    -------
    >>> scorer = QualityScorer()
    >>> result = scorer.score(dataset)
    >>> print(result.composite)   # e.g. 0.72
    >>> print(result.issues)      # ["License not specified"]
    """

    # Minimum description length (chars) to count as "has description"
    _MIN_DESCRIPTION_LENGTH: int = 20

    # Minimum number of tags to count as "has tags"
    _MIN_TAG_COUNT: int = 1

    def score(self, dataset: RawDataset) -> QualityScore:
        """Score *dataset* and return a :class:`QualityScore`.

        This method **never raises**.  Any unexpected exception is caught,
        logged as a warning, and a zero-score result is returned.

        Parameters
        ----------
        dataset:
            The dataset to score.

        Returns
        -------
        QualityScore:
            A fully populated score object.
        """
        try:
            return self._compute(dataset)
        except Exception as exc:
            logger.warning(
                "quality_scorer_unexpected_error",
                extra={
                    "canonical_id": getattr(dataset, "canonical_id", "unknown"),
                    "error": str(exc),
                },
            )
            return QualityScore(
                completeness=0.0,
                consistency=0.0,
                validity=0.0,
                composite=0.0,
                issues=["Quality scoring failed unexpectedly"],
            )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute(self, dataset: RawDataset) -> QualityScore:
        """Internal implementation — may be called with any RawDataset."""
        issues: list[str] = []

        completeness = self._score_completeness(dataset, issues)
        consistency = self._score_consistency(dataset, issues)
        validity = self._score_validity(dataset, issues)

        composite = (
            completeness * _W_COMPLETENESS
            + consistency * _W_CONSISTENCY
            + validity * _W_VALIDITY
        )

        logger.debug(
            "quality_score_computed",
            extra={
                "canonical_id": dataset.canonical_id,
                "completeness": round(completeness, 3),
                "consistency": round(consistency, 3),
                "validity": round(validity, 3),
                "composite": round(composite, 3),
                "issue_count": len(issues),
            },
        )

        return QualityScore(
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            composite=composite,
            issues=issues,
        )

    def _score_completeness(
        self, dataset: RawDataset, issues: list[str]
    ) -> float:
        """Derive completeness from ``metadata_completeness`` on the dataset.

        The ingestion pipeline already computes this value (0–1) by counting
        which of the expected fields are non-null.  We use it directly here so
        that the scorer stays in sync with whatever definition the ingestion
        layer uses.
        """
        score: float = getattr(dataset, "metadata_completeness", 0.0) or 0.0
        score = max(0.0, min(1.0, float(score)))
        if score < 0.5:
            issues.append(
                f"Metadata completeness is low ({round(score * 100)}%): "
                "many expected fields are missing"
            )
        return score

    def _score_consistency(
        self, dataset: RawDataset, issues: list[str]
    ) -> float:
        """Check internal consistency of the metadata.

        Currently verifies that ``column_count`` matches ``len(column_names)``
        when both are present.  Returns 1.0 if neither or only one is present
        (no contradiction possible).
        """
        column_count: int | None = getattr(dataset, "column_count", None)
        column_names: list | None = getattr(dataset, "column_names", None)

        if column_count is not None and column_names is not None:
            declared = int(column_count)
            actual = len(column_names)
            if declared != actual:
                issues.append(
                    f"Column count mismatch: declared {declared} "
                    f"but column_names has {actual} entries"
                )
                # Partial score proportional to agreement
                agreement = 1.0 - abs(declared - actual) / max(declared, actual, 1)
                return max(0.0, agreement)
            return 1.0

        # No contradiction possible — can't score negatively
        return 1.0

    def _score_validity(
        self, dataset: RawDataset, issues: list[str]
    ) -> float:
        """Score validity based on binary checks (equal weight per check).

        Checks (all datasets)
        ---------------------
        1. Has a meaningful description (≥ _MIN_DESCRIPTION_LENGTH chars).
        2. Has at least one tag.
        3. License information is present.

        Tabular-only check
        ------------------
        4. Schema information (column names) present.
           SKIPPED for image datasets — image datasets are organised as
           folders-per-class, not tabular columns. Flagging missing column
           schema on an image dataset is a false positive that misleads users.
        """
        # Detect image dataset defensively — Modality enum repr varies by version
        modalities = getattr(dataset, "modalities", None) or []
        is_image = any(
            str(m).lower() in ("image", "modality.image", "<modality.image: 'image'>")
            for m in modalities
        )

        checks_passed = 0
        total_checks = 3 if is_image else 4   # image datasets: 3 checks, not 4

        # 1. Description
        description: str = getattr(dataset, "description", "") or ""
        if len(description.strip()) >= self._MIN_DESCRIPTION_LENGTH:
            checks_passed += 1
        else:
            issues.append(
                "No meaningful description provided"
                if not description.strip()
                else "Description is too short to be useful"
            )

        # 2. Tags
        tags: list = getattr(dataset, "tags", []) or []
        if len(tags) >= self._MIN_TAG_COUNT:
            checks_passed += 1
        else:
            issues.append("No tags provided — discovery will be impaired")

        # 3. License
        license_type = getattr(dataset, "license_type", None)
        if license_type is not None:
            checks_passed += 1
        else:
            issues.append("License not specified — legal reuse status unknown")

        # 4. Schema — tabular datasets only. Image datasets have folder-per-class
        # structure, not column schemas. Skip this check for image datasets.
        if not is_image:
            column_names: list | None = getattr(dataset, "column_names", None)
            if column_names:
                checks_passed += 1
            else:
                issues.append(
                    "No column schema provided — structure is unknown"
                )

        return checks_passed / total_checks