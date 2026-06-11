"""
datascout.analysis.summary_generator
──────────────────────────────────────
Deterministic human-readable summary generation for ``RawDataset`` objects.

``SummaryGenerator`` produces a ``DatasetSummary`` from the metadata already
present on a ``RawDataset``.  No LLM or external API is required; every field
is computed deterministically so the output is stable across runs and safe to
cache.

The one-liner, size description, and freshness label are all built using
simple string formatting rules documented on each method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from datascout.contracts.models import RawDataset

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetSummary",
    "SummaryGenerator",
]

# Maximum characters in one_liner
_MAX_ONE_LINER_CHARS = 120

# Maximum tags to surface in top_tags
_MAX_TOP_TAGS = 5

# Boundary (days) for freshness labels
_FRESHNESS_VERY_RECENT_DAYS = 30
_FRESHNESS_RECENT_DAYS = 90
_FRESHNESS_MODERATE_DAYS = 365
_FRESHNESS_OLD_DAYS = 730


@dataclass
class DatasetSummary:
    """A concise, human-readable summary of a single dataset.

    All string fields are non-None and non-empty where the underlying data
    permits.  Fields that cannot be computed fall back to a descriptive
    placeholder (e.g. ``"Size unknown"``).

    Attributes
    ----------
    canonical_id:
        Matches :attr:`RawDataset.canonical_id` — e.g. ``"kaggle:titanic"``.
    title:
        Dataset title, taken directly from :attr:`RawDataset.title`.
    one_liner:
        At most ``_MAX_ONE_LINER_CHARS`` chars.  Built from the first 80 chars
        of the description if available; otherwise ``"{title} — {top 3 tags}"``.
    size_description:
        E.g. ``"45,000 rows × 12 columns"`` or ``"Size unknown"``.
    task_labels:
        Human-readable task type names derived from ``task_types``.
    top_tags:
        First ``_MAX_TOP_TAGS`` entries from ``tags``.
    freshness_label:
        E.g. ``"Updated 3 months ago"`` or ``"Last update unknown"``.
    generated_at:
        UTC timestamp when this summary was generated.
    """

    canonical_id: str
    title: str
    one_liner: str
    size_description: str
    task_labels: List[str]
    top_tags: List[str]
    freshness_label: str
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


class SummaryGenerator:
    """Generate a :class:`DatasetSummary` for any ``RawDataset``.

    The generator is stateless and re-entrant; one instance can be reused
    across the entire pipeline.

    Example
    -------
    >>> gen = SummaryGenerator()
    >>> summary = gen.generate(dataset)
    >>> print(summary.one_liner)
    >>> print(summary.freshness_label)
    """

    def generate(self, dataset: RawDataset) -> DatasetSummary:
        """Generate a :class:`DatasetSummary` from *dataset*.

        This method **never raises**.  Any unexpected exception is caught,
        logged, and a minimal (all-placeholder) summary is returned.

        Parameters
        ----------
        dataset:
            The dataset to summarise.

        Returns
        -------
        DatasetSummary:
            Fully populated summary object.
        """
        try:
            return self._build(dataset)
        except Exception as exc:
            logger.warning(
                "summary_generator_unexpected_error",
                extra={
                    "canonical_id": getattr(dataset, "canonical_id", "unknown"),
                    "error": str(exc),
                },
            )
            return DatasetSummary(
                canonical_id=getattr(dataset, "canonical_id", "unknown"),
                title=getattr(dataset, "title", "Unknown Dataset"),
                one_liner="Summary generation failed.",
                size_description="Size unknown",
                task_labels=[],
                top_tags=[],
                freshness_label="Last update unknown",
            )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, dataset: RawDataset) -> DatasetSummary:
        """Core build logic — called inside try/except by :meth:`generate`."""
        canonical_id: str = dataset.canonical_id
        title: str = dataset.title or "Untitled Dataset"

        one_liner = self._build_one_liner(dataset, title)
        size_description = self._build_size_description(dataset)
        task_labels = self._build_task_labels(dataset)
        top_tags = self._build_top_tags(dataset)
        freshness_label = self._build_freshness_label(dataset)

        return DatasetSummary(
            canonical_id=canonical_id,
            title=title,
            one_liner=one_liner,
            size_description=size_description,
            task_labels=task_labels,
            top_tags=top_tags,
            freshness_label=freshness_label,
        )

    def _build_one_liner(self, dataset: RawDataset, title: str) -> str:
        """Build a one-liner of at most ``_MAX_ONE_LINER_CHARS`` characters.

        Strategy
        --------
        1. If ``description`` has ≥ 20 meaningful characters, take the first
           80 characters (stripped) and truncate at the last complete word
           to avoid mid-word cuts.
        2. Otherwise fall back to ``"{title} — {top 3 tags joined by ', '}"``.
        3. If no tags either, use title alone.
        """
        description: str = (getattr(dataset, "description", "") or "").strip()
        if len(description) >= 20:
            raw = description[:80]
            # Snap to last whitespace boundary to avoid mid-word cuts
            if len(description) > 80 and " " in raw:
                raw = raw.rsplit(" ", 1)[0]
            one_liner = raw.rstrip(".,; ")
        else:
            tags: list = getattr(dataset, "tags", []) or []
            top3 = [str(t) for t in tags[:3]]
            if top3:
                one_liner = f"{title} — {', '.join(top3)}"
            else:
                one_liner = title

        # Final hard cap
        if len(one_liner) > _MAX_ONE_LINER_CHARS:
            one_liner = one_liner[: _MAX_ONE_LINER_CHARS - 1].rstrip() + "…"

        return one_liner

    def _build_size_description(self, dataset: RawDataset) -> str:
        """Format row and column counts with thousands separators.

        Returns
        -------
        str:
            Examples: ``"45,000 rows × 12 columns"``,
            ``"45,000 rows"``, ``"12 columns"``, ``"Size unknown"``.
        """
        row_count: Optional[int] = getattr(dataset, "row_count", None)
        column_count: Optional[int] = getattr(dataset, "column_count", None)

        parts: list[str] = []
        if row_count is not None and row_count > 0:
            parts.append(f"{row_count:,} rows")
        if column_count is not None and column_count > 0:
            parts.append(f"{column_count:,} columns")

        if not parts:
            return "Size unknown"
        return " × ".join(parts)

    def _build_task_labels(self, dataset: RawDataset) -> list[str]:
        """Convert ``task_types`` enum values to human-readable labels.

        Falls back to ``str(task_type)`` if the enum value has no ``value``
        attribute (forward-compatibility guard).
        """
        task_types = getattr(dataset, "task_types", []) or []
        labels: list[str] = []
        for tt in task_types:
            # Prefer .value (enum string), then .name, then str()
            if hasattr(tt, "value"):
                labels.append(str(tt.value).replace("_", " ").title())
            elif hasattr(tt, "name"):
                labels.append(str(tt.name).replace("_", " ").title())
            else:
                labels.append(str(tt).replace("_", " ").title())
        return labels

    def _build_top_tags(self, dataset: RawDataset) -> list[str]:
        """Return the first ``_MAX_TOP_TAGS`` tags as strings."""
        tags = getattr(dataset, "tags", []) or []
        return [str(t) for t in tags[:_MAX_TOP_TAGS]]

    def _build_freshness_label(self, dataset: RawDataset) -> str:
        """Compute a human-friendly freshness label from ``last_updated``.

        Computes the number of days between ``last_updated`` (UTC) and now,
        then maps to a friendly string.

        Returns
        -------
        str:
            Examples: ``"Updated 2 weeks ago"``, ``"Updated 3 months ago"``,
            ``"Updated over 2 years ago"``, ``"Last update unknown"``.
        """
        last_updated: Optional[datetime] = getattr(dataset, "last_updated", None)
        if last_updated is None:
            return "Last update unknown"

        now = datetime.now(tz=timezone.utc)
        # Ensure both are timezone-aware for comparison
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        try:
            delta = now - last_updated
        except Exception:
            return "Last update unknown"

        days = delta.days

        if days < 0:
            # Future date — treat as very recent
            return "Updated recently"
        if days == 0:
            return "Updated today"
        if days == 1:
            return "Updated yesterday"
        if days <= _FRESHNESS_VERY_RECENT_DAYS:
            weeks = days // 7
            if weeks <= 1:
                return f"Updated {days} days ago"
            return f"Updated {weeks} weeks ago"
        if days <= _FRESHNESS_RECENT_DAYS:
            months = days // 30
            return f"Updated {months} month{'s' if months != 1 else ''} ago"
        if days <= _FRESHNESS_MODERATE_DAYS:
            months = days // 30
            return f"Updated {months} months ago"
        if days <= _FRESHNESS_OLD_DAYS:
            return "Updated over a year ago"
        years = days // 365
        return f"Updated over {years} year{'s' if years != 1 else ''} ago"