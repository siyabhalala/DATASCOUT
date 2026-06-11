"""
datascout.evaluation.metrics
────────────────────────────
Individual deterministic evaluator signals.
Each module is independently testable and composable.

Usage:
    from evaluation.metrics.freshness import score_freshness
    from evaluation.metrics.popularity import score_popularity
    from evaluation.metrics.quality import score_quality
    from evaluation.metrics.task_match import score_task_match
"""

from .freshness import FreshnessResult, score_freshness
from .popularity import PopularityResult, score_popularity
from .quality import BiasSignal, QualityResult, score_quality
from .task_match import TaskMatchResult, score_task_match

__all__ = [
    "FreshnessResult", "score_freshness",
    "PopularityResult", "score_popularity",
    "BiasSignal", "QualityResult", "score_quality",
    "TaskMatchResult", "score_task_match",
]