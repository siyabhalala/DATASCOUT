"""
datascout.evaluation
────────────────────
Authoritative dataset evaluation and ranking system.

Primary entry point:
    from evaluation.pipeline import EvaluatorPipeline

All scoring is deterministic. LLM NEVER controls ranking.
"""

from .pipeline import (
    EvaluatorPipeline,
    EvaluationResult,
    EvaluationDiagnostics,
    DatasetDiagnostics,
)
from .scorer import DatasetScorer, ScoredDataset, ScoreBreakdown
from .ranker import RankingEngine, RankingResult
from .filter_engine import FilterEngine, FilterResult

__all__ = [
    "EvaluatorPipeline",
    "EvaluationResult",
    "EvaluationDiagnostics",
    "DatasetDiagnostics",
    "DatasetScorer",
    "ScoredDataset",
    "ScoreBreakdown",
    "RankingEngine",
    "RankingResult",
    "FilterEngine",
    "FilterResult",
]