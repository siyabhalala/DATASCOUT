"""
datascout.evaluation.pipeline
────────────────────────────────────────────────────────────
AUTHORITATIVE DATASET INTELLIGENCE ENGINE — Phase 3 Core

This is the single entry point for all dataset evaluation.
It consolidates the scorer, ranker, filter engine, and the four
metric modules into ONE reproducible pipeline.

ARCHITECTURAL RULE (enforced here):
  LLM NEVER controls ranking.
  All scoring is deterministic math. This module produces rankings.
  The LLM explanation layer reads these rankings AFTER the fact.

Pipeline flow:
  datasets + query
    → FilterEngine (hard constraints)
    → DatasetScorer (5-dimension composite score)
    → metric modules (freshness, popularity, quality, task_match)
    → RankingEngine (diversity boost + confidence)
    → EvaluationResult (ranked + explained + observable)

Usage:
    pipeline = EvaluatorPipeline(query_parse_result)
    result = pipeline.evaluate(datasets, filters)
    # result.ranked: sorted ScoredDataset list
    # result.diagnostics: full evaluator transparency
    # result.bias_report: per-dataset bias signals
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from datascout.contracts import RawDataset
from datascout.contracts.requests import SearchFilters
from datascout.contracts.task_types import Modality, TaskType

from datascout.evaluation.filter_engine import FilterEngine, FilterResult
from datascout.evaluation.ranker import RankingEngine, RankingResult
from datascout.evaluation.scorer import DatasetScorer, ScoredDataset, ScoreBreakdown
from datascout.evaluation.metrics.freshness import FreshnessResult, score_freshness
from datascout.evaluation.metrics.popularity import PopularityResult, score_popularity
from datascout.evaluation.metrics.quality import BiasSignal, QualityResult, score_quality
from datascout.evaluation.metrics.task_match import TaskMatchResult, score_task_match

logger = logging.getLogger("datascout.evaluation.pipeline")


@dataclass
class DatasetDiagnostics:
    """Full evaluator transparency for one dataset."""
    canonical_id: str
    title: str
    composite_score: float
    rank: Optional[int]

    # Per-dimension breakdowns
    freshness: FreshnessResult
    popularity: PopularityResult
    quality: QualityResult
    task_match: TaskMatchResult

    # Score breakdown from composite scorer
    score_breakdown: dict

    # Ranking inputs
    diversity_adjusted: bool = False

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "title": self.title,
            "composite_score": self.composite_score,
            "rank": self.rank,
            "score_breakdown": self.score_breakdown,
            "freshness": {
                "score": self.freshness.score,
                "days_since_update": self.freshness.days_since_update,
                "explanation": self.freshness.explanation,
            },
            "popularity": {
                "score": self.popularity.score,
                "download_count": self.popularity.download_count,
                "upvote_count": self.popularity.upvote_count,
                "explanation": self.popularity.explanation,
            },
            "quality": {
                "score": self.quality.score,
                "completeness_score": self.quality.completeness_score,
                "bias_signals": [
                    {"type": b.signal_type, "severity": b.severity, "description": b.description}
                    for b in self.quality.bias_signals
                ],
                "missing_fields": self.quality.missing_fields,
                "explanation": self.quality.explanation,
            },
            "task_match": {
                "score": self.task_match.score,
                "match_type": self.task_match.match_type,
                "explanation": self.task_match.explanation,
            },
        }


@dataclass
class EvaluationDiagnostics:
    """Pipeline-level observability."""
    total_input: int
    total_after_filter: int
    total_scored: int
    total_ranked: int
    filter_pass_rate: float
    elapsed_ms: float
    filters_applied: list[str]
    confidence: str
    diversity_applied: bool
    query_task: Optional[str]
    query_modality: Optional[str]

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "total_after_filter": self.total_after_filter,
            "total_scored": self.total_scored,
            "total_ranked": self.total_ranked,
            "filter_pass_rate": round(self.filter_pass_rate, 3),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "filters_applied": self.filters_applied,
            "confidence": self.confidence,
            "diversity_applied": self.diversity_applied,
            "query_task": self.query_task,
            "query_modality": self.query_modality,
        }


@dataclass
class EvaluationResult:
    """Full output of the evaluator pipeline."""
    ranked: list[ScoredDataset]
    ranking_result: RankingResult
    filter_result: FilterResult
    dataset_diagnostics: list[DatasetDiagnostics]
    pipeline_diagnostics: EvaluationDiagnostics

    @property
    def diagnostics(self) -> "list[DatasetDiagnostics]":
        """Alias for ``dataset_diagnostics``.

        ``search_v2._run_pipeline()`` reads diagnostics via::

            for diag in getattr(eval_result, "diagnostics", []) or []: ...

        This property makes that access work without changing search_v2.
        """
        return self.dataset_diagnostics

    @property
    def top_dataset(self) -> Optional[ScoredDataset]:
        return self.ranked[0] if self.ranked else None

    @property
    def total_found(self) -> int:
        return self.ranking_result.total_candidates

    def get_bias_report(self) -> list[dict]:
        """Return all datasets with detected bias signals."""
        report = []
        for diag in self.dataset_diagnostics:
            if diag.quality.bias_signals:
                report.append({
                    "canonical_id": diag.canonical_id,
                    "title": diag.title,
                    "rank": diag.rank,
                    "bias_signals": [
                        {"type": b.signal_type, "severity": b.severity, "description": b.description}
                        for b in diag.quality.bias_signals
                    ],
                })
        return report

    def to_dict(self) -> dict:
        return {
            "ranked": [r.to_dict() for r in self.ranked],
            "pipeline": self.pipeline_diagnostics.to_dict(),
            "dataset_diagnostics": [d.to_dict() for d in self.dataset_diagnostics],
            "bias_report": self.get_bias_report(),
            "filter_summary": self.filter_result.to_dict(),
            "ranking_summary": self.ranking_result.to_dict(),
        }


class EvaluatorPipeline:
    """
    The authoritative dataset intelligence engine.

    Deterministic. Observable. Composable.
    Same inputs always produce same outputs — no hidden state.

    Args:
        query_task:     Detected ML task type from query understanding
        query_modality: Detected data modality from query understanding
        keywords:       Extracted search keywords for description matching
        top_k:          Maximum results to return
        diversity_boost: Enable source diversity in ranking
    """

    def __init__(
        self,
        query_task: Optional[TaskType] = None,
        query_modality: Optional[Modality] = None,
        keywords: Optional[list[str]] = None,
        top_k: int = 10,
        diversity_boost: bool = True,
        strict_task_matching: bool = False,
    ) -> None:
        self._query_task = query_task
        self._query_modality = query_modality
        self._keywords = keywords or []
        self._top_k = top_k

        self._filter_engine = FilterEngine(strict_task_matching=strict_task_matching)
        self._scorer = DatasetScorer(
            query_task=query_task,
            query_modality=query_modality,
            keywords=keywords,
        )
        self._ranker = RankingEngine(
            top_k=top_k,
            diversity_boost=diversity_boost,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Async entry point — used by search_v2._run_pipeline()
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        datasets: list,
        query: Optional[str] = None,
        enrichment: Optional[object] = None,
    ) -> "EvaluationResult":
        """
        Async adapter for ``evaluate()``.

        ``search_v2._run_pipeline()`` calls this as::

            pipeline = EvaluatorPipeline()
            eval_result = await pipeline.run(
                datasets=raw_datasets,
                query=expanded_query,
                enrichment=enrichment,
            )

        Since ``evaluate()`` is purely CPU-bound (no I/O), this method
        extracts task/modality/keywords from *enrichment* to configure a
        properly-tuned evaluator, then runs ``evaluate()`` in a thread-pool
        executor so the event loop is never blocked.

        Parameters
        ----------
        datasets:
            List of ``RawDataset`` objects to evaluate.
        query:
            Expanded query string (used only for logging).
        enrichment:
            ``QueryPipelineResult`` from ``QueryUnderstandingPipeline``
            (or any object with ``task_type``, ``modality``, ``keywords``
            attributes).  May be None — scoring still runs, just without
            task-relevance boosts.

        Returns
        -------
        EvaluationResult:
            Ranked, scored, and diagnosed results.  Never raises.
        """
        import asyncio as _asyncio

        # ── Extract enrichment fields ──────────────────────────────────────
        task_type: Optional[TaskType] = None
        modality: Optional[Modality] = None
        keywords: list[str] = self._keywords  # already set via __init__ if any

        if enrichment is not None:
            task_type = getattr(enrichment, "task_type", None) or task_type
            modality = getattr(enrichment, "modality", None) or modality
            kw = getattr(enrichment, "keywords", None)
            if kw:
                keywords = list(kw)

        # ── Re-configure if enrichment provided better signals ─────────────
        # Rebuild whenever task_type, modality, OR keywords changed.
        # Critical: keywords MUST flow into the scorer for query_match scoring.
        # Previously only rebuilt on task/modality change — keywords were silently
        # dropped when both were None, causing Query match: 0% on every result.
        enrichment_keywords_differ = set(keywords) != set(self._keywords)
        if (
            task_type is not self._query_task
            or modality is not self._query_modality
            or enrichment_keywords_differ
        ):
            configured = EvaluatorPipeline(
                query_task=task_type,
                query_modality=modality,
                keywords=keywords,
                top_k=self._top_k,
            )
        else:
            configured = self

        logger.debug(
            "evaluator_pipeline_run",
            extra={
                "datasets": len(datasets),
                "query": (query or "")[:60],
                "task": task_type.value if task_type else "none",
                "modality": modality.value if modality else "none",
            },
        )

        # ── Run synchronously in executor ──────────────────────────────────
        try:
            loop = _asyncio.get_event_loop()
            result: EvaluationResult = await loop.run_in_executor(
                None, lambda: configured.evaluate(datasets)
            )
            return result
        except Exception as exc:
            logger.error(
                "evaluator_pipeline_run_failed",
                extra={"error": str(exc)[:200]},
                exc_info=True,
            )
            # Return empty-but-valid result so the caller degrades gracefully
            from datascout.contracts.responses import ConfidenceLevel
            from datascout.evaluation.ranker import RankingResult
            from datascout.evaluation.filter_engine import FilterResult
            empty_ranking = RankingResult(
                ranked=[], total_candidates=0,
                confidence=ConfidenceLevel.LOW,
                diversity_applied=False,
                top_k=self._top_k,
            )
            empty_filter = FilterResult(
                passed=[], rejected=[], filters_applied=[], pass_rate=0.0
            )
            return EvaluationResult(
                ranked=[],
                ranking_result=empty_ranking,
                filter_result=empty_filter,
                dataset_diagnostics=[],
                pipeline_diagnostics=EvaluationDiagnostics(
                    total_input=len(datasets),
                    total_after_filter=0,
                    total_scored=0,
                    total_ranked=0,
                    filter_pass_rate=0.0,
                    elapsed_ms=0.0,
                    filters_applied=[],
                    confidence="low",
                    diversity_applied=False,
                    query_task=task_type.value if task_type else None,
                    query_modality=modality.value if modality else None,
                ),
            )

    def evaluate(
        self,
        datasets: list[RawDataset],
        filters: Optional[SearchFilters] = None,
        exclude_duplicates: bool = True,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline.

        Flow:
          1. Filter (hard constraints)
          2. Score (5-dimension composite)
          3. Enrich with individual metric modules
          4. Rank (diversity + confidence)
          5. Build diagnostics

        Never raises. Returns empty EvaluationResult on total failure.
        """
        t0 = time.monotonic()
        filters = filters or SearchFilters()

        # ── Step 1: Filter ────────────────────────────────────────────
        filter_result = self._filter_engine.apply(
            datasets=datasets,
            filters=filters,
            query_task=self._query_task,
            query_modality=self._query_modality,
            exclude_duplicates=exclude_duplicates,
        )

        if not filter_result.passed:
            logger.warning(
                "evaluator_all_filtered",
                extra={"input": len(datasets), "filters": filter_result.filters_applied},
            )
            elapsed = (time.monotonic() - t0) * 1000
            empty_diag = EvaluationDiagnostics(
                total_input=len(datasets),
                total_after_filter=0,
                total_scored=0,
                total_ranked=0,
                filter_pass_rate=0.0,
                elapsed_ms=elapsed,
                filters_applied=filter_result.filters_applied,
                confidence="low",
                diversity_applied=False,
                query_task=self._query_task.value if self._query_task else None,
                query_modality=self._query_modality.value if self._query_modality else None,
            )
            from datascout.contracts.responses import ConfidenceLevel
            from .ranker import RankingResult
            empty_ranking = RankingResult(
                ranked=[], total_candidates=0,
                confidence=ConfidenceLevel.LOW,
                diversity_applied=False, top_k=self._top_k,
            )
            return EvaluationResult(
                ranked=[],
                ranking_result=empty_ranking,
                filter_result=filter_result,
                dataset_diagnostics=[],
                pipeline_diagnostics=empty_diag,
            )

        # ── Step 2: Score ─────────────────────────────────────────────
        scored = self._scorer.score_all(filter_result.passed)

        # ── Step 3: Enrich with individual metric diagnostics ─────────
        diagnostics = self._build_diagnostics(scored)

        # ── Step 4: Rank ──────────────────────────────────────────────
        ranking_result = self._ranker.rank(scored)

        # Assign ranks in diagnostics
        rank_map = {s.dataset.canonical_id: i + 1 for i, s in enumerate(ranking_result.ranked)}
        for diag in diagnostics:
            diag.rank = rank_map.get(diag.canonical_id)

        elapsed = (time.monotonic() - t0) * 1000

        # ── Step 5: Pipeline diagnostics ──────────────────────────────
        pipeline_diag = EvaluationDiagnostics(
            total_input=len(datasets),
            total_after_filter=len(filter_result.passed),
            total_scored=len(scored),
            total_ranked=len(ranking_result.ranked),
            filter_pass_rate=filter_result.pass_rate,
            elapsed_ms=elapsed,
            filters_applied=filter_result.filters_applied,
            confidence=ranking_result.confidence.value,
            diversity_applied=ranking_result.diversity_applied,
            query_task=self._query_task.value if self._query_task else None,
            query_modality=self._query_modality.value if self._query_modality else None,
        )

        logger.info(
            "evaluator_pipeline_complete",
            extra={
                "input": len(datasets),
                "filtered": filter_result.pass_count,
                "scored": len(scored),
                "ranked": len(ranking_result.ranked),
                "confidence": ranking_result.confidence.value,
                "elapsed_ms": round(elapsed, 1),
                "top_score": round(ranking_result.ranked[0].composite_score, 4)
                             if ranking_result.ranked else 0.0,
            },
        )

        return EvaluationResult(
            ranked=ranking_result.ranked,
            ranking_result=ranking_result,
            filter_result=filter_result,
            dataset_diagnostics=diagnostics,
            pipeline_diagnostics=pipeline_diag,
        )

    def _build_diagnostics(self, scored: list[ScoredDataset]) -> list[DatasetDiagnostics]:
        """
        Run individual metric modules for each scored dataset.
        Failures in individual modules are isolated — never propagate.
        """
        diagnostics: list[DatasetDiagnostics] = []

        for s in scored:
            try:
                ds = s.dataset

                freshness_result = score_freshness(ds)
                popularity_result = score_popularity(ds)
                quality_result = score_quality(ds)
                task_match_result = score_task_match(
                    dataset=ds,
                    query_task=self._query_task,
                    query_modality=self._query_modality,
                )

                diag = DatasetDiagnostics(
                    canonical_id=ds.canonical_id,
                    title=ds.title,
                    composite_score=s.composite_score,
                    rank=None,  # Filled in after ranking
                    freshness=freshness_result,
                    popularity=popularity_result,
                    quality=quality_result,
                    task_match=task_match_result,
                    score_breakdown=s.breakdown.to_dict(),
                )
                diagnostics.append(diag)

            except Exception as exc:
                logger.warning(
                    "evaluator_diagnostics_failed",
                    extra={
                        "canonical_id": s.dataset.canonical_id,
                        "error": str(exc)[:80],
                    },
                )
                # Add minimal diagnostics so the dataset still surfaces
                continue

        return diagnostics