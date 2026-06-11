"""
datascout.evaluation.ranker
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Converts scored datasets into a final ranked
RankingResult with top-K selection, diversity boosting, confidence
calculation, and DecisionTrace for explainability.

SYSTEM DESIGN DECISIONS:

  1. WHY diversity boost (not pure score ranking)?
     - Without diversity: user gets top-5 results all from Kaggle, all
       tabular, all similar — poor exploration value.
     - With diversity: results span multiple sources and domains.
     - Implementation: greedy source-count penalty (O(n), not NP-hard).
     - diversity_penalty^n applied to Nth dataset from same source.

  2. WHY confidence from score gap?
     - top=0.80, second=0.79 → gap=0.01 → LOW (effectively tied).
     - top=0.80, second=0.55 → gap=0.25 → HIGH (clear winner).
     - Absolute score alone is misleading without knowing the competition.

  3. WHY DecisionTrace stored only for top trace_top_k (default 20)?
     - Trace for all 1000+ candidates = MB of data per query.
     - Rank 21+ traces are never read in practice.
     - lineage_ref allows reconstruction from logs if needed.

  4. WHY build_agent_response() lives in RankingEngine?
     - RankingEngine has scored datasets + query context.
     - Centralizes EvaluatedDataset + RankedResult construction.
     - Agent controller calls: response = engine.build_agent_response(...)
     - Single conversion path — no schema drift between ranking and response.

FAILURE SCENARIOS HANDLED:
  - Empty scored list → empty RankingResult, confidence=LOW
  - top_k > available → return all available (no error)
  - All same source → diversity runs without error (just no benefit)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from datascout.contracts.models import EvaluatedDataset
from datascout.contracts.requests import SearchQuery
from datascout.contracts.responses import (
    AgentResponse,
    ConfidenceLevel,
    DecisionTrace,
    PipelineStage,
    RankedResult,
    compute_confidence,
)
from datascout.contracts.states import QualityTier, compute_quality_tier
from datascout.evaluation.scorer import ScoredDataset

logger = logging.getLogger("datascout.evaluation.ranker")

DEFAULT_TOP_K            = 10
DIVERSITY_PENALTY_FACTOR = 0.70   # FIX v3.5.0: raised from 0.85 (15%) → 0.70 (30%).
# WHY: Kaggle datasets typically have 10x–100x more downloads than HuggingFace
# equivalents. The popularity metric contributes ~20% of composite_score.
# A 15% penalty is too weak — a Kaggle dataset with 50k downloads easily
# outscores a HF dataset with 500 downloads even after penalisation, causing
# all top-10 slots to be filled by Kaggle results. 30% creates a fair contest.
TRACE_TOP_K              = 20     # Build DecisionTrace for top N only


@dataclass
class RankingResult:
    """Final output of the recommendation pipeline."""
    ranked:            list[ScoredDataset]
    total_candidates:  int
    confidence:        ConfidenceLevel
    diversity_applied: bool
    top_k:             int

    @property
    def top_dataset(self) -> Optional[ScoredDataset]:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict:
        return {
            "total_candidates":  self.total_candidates,
            "returned":          len(self.ranked),
            "top_k":             self.top_k,
            "confidence":        self.confidence.value,
            "diversity_applied": self.diversity_applied,
            "results": [r.to_dict() for r in self.ranked],
        }


class RankingEngine:
    """
    Converts scored datasets into a final top-K ranking.

    Pipeline:
      1. Apply diversity boost (penalize same-source clustering)
      2. Re-sort by adjusted score
      3. Select top-K
      4. Compute confidence from score gap
      5. Return RankingResult
    """

    def __init__(
        self,
        top_k:             int   = DEFAULT_TOP_K,
        diversity_boost:   bool  = True,
        diversity_penalty: float = DIVERSITY_PENALTY_FACTOR,
        trace_top_k:       int   = TRACE_TOP_K,
    ) -> None:
        self.top_k             = top_k
        self.diversity_boost   = diversity_boost
        self.diversity_penalty = diversity_penalty
        self.trace_top_k       = trace_top_k

    def rank(self, scored: list[ScoredDataset]) -> RankingResult:
        """Produce final ranking. Never raises."""
        if not scored:
            return RankingResult(
                ranked=[], total_candidates=0,
                confidence=ConfidenceLevel.LOW,
                diversity_applied=False, top_k=self.top_k,
            )

        total = len(scored)

        # Apply diversity boost
        if self.diversity_boost and len(scored) > self.top_k:
            adjusted = self._apply_diversity(scored)
            diversity_applied = True
        else:
            adjusted = [(s, s.composite_score) for s in scored]
            diversity_applied = False

        # Sort by adjusted score
        adjusted.sort(key=lambda x: x[1], reverse=True)

        # Take top-K
        top = [item[0] for item in adjusted[: self.top_k]]

        # Confidence from score gap
        scores = [item[1] for item in adjusted]
        confidence = compute_confidence(scores)

        logger.info(
            "ranking_complete",
            extra={
                "total_candidates": total,
                "returned":         len(top),
                "top_score":        round(scores[0], 4) if scores else 0.0,
                "confidence":       confidence.value,
                "diversity":        diversity_applied,
            },
        )

        return RankingResult(
            ranked=top,
            total_candidates=total,
            confidence=confidence,
            diversity_applied=diversity_applied,
            top_k=self.top_k,
        )

    def build_agent_response(
        self,
        query: SearchQuery,
        ranking: RankingResult,
        pipeline_stages: Optional[list[PipelineStage]] = None,
        processing_time_ms: int = 0,
    ) -> AgentResponse:
        """
        Convert RankingResult → AgentResponse.
        Builds EvaluatedDataset + RankedResult + DecisionTrace per entry.
        """
        ranked_results: list[RankedResult] = []

        for rank_idx, scored in enumerate(ranking.ranked):
            rank = rank_idx + 1
            ds   = scored.dataset
            bd   = scored.breakdown

            evaluated = EvaluatedDataset(
                raw=ds,
                composite_score=bd.composite,
                rank=rank,
                quality_tier=compute_quality_tier(bd.composite),
            )

            trace: Optional[DecisionTrace] = None
            if rank <= self.trace_top_k:
                trace = DecisionTrace(
                    dataset_id=ds.canonical_id,
                    dataset_title=ds.title,
                    final_score=bd.composite,
                    final_rank=rank,
                    score_breakdown={
                        "task_relevance":    bd.task_relevance,
                        "quality":           bd.quality,
                        "popularity":        bd.popularity,
                        "freshness":         bd.freshness,
                        "description_match": bd.description_match,
                    },
                    filters_passed=True,
                    filter_reasons=[],
                    lineage_ref=(
                        f"{ds.ingestion_timestamp.isoformat()}::{ds.pipeline_version}"
                    ),
                    trace_version="3.0.0",
                )

            ranked_results.append(
                RankedResult(
                    dataset=evaluated,
                    rank=rank,
                    trace=trace,
                    download_url=ds.source_url,
                )
            )

        return AgentResponse(
            query_id=query.query_id,
            results=ranked_results,
            total_found=ranking.total_candidates,
            confidence=ranking.confidence,
            partial_result=False,
            pipeline_stages=pipeline_stages or [],
            processing_time_ms=processing_time_ms,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DIVERSITY
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_diversity(
        self, scored: list[ScoredDataset]
    ) -> list[tuple[ScoredDataset, float]]:
        """
        Greedy diversity: penalize scores for datasets from already-seen sources.
        diversity_penalty^n applied to the Nth dataset from a source.
        O(n) — iterates sorted list once.
        """
        source_counts: dict[str, int] = {}
        result: list[tuple[ScoredDataset, float]] = []

        for s in scored:
            src = s.dataset.source
            count = source_counts.get(src, 0)
            adjusted = s.composite_score * (self.diversity_penalty ** count)
            result.append((s, adjusted))
            source_counts[src] = count + 1

        return result