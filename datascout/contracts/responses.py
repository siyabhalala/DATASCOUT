"""
datascout.contracts.responses
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Output contracts — structured response types for
everything the system returns to users, agents, and monitoring.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY StructuredExplanation over raw LLM string output?
     - Raw strings cannot be unit tested — fields can
     - UI renders summary bold, weaknesses red, recommendations as buttons
     - With raw string: parse it (fragile) or render all the same (ugly)
     - i18n: translate per field, not a blob of mixed text
     - A/B testing: swap explanation strategy without changing UI contract

  2. WHY score-gap confidence algorithm over fixed thresholds?
     - Fixed threshold: composite_score >= 0.8 = HIGH confidence
     - Problem: score=0.95 vs score=0.94 = effectively tied → should be LOW
     - Problem: score=0.72 vs score=0.51 = clear winner → should be HIGH
     - Score-gap captures the relative separation, not the absolute value

  3. WHY DecisionTrace for top 20 only?
     - Storing full trace for all 1M records = ~2GB per query
     - Users only ever see top results — traces for ranks 21+ are never read
     - Stored with lineage_ref → can always retrieve full record if needed

  4. WHY LLMMetadata on every response?
     - Cost tracking: know exactly what each request costs in USD
     - Budget alerts: cost_usd > threshold → PagerDuty before runaway spend
     - Fallback tracking: how often is primary model failing?
     - Compliance: EU AI Act requires disclosure of AI involvement

  5. WHY PipelineStage tracking?
     - SLA monitoring: know exactly where latency is spent
     - Debugging: "which stage failed for query X?"
     - Alerting: stage duration_ms > threshold → alert

FAILURE SCENARIOS HANDLED:
  - Empty results (all adapters failed) → AgentResponse with empty list + error_message
  - Partial results (some adapters failed) → partial_result=True + adapter_errors logged
  - LLM explanation failed → explanation=None + scores still returned
  - Score computation tie → ConfidenceLevel.LOW + alternatives highlighted

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add alternative_queries field to AgentResponse
  Breaking: v4.0.0 — remove legacy_explanation field

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from .errors import ValidationError
from .models import EvaluatedDataset
from .states import StageStatus

logger = logging.getLogger("datascout.contracts.responses")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE LEVEL
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """
    Recommendation confidence level — computed from score gap, not absolute score.
    HIGH: clear winner | MEDIUM: reasonable preference | LOW: near-tie
    """
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


def compute_confidence(scores: list[float]) -> ConfidenceLevel:
    """
    Score-gap algorithm for confidence.

    WHY score-gap over fixed thresholds:
    - score=0.95 vs score=0.94 → gap=0.01 → LOW (effectively tied)
    - score=0.72 vs score=0.51 → gap=0.21 → HIGH (clear winner)
    - Fixed threshold of 0.8 would call 0.95 HIGH even if #2 is 0.94

    Thresholds:
      gap >= 0.20 → HIGH    (clear winner)
      gap >= 0.10 → MEDIUM  (reasonable preference)
      gap < 0.10  → LOW     (close race — show alternatives)
    """
    if len(scores) == 0:
        return ConfidenceLevel.LOW
    if len(scores) == 1:
        return ConfidenceLevel.HIGH
    sorted_scores = sorted(scores, reverse=True)
    gap = sorted_scores[0] - sorted_scores[1]
    if gap >= 0.20:
        return ConfidenceLevel.HIGH
    elif gap >= 0.10:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGE
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStage(BaseModel):
    """
    Execution record for a single pipeline stage.

    WHY tracked per-stage:
    - SLA monitoring: identify which stage is the bottleneck
    - Debugging: "query X failed at stage Y" without guessing
    - Alerting: duration_ms > threshold triggers PagerDuty

    retry_count > 0 means the stage succeeded after transient failures.
    """

    model_config = ConfigDict(frozen=False)

    stage: str
    status: StageStatus = StageStatus.PENDING
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[int] = None            # ErrorCode int value
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    metadata: dict[str, Any] = {}               # Stage-specific debug info

    def mark_running(self) -> None:
        self.status = StageStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_done(self) -> None:
        self.status = StageStatus.DONE
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

    def mark_failed(self, error: str, error_code: Optional[int] = None) -> None:
        self.status = StageStatus.FAILED
        self.error = error
        self.error_code = error_code
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

    def mark_skipped(self, reason: str) -> None:
        self.status = StageStatus.SKIPPED
        self.metadata["skip_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "error_code": self.error_code,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM METADATA
# ─────────────────────────────────────────────────────────────────────────────

class LLMMetadata(BaseModel):
    """
    Per-LLM-call tracking for cost, performance, and fallback monitoring.

    WHY tracked:
    - Cost: cost_usd per call → aggregate per user → billing / budget alerts
    - Fallback: fallback_used=True → primary model reliability metric
    - Performance: duration_ms per model → latency regression detection
    - Compliance: EU AI Act requires disclosure of which AI systems were used
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float                         # Computed: tokens × model rate
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    duration_ms: int
    request_id: str                         # Links to distributed trace
    called_at: datetime

    @field_validator("cost_usd")
    @classmethod
    def validate_cost(cls, v: float) -> float:
        if v < 0:
            raise ValidationError(
                message=f"cost_usd cannot be negative, got {v}",
                field="cost_usd",
                invalid_value=v,
            )
        return round(v, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "duration_ms": self.duration_ms,
            "called_at": self.called_at.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

class StructuredExplanation(BaseModel):
    """
    Typed LLM explanation output — replaces raw string responses.

    WHY structured (not raw string):
    - Unit testable: assert explanation.strengths[0] contains "large dataset"
    - UI renderable: summary bold, weaknesses red, recommendations as buttons
    - i18n ready: translate per field
    - A/B testable: swap generation strategy without changing consumer contract
    - Auditable: model_used + generated_at for compliance and cache invalidation

    Generated by Agent-1's explanation module.
    Consumed by RankedResult and AgentResponse.
    """

    model_config = ConfigDict(frozen=True)

    summary: str                            # 1–2 sentences: the headline reason
    key_factors: list[str]                  # Top 5 factors that drove the recommendation
    strengths: list[str]                    # Positive aspects for this query
    weaknesses: list[str]                   # Honest gaps or limitations
    recommendations: list[str]              # What to do next (preprocess, alternatives)
    confidence: float                       # 0.0–1.0 — explanation confidence
    reasoning: str                          # Full reasoning chain — for debug/audit
    model_used: str                         # Which LLM generated this
    generated_at: datetime                  # UTC — for cache invalidation

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValidationError(
                message=f"explanation confidence must be in [0.0, 1.0], got {v}",
                field="confidence",
                invalid_value=v,
                constraint="[0.0, 1.0]",
            )
        return round(v, 3)

    @field_validator("key_factors", "strengths", "weaknesses", "recommendations")
    @classmethod
    def validate_list_lengths(cls, v: list[str], info: Any) -> list[str]:
        if len(v) > 10:
            logger.warning(
                "explanation_list_too_long",
                extra={"field": info.field_name, "length": len(v)},
            )
            return v[:10]
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_factors": self.key_factors,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_dict_debug(self) -> dict[str, Any]:
        d = self.to_dict()
        d["reasoning"] = self.reasoning
        return d


# ─────────────────────────────────────────────────────────────────────────────
# DECISION TRACE
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTrace(BaseModel):
    """
    Full audit trail for a single ranking decision.

    WHY required:
    - EU AI Act Article 13: automated decisions must be explainable
    - Product debugging: "why was dataset X ranked #3 and not #1?"
    - A/B testing: compare score breakdowns across algorithm versions

    WHY top 20 only:
    - Storing all 1M traces = ~2GB per query
    - Users see top results — traces for rank 21+ are never read
    - lineage_ref → full record always retrievable if needed

    trace_version: versioned independently — trace format can evolve
    without schema_version bump on the full pipeline.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str                             # canonical_id
    dataset_title: str
    final_score: float
    final_rank: int
    score_breakdown: dict[str, float]           # {"semantic": 0.82, "quality": 0.71, ...}
    filters_passed: bool
    filter_reasons: list[str]                   # Why this dataset passed/failed filters
    lineage_ref: str                            # ingestion_timestamp + pipeline_version
    trace_version: str = "3.0.0"

    @field_validator("final_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    @field_validator("final_rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v < 1:
            raise ValidationError(
                message=f"final_rank must be >= 1, got {v}",
                field="final_rank",
                invalid_value=v,
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_title": self.dataset_title,
            "final_score": self.final_score,
            "final_rank": self.final_rank,
            "score_breakdown": self.score_breakdown,
            "filters_passed": self.filters_passed,
            "filter_reasons": self.filter_reasons,
            "lineage_ref": self.lineage_ref,
            "trace_version": self.trace_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RANKED RESULT — one entry in the recommendation list
# ─────────────────────────────────────────────────────────────────────────────

class RankedResult(BaseModel):
    """
    A single dataset recommendation — what the user sees.

    Combines:
    - EvaluatedDataset (scores + metadata)
    - StructuredExplanation (why this was recommended)
    - DecisionTrace (audit trail)
    - Download URL (direct link to data)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: EvaluatedDataset
    rank: int
    explanation: Optional[StructuredExplanation] = None
    trace: Optional[DecisionTrace] = None
    download_url: Optional[str] = None

    def to_dict_safe(self) -> dict[str, Any]:
        """API response representation — safe for external consumers."""
        return {
            "rank": self.rank,
            "dataset": self.dataset.to_dict_safe(),
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "download_url": self.download_url,
        }

    def to_dict_debug(self) -> dict[str, Any]:
        d = self.to_dict_safe()
        if self.explanation:
            d["explanation"] = self.explanation.to_dict_debug()
        if self.trace:
            d["trace"] = self.trace.to_dict()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RESPONSE — top-level response wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """
    Top-level response returned by the pipeline to the caller.

    Wraps ranked results with pipeline metadata, cost tracking,
    and partial-result signals.

    partial_result=True means some adapters failed but results were still returned.
    The caller should decide whether partial results are acceptable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Core results ──────────────────────────────────────────────────────────
    query_id: str                               # Matches SearchQuery.query_id
    results: list[RankedResult] = []
    total_found: int = 0                        # Total matching before top-N cutoff
    confidence: ConfidenceLevel = ConfidenceLevel.LOW

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    partial_result: bool = False                # True if some adapters failed
    adapter_results: dict[str, int] = {}        # {"kaggle": 15, "huggingface": 8}
    adapter_errors: dict[str, str] = {}         # {"openml": "timeout"} — failed adapters
    pipeline_stages: list[PipelineStage] = []

    # ── Cost + LLM tracking ───────────────────────────────────────────────────
    llm_metadata: list[LLMMetadata] = []
    total_cost_usd: float = 0.0

    # ── Timing ───────────────────────────────────────────────────────────────
    processing_time_ms: int = 0
    responded_at: datetime = None               # type: ignore[assignment]

    # ── Error (populated if total failure) ───────────────────────────────────
    error_message: Optional[str] = None
    error_code: Optional[int] = None

    # ── Versioning ────────────────────────────────────────────────────────────
    response_version: str = "3.0.0"

    def model_post_init(self, __context: Any) -> None:
        if self.responded_at is None:
            self.responded_at = datetime.now(timezone.utc)
        # Compute total_cost_usd from llm_metadata
        if self.llm_metadata and self.total_cost_usd == 0.0:
            self.total_cost_usd = round(sum(m.cost_usd for m in self.llm_metadata), 6)
        # Compute confidence from result scores
        if self.results and self.confidence == ConfidenceLevel.LOW:
            scores = [r.dataset.composite_score for r in self.results]
            self.confidence = compute_confidence(scores)

    def is_success(self) -> bool:
        """True if at least one result was returned."""
        return len(self.results) > 0

    def is_total_failure(self) -> bool:
        """True if no results and error_message is set."""
        return len(self.results) == 0 and self.error_message is not None

    def to_dict_safe(self) -> dict[str, Any]:
        """API response — safe for external consumers."""
        return {
            "query_id": self.query_id,
            "results": [r.to_dict_safe() for r in self.results],
            "total_found": self.total_found,
            "confidence": self.confidence.value,
            "partial_result": self.partial_result,
            "adapter_errors": self.adapter_errors,
            "processing_time_ms": self.processing_time_ms,
            "total_cost_usd": self.total_cost_usd,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "error_message": self.error_message,
            "response_version": self.response_version,
        }

    def to_dict_debug(self) -> dict[str, Any]:
        """Full representation including pipeline stages and LLM metadata."""
        d = self.to_dict_safe()
        d.update({
            "results": [r.to_dict_debug() for r in self.results],
            "adapter_results": self.adapter_results,
            "pipeline_stages": [s.to_dict() for s in self.pipeline_stages],
            "llm_metadata": [m.to_dict() for m in self.llm_metadata],
            "error_code": self.error_code,
        })
        return d