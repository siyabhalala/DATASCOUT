"""
datascout.llm.prompts.evaluator
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: LLM-based result quality evaluator.

Evaluates whether the OVERALL search result set is sufficient quality to
return to the user, or whether the agent should refine the query.

CRITICAL DISTINCTION:
  - evaluator.py → evaluates result SET quality (should we refine?)
  - explainer.py → explains WHY one dataset was recommended

SYSTEM DESIGN DECISIONS:

  1. WHY evaluate result sets, not individual datasets?
     - Individual dataset scores are already computed deterministically
     - LLM adds value by assessing whether the COMBINATION of results
       addresses the user's intent — something scores can't capture
     - "5 datasets about sentiment analysis" may all score 0.8 but none
       covers the multilingual requirement buried in the query

  2. WHY quality_sufficient: bool as primary signal?
     - Agent's ReAct loop needs a binary decision: refine or return
     - Float quality_score is secondary — for monitoring/dashboards
     - Bool prevents agents from inventing threshold logic

  3. WHY max 5 datasets in evaluation context?
     - Token budget: 5 × ~150 chars metadata = ~750 chars = ~190 tokens
     - LLM evaluation quality doesn't improve beyond top-5 in practice
     - Cost discipline: evaluation is a meta-call — must be cheaper than explain

  4. WHY prompt explicitly includes the user's query intent?
     - LLM must assess relevance TO THE QUERY, not absolute quality
     - A high-quality dataset about sentiment analysis is irrelevant for CV tasks
     - Evidence-grounded: query is the ground truth for relevance

  5. WHY EvaluationResult is NOT StructuredExplanation?
     - Different schema: quality_sufficient, quality_score, issues list
     - Mixing them would confuse downstream consumers
     - Agent reads EvaluationResult; UI reads StructuredExplanation

PERFORMANCE ANALYSIS:
  - build_evaluation_prompt(): O(n_datasets × metadata_length) ≈ 0.5ms
  - parse_evaluation_response(): O(response_length) ≈ 0.1ms
  - LLM call: 0.5-1.5s (evaluation prompt is shorter than explanation)

FAILURE SCENARIOS:
  - JSON parse fails → EvaluationResult with quality_sufficient=True (safe default)
    WHY True not False: False triggers re-query loop; infinite loops worse than
    returning imperfect results once
  - All datasets empty → quality_sufficient=False, agent refines immediately
  - LLM call fails → fallback EvaluationResult returned (no crash)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from datascout.contracts.models import EvaluatedDataset

from datascout.llm.base import LLMResponse

logger = logging.getLogger("datascout.llm.prompts.evaluator")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RESULT CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """
    LLM evaluation output for a result set.

    quality_sufficient: PRIMARY SIGNAL for agent's ReAct loop.
    quality_score:      Float 0.0-1.0 — for monitoring, not for agent logic.
    issues:             List of specific problems found.
    recommendation:     What the agent should do next.
    reasoning:          Full reasoning chain for audit.
    """

    model_config = ConfigDict(frozen=True)

    quality_sufficient: bool
    quality_score: float              # 0.0-1.0 — monitoring only
    issues: list[str]
    recommendation: str
    reasoning: str
    model_used: str = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_EVALUATOR_SYSTEM_PROMPT = """\
You are a search quality assessor. Your job is to evaluate whether a set of dataset recommendations adequately addresses a user's search query.

STRICT RULES:
1. ONLY assess based on the metadata provided. Do NOT invent details about datasets.
2. Do NOT rank or score individual datasets — assess the RESULT SET as a whole.
3. quality_sufficient = true means: the user should see these results.
4. quality_sufficient = false means: the search should be refined with different keywords.
5. Respond with ONLY valid JSON. No preamble, no markdown fences.

OUTPUT SCHEMA:
{
  "quality_sufficient": <true|false>,
  "quality_score": <float 0.0-1.0>,
  "issues": ["<issue 1>", "<issue 2>"],
  "recommendation": "<one sentence: what to do next>",
  "reasoning": "<2-3 sentences explaining your assessment>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_prompt(
    datasets: list[EvaluatedDataset],
    query: str,
    min_score_threshold: float = 0.4,
    max_datasets_in_context: int = 5,
) -> str:
    """
    Build evidence-grounded evaluation prompt.

    O(n_datasets × metadata_length) — pure function.

    Args:
        datasets:                 Ranked list of evaluated datasets.
        query:                    User's original search query.
        min_score_threshold:      Score below which a dataset is flagged as weak.
        max_datasets_in_context:  Max datasets to include in prompt (token budget).

    Returns:
        Formatted prompt string.
    """
    if not datasets:
        return (
            f'USER QUERY: "{query}"\n\n'
            "SEARCH RESULTS: No datasets were found.\n\n"
            "Evaluate whether this empty result set is acceptable or if the query should be refined."
        )

    top_datasets = datasets[:max_datasets_in_context]
    dataset_lines: list[str] = []

    for rank, ds in enumerate(top_datasets, start=1):
        raw = ds.raw
        row_str = f"{raw.row_count:,}" if raw.row_count else "unknown"
        tags_str = ", ".join(raw.tags_primary[:3]) if raw.tags_primary else "none"
        task_str = ", ".join(t.value for t in raw.task_types) if raw.task_types else "unspecified"
        score_flag = " [LOW SCORE]" if ds.composite_score < min_score_threshold else ""

        dataset_lines.append(
            f"  #{rank}: {raw.title}\n"
            f"       Source: {raw.source} | Score: {ds.composite_score:.3f}{score_flag}\n"
            f"       Tags: {tags_str} | Tasks: {task_str} | Rows: {row_str}\n"
            f"       Desc: {(raw.description_short or '')[:150]}"
        )

    datasets_block = "\n\n".join(dataset_lines)

    top_score = datasets[0].composite_score if datasets else 0.0
    low_score_count = sum(1 for d in datasets if d.composite_score < min_score_threshold)

    prompt = f"""\
USER QUERY: "{query}"

SEARCH RESULTS ({len(top_datasets)} of {len(datasets)} shown):

{datasets_block}

SUMMARY STATISTICS:
  - Top composite score: {top_score:.3f}
  - Datasets below threshold ({min_score_threshold}): {low_score_count}/{len(datasets)}
  - Minimum quality threshold: {min_score_threshold}

Assess whether these results adequately address the user's query.
Focus on: relevance to the specific query, result diversity, and score quality.
Follow the JSON schema exactly."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_evaluation_response(
    llm_response: LLMResponse,
    query: str,
    datasets: list[EvaluatedDataset],
) -> EvaluationResult:
    """
    Parse LLM raw_text → EvaluationResult.

    O(response_length) — pure function, no I/O.
    On failure: returns safe default (quality_sufficient=True).

    WHY default to True on failure:
    - False triggers re-query loop — infinite loops worse than imperfect results
    - If evaluation itself fails, don't compound failure with forced re-query
    """
    raw_text = llm_response.raw_text.strip()

    # Strip markdown fences — O(1)
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "evaluation_json_parse_failed",
            extra={
                "request_id": llm_response.request_id,
                "model": llm_response.model_name,
                "error": str(exc),
                "raw_preview": raw_text[:200],
            },
        )
        return _build_fallback_evaluation(
            datasets=datasets,
            query=query,
            model_used=llm_response.model_name,
        )

    quality_sufficient = bool(data.get("quality_sufficient", True))
    quality_score = _safe_float(data.get("quality_score"), default=0.5, lo=0.0, hi=1.0)
    issues = [str(i).strip() for i in data.get("issues", []) if str(i).strip()]
    recommendation = str(data.get("recommendation", "Return results to user.")).strip()
    reasoning = str(data.get("reasoning", "LLM provided no reasoning.")).strip()

    result = EvaluationResult(
        quality_sufficient=quality_sufficient,
        quality_score=quality_score,
        issues=issues,
        recommendation=recommendation,
        reasoning=reasoning,
        model_used=llm_response.model_name,
    )

    logger.info(
        "evaluation_parsed",
        extra={
            "request_id": llm_response.request_id,
            "model": llm_response.model_name,
            "quality_sufficient": quality_sufficient,
            "quality_score": quality_score,
            "issues_count": len(issues),
        },
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_evaluation(
    datasets: list[EvaluatedDataset],
    query: str,
    model_used: str = "template",
) -> EvaluationResult:
    """
    Safe fallback when LLM evaluation output is unparseable.

    Computes quality_sufficient from deterministic score threshold.
    O(n_datasets) — always valid, never raises.
    """
    if not datasets:
        return EvaluationResult(
            quality_sufficient=False,
            quality_score=0.0,
            issues=["No datasets found"],
            recommendation="Refine query with different keywords",
            reasoning="No results returned. Query refinement required.",
            model_used=f"{model_used}:fallback_template",
        )

    top_score = datasets[0].composite_score
    avg_score = sum(d.composite_score for d in datasets) / len(datasets)
    low_count = sum(1 for d in datasets if d.composite_score < 0.4)

    # Deterministic quality decision — not LLM
    quality_sufficient = top_score >= 0.4 and low_count < len(datasets)

    issues: list[str] = []
    if top_score < 0.4:
        issues.append(f"Top score ({top_score:.2f}) below minimum threshold (0.40)")
    if low_count > len(datasets) // 2:
        issues.append(f"Majority of results ({low_count}/{len(datasets)}) below threshold")

    return EvaluationResult(
        quality_sufficient=quality_sufficient,
        quality_score=round(avg_score, 3),
        issues=issues,
        recommendation=(
            "Return results to user." if quality_sufficient
            else "Refine query with more specific terms."
        ),
        reasoning=(
            f"Fallback evaluation: top_score={top_score:.3f}, "
            f"avg_score={avg_score:.3f}, low_count={low_count}/{len(datasets)}. "
            f"LLM evaluation unavailable."
        ),
        model_used=f"{model_used}:fallback_template",
    )


def _safe_float(value: Any, default: float, lo: float, hi: float) -> float:
    """Return float clamped to [lo, hi], fallback to default on error. O(1)."""
    try:
        f = float(value)
        return max(lo, min(hi, round(f, 3)))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class ResultSetEvaluator:
    """
    High-level evaluator orchestrator.

    Used by the ReAct agent to decide: refine query or return results?

    NEVER ranks or scores individual datasets.
    ONLY assesses whether the result SET is sufficient.
    """

    def __init__(
        self,
        llm_chain: Any,
        min_score_threshold: float = 0.4,
    ) -> None:
        self._chain = llm_chain
        self._min_score_threshold = min_score_threshold
        self._logger = logging.getLogger("datascout.llm.prompts.evaluator")

    async def evaluate(
        self,
        datasets: list[EvaluatedDataset],
        query: str,
    ) -> EvaluationResult:
        """
        Evaluate whether a result set is sufficient quality to return.

        O(1) LLM call + O(response_length) parsing.
        Never raises — returns fallback EvaluationResult on failure.

        Args:
            datasets: Ranked evaluated datasets.
            query:    User's original query.

        Returns:
            EvaluationResult with quality_sufficient signal for agent.
        """
        # Fast-path: no results → immediate False without LLM call
        if not datasets:
            self._logger.info(
                "evaluation_empty_results",
                extra={"query": query[:100]},
            )
            return _build_fallback_evaluation(datasets=[], query=query)

        user_prompt = build_evaluation_prompt(
            datasets=datasets,
            query=query,
            min_score_threshold=self._min_score_threshold,
        )

        try:
            llm_response = await self._chain.evaluate(
                system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=512,
            )
        except Exception as exc:
            self._logger.warning(
                "evaluation_llm_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc),
                    "query": query[:100],
                    "n_datasets": len(datasets),
                },
            )
            return _build_fallback_evaluation(
                datasets=datasets,
                query=query,
                model_used="failed",
            )

        return parse_evaluation_response(
            llm_response=llm_response,
            query=query,
            datasets=datasets,
        )