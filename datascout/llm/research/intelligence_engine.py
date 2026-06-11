"""
datascout.llm.research.intelligence_engine
──────────────────────────────────────────────────────────────────────────────
PHASE 4: Research Intelligence Engine.

Transforms deterministic evaluator outputs into human-friendly research
explanations. Gemini/Claude is strictly an EXPLANATION LAYER — it receives
pre-ranked datasets and explains them. It NEVER ranks, scores, or fabricates.

ARCHITECTURE:
  evaluator scores  ──→  ResearchIntelligenceEngine  ──→  ResearchInsight
       (truth)              (grounding + prompting)          (explanation)

WHAT THIS MODULE DOES:
  1. Builds grounded prompts anchored to evaluator signals
  2. Calls LLM with strict instructions (no ranking, no fabrication)
  3. Parses structured JSON responses into ResearchInsight objects
  4. Falls back to template-based insights on LLM failure
  5. Generates follow-up search suggestions from query context
  6. Identifies metadata gaps from structural quality signals
  7. Detects bias/risk signals from evaluator outputs

WHAT THIS MODULE NEVER DOES:
  - Reorder datasets (ranking is deterministic, final, and authoritative)
  - Invent row counts, download numbers, or quality signals
  - Override evaluator scores
  - Expose raw LLM output to the frontend

Author: DataScout Engineering
Version: 4.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from datascout.contracts.errors.exceptions import LLMError

logger = logging.getLogger("datascout.llm.research.intelligence_engine")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetInsight:
    """
    Human-readable explanation for a single dataset's ranking.
    All content is grounded in evaluator outputs — nothing fabricated.
    """
    dataset_id: str
    rank: int
    why_ranked: str                          # Primary ranking reason (1-2 sentences)
    strengths: list[str]                     # Concrete strengths from evaluator signals
    weaknesses: list[str]                    # Honest limitations
    metadata_gaps: list[str]                 # Missing metadata identified deterministically
    usability_notes: list[str]               # Practical ML usage notes
    score_narrative: str                     # Human-readable score breakdown
    confidence_note: str                     # What confidence level means for this result
    is_llm_generated: bool = True
    fallback_used: bool = False
    generated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "rank": self.rank,
            "why_ranked": self.why_ranked,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "metadata_gaps": self.metadata_gaps,
            "usability_notes": self.usability_notes,
            "score_narrative": self.score_narrative,
            "confidence_note": self.confidence_note,
            "is_llm_generated": self.is_llm_generated,
            "fallback_used": self.fallback_used,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


@dataclass
class ResearchContext:
    """
    Query-level research intelligence — what the system learned about
    the dataset ecosystem for this query.
    """
    query: str
    ecosystem_summary: str                   # What the dataset landscape looks like
    coverage_observations: list[str]         # What's well-covered vs sparse
    quality_trends: list[str]                # Metadata/annotation quality patterns
    follow_up_searches: list[str]            # Suggested next queries
    research_gaps: list[str]                 # Areas with weak coverage
    provider_notes: list[str]                # Source-specific observations
    total_candidates: int = 0
    confidence_level: str = "medium"
    is_llm_generated: bool = True
    fallback_used: bool = False
    generated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "ecosystem_summary": self.ecosystem_summary,
            "coverage_observations": self.coverage_observations,
            "quality_trends": self.quality_trends,
            "follow_up_searches": self.follow_up_searches,
            "research_gaps": self.research_gaps,
            "provider_notes": self.provider_notes,
            "total_candidates": self.total_candidates,
            "confidence_level": self.confidence_level,
            "is_llm_generated": self.is_llm_generated,
            "fallback_used": self.fallback_used,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


@dataclass
class ResearchIntelligenceResult:
    """Complete Phase 4 intelligence output for a search response."""
    dataset_insights: list[DatasetInsight]
    research_context: ResearchContext
    llm_latency_ms: int = 0
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    pipeline_degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_insights": [i.to_dict() for i in self.dataset_insights],
            "research_context": self.research_context.to_dict(),
            "intelligence_meta": {
                "llm_latency_ms": self.llm_latency_ms,
                "llm_tokens_used": self.llm_tokens_used,
                "llm_cost_usd": round(self.llm_cost_usd, 6),
                "pipeline_degraded": self.pipeline_degraded,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — pure functions, no I/O
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_INSIGHT_SYSTEM = """\
You are a senior ML dataset research analyst. You explain why a dataset ranked
where it did and help practitioners make concrete, informed decisions.

CRITICAL RULES — violating any of these makes the system untrustworthy:
1. ONLY reference facts present in the metadata given to you. If a number is
   not in the metadata, do NOT mention it. Never fabricate download counts,
   row counts, or quality claims.
2. Do NOT re-rank or re-score. The ranking came from a deterministic evaluator.
   You are the explainability layer, not the decision layer.
3. Be specific. "High metadata completeness" is not a strength — tell the user
   WHAT is there: e.g. "Includes class labels, column schema, and CC0 license."
4. Be honest about weaknesses. "No obvious limitations" means you're not
   looking hard enough. Every dataset has trade-offs.
5. Write like a research colleague who has seen 10,000 datasets, not an
   overly enthusiastic assistant.
6. Respond with ONLY valid JSON. No markdown fences. No preamble. No trailing text.

GOOD strength examples (grounded in metadata signals):
  - "Source is HuggingFace with 12,000+ downloads — strong community validation"
  - "Dataset includes transcription-level annotations with speaker metadata"
  - "CC0 license: unrestricted use for commercial and research applications"
  - "Updated in 2024 — data freshness is above average for this domain"

BAD strength examples (generic, fabricated):
  - "Good quality dataset"
  - "Widely used in the research community"
  - "Contains rich features"

OUTPUT SCHEMA (all fields required):
{
  "why_ranked": "<1-2 sentences: the single most important signal that put this dataset at this rank — cite the actual score dimension that dominated>",
  "strengths": ["<grounded strength citing actual metadata>", "<second grounded strength>", "<third if applicable>"],
  "weaknesses": ["<honest concrete limitation>", "<second limitation>"],
  "usability_notes": ["<practical note for someone actually training a model on this>", "<second practical note>"],
  "score_narrative": "<2-3 sentences: what the score breakdown tells a practitioner — mention the weakest and strongest dimensions by name>",
  "confidence_note": "<1 sentence: why the ranking confidence is what it is for this specific result>"
}"""


_RESEARCH_CONTEXT_SYSTEM = """\
You are a senior dataset research analyst. You observe patterns across a set
of search results and give a researcher genuine signal about the ecosystem.

CRITICAL RULES:
1. Only reference facts from the provided dataset metadata. No invented stats.
2. Coverage observations must cite concrete evidence: missing modalities,
   geographic bias, annotation gaps, license restrictions, etc.
3. Follow-up searches must be specific and actionable — "benchmark datasets
   for Hindi ASR in low-resource settings" not "more speech datasets".
4. Research gaps should identify what was NOT found but would be useful —
   this is the highest-value signal you can provide.
5. Respond with ONLY valid JSON. No markdown. No preamble.

OUTPUT SCHEMA:
{
  "ecosystem_summary": "<2-3 sentences: what the dataset landscape actually looks like for this query — coverage, gaps, dominant providers>",
  "coverage_observations": ["<concrete observation citing what you saw>", "<second observation>"],
  "quality_trends": ["<pattern in quality signals across the result set>", "<second quality pattern>"],
  "follow_up_searches": ["<specific actionable follow-up query 1>", "<query 2>", "<query 3>"],
  "research_gaps": ["<missing coverage area that would be useful>", "<second gap>"],
  "provider_notes": ["<provider-specific observation if it's genuinely interesting>"]
}"""



_BATCH_INSIGHT_SYSTEM = """You are a senior ML dataset research analyst. You explain why datasets ranked
where they did and help practitioners make concrete, informed decisions.

CRITICAL RULES:
1. ONLY reference facts present in the metadata given to you. Never fabricate.
2. Do NOT re-rank or re-score. The ranking came from a deterministic evaluator.
3. Be specific. Cite actual metadata signals, not generic praise.
4. Be honest about weaknesses — every dataset has trade-offs.
5. Write like a research colleague, not an enthusiastic assistant.
6. Respond with ONLY valid JSON — a list, one object per dataset. No markdown.

OUTPUT SCHEMA (return a JSON array, one object per dataset, in the same order):
[
  {
    "dataset_id": "<canonical_id>",
    "why_ranked": "<1 sentence: the single most important signal>",
    "strengths": ["<grounded strength>", "<second strength>"],
    "weaknesses": ["<honest limitation>"],
    "usability_notes": ["<practical note for training>"],
    "score_narrative": "<1-2 sentences about the score breakdown>",
    "confidence_note": "<1 sentence about ranking confidence>"
  }
]"""


# Combined prompt — ONE Gemini call returns both dataset insights AND research context.
# Halves Gemini quota usage vs the previous 2-call approach.
_COMBINED_SYSTEM = """You are a senior ML dataset research analyst.

RULES:
1. Only reference facts in the metadata given. Never fabricate.
2. Do NOT re-rank. The ranking is final and came from a deterministic evaluator.
3. Be specific — cite actual signals. No generic praise.
4. Respond with ONLY valid JSON matching the schema below. No markdown, no preamble.

OUTPUT SCHEMA (one JSON object):
{
  "dataset_insights": [
    {
      "dataset_id": "<canonical_id>",
      "why_ranked": "<1 sentence: the dominant signal>",
      "strengths": ["<grounded strength>", "<second strength>"],
      "weaknesses": ["<honest limitation>"],
      "usability_notes": ["one practical ML usage note"],
      "score_narrative": "brief score explanation",
      "confidence_note": "brief confidence explanation"
    }
  ],
  "ecosystem_summary": "<2 sentences: what the dataset landscape looks like for this query>",
  "follow_up_searches": ["<specific follow-up query 1>", "<query 2>"],
  "research_gaps": ["<gap 1>", "<gap 2>"]
}"""


def _build_dataset_insight_prompt(
    dataset_dict: dict[str, Any],
    score_breakdown: dict[str, Any],
    rank: int,
    total_results: int,
    query: str,
) -> str:
    """Build evidence-grounded prompt for per-dataset insight."""
    raw = dataset_dict.get("raw", dataset_dict)
    title = raw.get("title", "Unknown")
    source = raw.get("source", "unknown")
    description = (raw.get("description_short") or raw.get("description") or "")[:400]
    tags = ", ".join(raw.get("tags_primary", [])[:6]) or "none"
    row_count = raw.get("row_count")
    col_count = raw.get("column_count")
    file_size = raw.get("file_size_bytes")
    license_type = raw.get("license_type", "unknown")
    last_updated = raw.get("last_updated") or raw.get("last_updated_str", "unknown")
    task_types = ", ".join(raw.get("task_types", [])) or "unspecified"
    completeness = raw.get("metadata_completeness", 0)
    quality_tier = raw.get("quality_tier", "unknown")
    has_description = raw.get("has_description", False)
    has_schema = raw.get("has_schema_info", False)
    has_license = raw.get("has_license_info", False)
    composite_score = dataset_dict.get("composite_score", 0)

    # Format score breakdown readably
    score_lines = []
    for dim, score in score_breakdown.items():
        if isinstance(score, (int, float)):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            score_lines.append(f"  {dim:20s} [{bar}] {score:.3f}")
    score_block = "\n".join(score_lines) if score_lines else "  (no breakdown available)"

    # Format metadata completeness signals
    completeness_signals = []
    if not has_description:
        completeness_signals.append("⚠ No description available")
    if not has_schema:
        completeness_signals.append("⚠ Column/feature info missing")
    if not has_license:
        completeness_signals.append("⚠ License not specified")
    if row_count is None:
        completeness_signals.append("⚠ Row count unknown")
    completeness_str = "\n".join(completeness_signals) if completeness_signals else "All core fields present"

    return f"""USER QUERY: "{query}"

DATASET RANKED #{rank} of {total_results}:
  Title:          {title}
  Source:         {source}
  Description:    {description}
  Tags:           {tags}
  Task Types:     {task_types}
  Rows:           {row_count if row_count else "unknown"}
  Columns:        {col_count if col_count else "unknown"}
  File Size:      {_fmt_size(file_size)}
  License:        {license_type}
  Last Updated:   {last_updated}
  Quality Tier:   {quality_tier}
  Completeness:   {completeness:.1%}

EVALUATOR SCORE BREAKDOWN (DO NOT RE-SCORE):
  Composite:      {composite_score:.4f}
{score_block}

METADATA COMPLETENESS SIGNALS:
{completeness_str}

Based solely on the evidence above, explain this dataset's recommendation.
Follow the JSON schema exactly. No fabrication. No re-ranking."""



def _build_batch_insight_prompt(
    datasets: list[dict],
    score_breakdowns: list[dict],
    query: str,
) -> str:
    """
    Build one prompt covering ALL datasets.
    One Gemini call instead of N sequential calls — prevents rate limit cascade.
    """
    blocks = []
    for i, (ds, breakdown) in enumerate(zip(datasets, score_breakdowns)):
        raw = ds.get("raw", ds)
        title        = raw.get("title", "Unknown")
        source       = raw.get("source", "unknown")
        description  = (raw.get("description_short") or raw.get("description") or "")[:200]
        tags         = ", ".join(raw.get("tags_primary", [])[:5]) or "none"
        license_type = raw.get("license_type", "unknown")
        last_updated = raw.get("last_updated") or raw.get("last_updated_str", "unknown")
        composite    = ds.get("composite_score", 0)
        # FIX: Check top-level canonical_id first (search_v2 stores it there),
        # then fall back to raw dict, then to positional rank ID.
        canonical_id = (
            ds.get("canonical_id")
            or raw.get("canonical_id")
            or f"rank_{i+1}"
        )

        score_lines = []
        for dim, score in breakdown.items():
            if isinstance(score, (int, float)):
                score_lines.append(f"    {dim}: {score:.3f}")
        score_block = "\n".join(score_lines) or "    (no breakdown)"

        blocks.append(f"""DATASET #{i+1} (id: {canonical_id}):
  Title:       {title}
  Source:      {source}
  Description: {description}
  Tags:        {tags}
  License:     {license_type}
  Updated:     {last_updated}
  Composite:   {composite:.4f}
  Scores:
{score_block}""")

    # Check if results look relevant to the query
    # A dataset with query_match=0 and no keyword overlap is probably wrong
    query_tokens = set(query.lower().split())
    relevant_count = 0
    for ds in datasets:
        raw = ds.get("raw", ds)
        title = (raw.get("title") or "").lower()
        tags  = " ".join(raw.get("tags_primary", []) or []).lower()
        if any(t in f"{title} {tags}" for t in query_tokens if len(t) > 3):
            relevant_count += 1

    relevance_note = ""
    if relevant_count == 0:
        relevance_note = (
            "\n\nIMPORTANT: These results may not be directly relevant to the query. "
            "Be honest about this in your insights — note that the dataset may not match "
            "what the user is looking for and suggest what they should search for instead."
        )

    return f"""USER QUERY: "{query}"

{chr(10).join(blocks)}

Return a JSON array with one insight object per dataset, in order.
Follow the schema exactly. No markdown. No fabrication. No re-ranking.{relevance_note}"""


def _build_research_context_prompt(
    query: str,
    top_datasets: list[dict[str, Any]],
    total_candidates: int,
    confidence_level: str,
) -> str:
    """Build prompt for query-level ecosystem analysis."""
    dataset_summaries = []
    source_counts: dict[str, int] = {}

    for i, ds in enumerate(top_datasets[:5], 1):
        raw = ds.get("raw", ds)
        title = raw.get("title", "?")
        source = raw.get("source", "?")
        score = ds.get("composite_score", 0)
        completeness = raw.get("metadata_completeness", 0)
        has_desc = raw.get("has_description", False)
        task_types = ", ".join(raw.get("task_types", [])[:3]) or "unspecified"
        last_updated = raw.get("last_updated") or "unknown"

        source_counts[source] = source_counts.get(source, 0) + 1
        dataset_summaries.append(
            f"  #{i} [{source}] {title!r}  score={score:.3f}  "
            f"completeness={completeness:.1%}  has_description={has_desc}  "
            f"tasks={task_types}  updated={last_updated}"
        )

    summaries_block = "\n".join(dataset_summaries) or "  (no datasets)"
    source_dist = ", ".join(f"{s}: {n}" for s, n in source_counts.items())

    return f"""USER QUERY: "{query}"

SEARCH RESULTS OVERVIEW:
  Total candidates found: {total_candidates}
  Results returned:       {len(top_datasets)}
  Confidence level:       {confidence_level}
  Source distribution:    {source_dist or "mixed"}

TOP DATASETS (DO NOT RE-RANK — this is your analytical input only):
{summaries_block}

Based on the above, provide research-level ecosystem analysis.
Be specific to what you can see — don't make up coverage gaps you can't verify.
Follow-up searches should be genuinely useful next steps for this research goal."""


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC METADATA GAP DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_metadata_gaps(dataset_dict: dict[str, Any]) -> list[str]:
    """
    Deterministically identify metadata gaps from structural signals.
    No LLM involved — pure rule-based analysis.

    Returns list of human-readable gap descriptions.
    """
    raw = dataset_dict.get("raw", dataset_dict)
    gaps = []

    if not raw.get("has_description"):
        gaps.append("No dataset description — hard to assess domain fit without reading the data directly")

    if not raw.get("has_schema_info"):
        gaps.append("Column/feature names unavailable — feature engineering planning requires manual exploration")

    if not raw.get("has_license_info"):
        gaps.append("License not specified — verify usage rights before commercial or production use")

    if raw.get("row_count") is None:
        gaps.append("Dataset size unknown — cannot assess if it's large enough for your use case")

    if not raw.get("task_types"):
        gaps.append("Task types not declared — dataset-task alignment unverified")

    if not raw.get("tags_primary"):
        gaps.append("No tags available — domain classification is uncertain")

    last_updated = raw.get("last_updated")
    if last_updated is None:
        gaps.append("Last update date unknown — freshness cannot be assessed")

    return gaps


def detect_bias_signals(dataset_dict: dict[str, Any], score_breakdown: dict[str, Any]) -> list[str]:
    """
    Identify bias and risk signals from evaluator outputs.
    Returns actionable warnings — never fabricated.
    """
    raw = dataset_dict.get("raw", dataset_dict)
    signals = []

    freshness = score_breakdown.get("freshness", 1.0)
    if isinstance(freshness, (int, float)) and freshness < 0.4:
        signals.append("Dataset may be outdated — low freshness score suggests data older than 2 years")

    quality = score_breakdown.get("quality", 1.0)
    if isinstance(quality, (int, float)) and quality < 0.3:
        signals.append("Low quality score — documentation gaps may hide data issues worth investigating")

    completeness = raw.get("metadata_completeness", 1.0)
    if completeness < 0.4:
        signals.append("Incomplete metadata — key fields missing, which may indicate undocumented collection bias")

    task_relevance = score_breakdown.get("task_relevance", 1.0)
    if isinstance(task_relevance, (int, float)) and task_relevance < 0.5:
        signals.append("Moderate task alignment — verify the dataset actually contains labels/features for your task")

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK GENERATORS — deterministic, no LLM
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_dataset_insight(
    dataset_dict: dict[str, Any],
    score_breakdown: dict[str, Any],
    rank: int,
    query: str,
) -> DatasetInsight:
    """Template-based insight when LLM fails. Always valid, never fabricates."""
    raw = dataset_dict.get("raw", dataset_dict)
    title = raw.get("title", "Unknown Dataset")
    source = raw.get("source", "unknown")
    composite = dataset_dict.get("composite_score", 0)
    completeness = raw.get("metadata_completeness", 0)

    # Identify top scoring dimension
    score_dims = {k: v for k, v in score_breakdown.items()
                  if k != "composite" and isinstance(v, (int, float))}
    top_dim = max(score_dims, key=score_dims.get) if score_dims else "quality"
    top_score = score_dims.get(top_dim, 0)

    why_ranked = (
        f"Ranked #{rank} with a composite score of {composite:.2f}. "
        f"Strongest signal was {top_dim.replace('_', ' ')} ({top_score:.2f}), "
        f"suggesting good alignment with your query on that dimension."
    )

    strengths = [
        f"Source: {source} — a recognized dataset provider",
        f"Metadata completeness: {completeness:.1%} of tracked fields populated",
    ]
    if raw.get("has_description"):
        strengths.append("Has a description to help assess domain fit")
    if raw.get("has_license_info"):
        strengths.append("License information available")

    weaknesses = detect_metadata_gaps(dataset_dict)[:2] or [
        "Review dataset card manually to verify fit for your use case"
    ]

    # Score narrative from breakdown
    task_rel = score_dims.get("task_relevance", 0)
    freshness = score_dims.get("freshness", 0)
    score_narrative = (
        f"Task relevance score of {task_rel:.2f} indicates {'strong' if task_rel > 0.6 else 'moderate'} "
        f"alignment with your ML objective. Freshness score of {freshness:.2f} reflects "
        f"{'recent maintenance' if freshness > 0.6 else 'potentially older data'}."
    )

    return DatasetInsight(
        dataset_id=(
            dataset_dict.get("canonical_id")
            or raw.get("canonical_id")
            or f"rank_{rank}"
        ),
        rank=rank,
        why_ranked=why_ranked,
        strengths=strengths,
        weaknesses=weaknesses,
        metadata_gaps=detect_metadata_gaps(dataset_dict),
        usability_notes=[
            "Run exploratory data analysis before committing to training",
            "Verify label quality and class distribution matches your requirements",
        ],
        score_narrative=score_narrative,
        confidence_note="Confidence reflects how clearly this dataset outscored alternatives.",
        is_llm_generated=False,
        fallback_used=True,
        generated_at=datetime.now(tz=timezone.utc),
    )


def _build_fallback_research_context(
    query: str,
    top_datasets: list[dict[str, Any]],
    total_candidates: int,
    confidence_level: str,
) -> ResearchContext:
    """Template-based context when LLM fails. Deterministic and always valid."""
    sources = list({(ds.get("raw") or ds).get("source", "unknown") for ds in top_datasets})
    source_str = " and ".join(sources) if sources else "multiple providers"

    # Compute avg completeness
    completeness_vals = [
        (ds.get("raw") or ds).get("metadata_completeness", 0)
        for ds in top_datasets
    ]
    avg_completeness = sum(completeness_vals) / len(completeness_vals) if completeness_vals else 0

    ecosystem_summary = (
        f"Found {total_candidates} candidate datasets for '{query}' across {source_str}. "
        f"Top results show average metadata completeness of {avg_completeness:.1%}. "
        f"Search confidence is {confidence_level}, "
        f"{'suggesting a clear best option' if confidence_level == 'high' else 'review multiple options before deciding'}."
    )

    # Generate follow-up suggestions based on query keywords
    words = [w for w in query.lower().split() if len(w) > 3]
    follow_ups = []
    if words:
        follow_ups.append(f"{query} benchmark")
        follow_ups.append(f"{words[0]} labeled dataset")
        if len(words) > 1:
            follow_ups.append(f"{words[-1]} annotations")
    follow_ups.append(f"{query} evaluation")

    return ResearchContext(
        query=query,
        ecosystem_summary=ecosystem_summary,
        coverage_observations=[
            f"Results span {len(sources)} provider(s): {', '.join(sources)}",
            f"Average metadata completeness is {avg_completeness:.1%} across top results",
        ],
        quality_trends=[
            "Review description availability — some datasets may lack sufficient documentation",
            "Check license information before production deployment",
        ],
        follow_up_searches=follow_ups[:4],
        research_gaps=[
            "Verify task type labels are present in the actual data",
            "Check for temporal coverage if your use case involves time-sensitive data",
        ],
        provider_notes=[f"Results from: {source_str}"],
        total_candidates=total_candidates,
        confidence_level=confidence_level,
        is_llm_generated=False,
        fallback_used=True,
        generated_at=datetime.now(tz=timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ResearchIntelligenceEngine:
    """
    Phase 4 AI research intelligence layer.

    Receives pre-ranked datasets + evaluator scores.
    Returns human-friendly research explanations.
    NEVER re-ranks, never fabricates, never overrides evaluator authority.
    """

    def __init__(self, llm_provider: Any = None) -> None:
        """
        Args:
            llm_provider: LLM provider (ClaudeProvider, etc.) or None.
                          If None, falls back to deterministic templates.
        """
        self._llm = llm_provider
        self._logger = logging.getLogger("datascout.llm.research.intelligence_engine")

    async def generate_intelligence(
        self,
        query: str,
        ranked_datasets: list[dict[str, Any]],
        score_breakdowns: list[dict[str, Any]],
        total_candidates: int,
        confidence_level: str = "medium",
        max_explain: int = 3,
    ) -> ResearchIntelligenceResult:
        """
        Generate full research intelligence in ONE Gemini call.

        v5.0: Merged the previous 2 concurrent calls (batch insights + research
        context) into a single combined call. This halves Gemini quota usage per
        search — critical for the free tier (15 RPM / 1M tokens/day).

        If Gemini fails, falls back to deterministic templates immediately.
        Search results are never affected regardless of LLM state.
        """
        t_start = time.monotonic()

        llm_ready = self._llm is not None and hasattr(self._llm, "complete")

        dataset_insights, research_context = await self._generate_combined(
            query=query,
            datasets=ranked_datasets[:max_explain],
            score_breakdowns=score_breakdowns[:max_explain],
            top_datasets=ranked_datasets,
            total_candidates=total_candidates,
            confidence_level=confidence_level,
            llm_ready=llm_ready,
        )

        degraded = (
            any(i.fallback_used for i in dataset_insights)
            or research_context.fallback_used
        )

        latency_ms = int((time.monotonic() - t_start) * 1000)
        self._logger.info(
            "research_intelligence_generated",
            extra={
                "query": query[:80],
                "insights_count": len(dataset_insights),
                "latency_ms": latency_ms,
                "degraded": degraded,
                "llm_calls": "1_combined" if llm_ready else "0_fallback",
            },
        )

        return ResearchIntelligenceResult(
            dataset_insights=dataset_insights,
            research_context=research_context,
            llm_latency_ms=latency_ms,
            llm_tokens_used=0,
            llm_cost_usd=0.0,
            pipeline_degraded=degraded,
        )

    async def _generate_combined(
        self,
        query: str,
        datasets: list[dict[str, Any]],
        score_breakdowns: list[dict[str, Any]],
        top_datasets: list[dict[str, Any]],
        total_candidates: int,
        confidence_level: str,
        llm_ready: bool,
    ) -> tuple[list[DatasetInsight], ResearchContext]:
        """
        ONE Gemini call that returns both dataset insights AND research context.

        Replaces the previous 2-call approach (batch insights + research context
        running concurrently). Halves quota usage: free tier is 15 RPM / 1M TPD,
        so every saved call directly reduces 503 frequency.

        Falls back fully to deterministic templates on any failure.
        """
        # ── Pure deterministic path (no LLM) ──────────────────────────────────
        if not llm_ready or not datasets:
            insights = [
                _build_fallback_dataset_insight(ds, bd, i + 1, query)
                for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns))
            ]
            context = _build_fallback_research_context(
                query, top_datasets, total_candidates, confidence_level
            )
            return insights, context

        # ── Build combined prompt ─────────────────────────────────────────────
        # Dataset blocks (reuse existing _build_batch_insight_prompt logic)
        dataset_section = _build_batch_insight_prompt(datasets, score_breakdowns, query)

        # Compact ecosystem section
        source_counts: dict[str, int] = {}
        summaries = []
        for i, ds in enumerate(top_datasets[:5], 1):
            raw = ds.get("raw", ds)
            title  = ds.get("title") or raw.get("title", "?")
            source = ds.get("source") or raw.get("source", "?")
            score  = ds.get("composite_score", 0)
            source_counts[source] = source_counts.get(source, 0) + 1
            summaries.append(f"  #{i} [{source}] {title!r}  score={score:.3f}")

        source_dist = ", ".join(f"{s}:{n}" for s, n in source_counts.items())
        eco_section = (
            f"ECOSYSTEM OVERVIEW:\n"
            f"  Total candidates: {total_candidates} | Confidence: {confidence_level}\n"
            f"  Sources: {source_dist or 'mixed'}\n"
            + "\n".join(summaries)
        )

        combined_prompt = (
            dataset_section
            + "\n\n"
            + eco_section
            + "\n\nReturn ONE JSON object matching the schema (dataset_insights array + "
            "ecosystem_summary + follow_up_searches + research_gaps). "
            "No markdown. No fabrication."
        )

        try:
            resp = await self._llm.complete(
                system_prompt=_COMBINED_SYSTEM,
                user_prompt=combined_prompt,
                max_tokens=2500,
                temperature=0.1,
            )
            raw_text = _strip_json_fences(resp.raw_text.strip())
            parsed = json.loads(raw_text)

        except json.JSONDecodeError:
            self._logger.warning(
                "gemini_invalid_json",
                extra={
                    "preview": raw_text[:500],
                    "length": len(raw_text),
                },
            )
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw_text[start:end + 1])
            else:
                raise

        except Exception as exc:
            self._logger.exception(
                "combined_intelligence_failed",
                extra={
                    "error": type(exc).__name__,
                    "detail": str(exc)[:500],
                },
            )
            insights = [
                _build_fallback_dataset_insight(ds, bd, i + 1, query)
                for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns))
            ]
            context = _build_fallback_research_context(
                query=query,
                top_datasets=top_datasets,
                total_candidates=total_candidates,
                confidence_level=confidence_level,
            )
            return insights, context

        # ── Parse dataset insights ────────────────────────────────────────────
        raw_items = parsed.get("dataset_insights", [])
        item_map: dict[str, dict] = {}
        for pos, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            returned_id = item.get("dataset_id", "")
            if returned_id:
                item_map[returned_id] = item
                item_map[returned_id.lower()] = item
            item_map[f"rank_{pos+1}"] = item
            item_map[f"dataset_{pos+1}"] = item
            item_map[str(pos+1)] = item

        insights: list[DatasetInsight] = []
        for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns)):
            raw_ds = ds.get("raw", ds)
            did = (
                ds.get("canonical_id")
                or raw_ds.get("canonical_id")
                or f"rank_{i+1}"
            )
            item = (
                item_map.get(did)
                or item_map.get(did.lower() if did else "")
                or item_map.get(f"rank_{i+1}")
                or item_map.get(f"dataset_{i+1}")
                or item_map.get(str(i+1))
                or (raw_items[i] if i < len(raw_items) else None)
            )

            if item and isinstance(item, dict):
                insights.append(DatasetInsight(
                    dataset_id=did,
                    rank=i + 1,
                    why_ranked=item.get("why_ranked", ""),
                    strengths=item.get("strengths", []),
                    weaknesses=item.get("weaknesses", []),
                    metadata_gaps=detect_metadata_gaps(ds),
                    usability_notes=item.get("usability_notes", []),
                    score_narrative=item.get("score_narrative", ""),
                    confidence_note=item.get("confidence_note", ""),
                    fallback_used=False,
                    is_llm_generated=True,
                    generated_at=datetime.now(tz=timezone.utc),
                ))
            else:
                insights.append(_build_fallback_dataset_insight(ds, bd, i + 1, query))

        # ── Parse research context ────────────────────────────────────────────
        context = ResearchContext(
            query=query,
            ecosystem_summary=parsed.get("ecosystem_summary", ""),
            coverage_observations=[],
            quality_trends=[],
            follow_up_searches=parsed.get("follow_up_searches", []),
            research_gaps=parsed.get("research_gaps", []),
            provider_notes=[],
            fallback_used=False,
            generated_at=datetime.now(tz=timezone.utc),
        )

        self._logger.info(
            "combined_intelligence_success",
            extra={"datasets": len(insights), "model": getattr(resp, "model_name", "?")},
        )
        return insights, context

    async def _generate_batch_insights(
        self,
        query: str,
        datasets: list[dict[str, Any]],
        score_breakdowns: list[dict[str, Any]],
        llm_ready: bool,
    ) -> list[DatasetInsight]:
        """
        Generate insights for ALL datasets in ONE Gemini call.
        Falls back to per-dataset templates if LLM fails.
        """
        if not llm_ready or not datasets:
            return [
                _build_fallback_dataset_insight(
                    dataset_dict=ds,
                    score_breakdown=bd,
                    rank=i + 1,
                    query=query,
                )
                for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns))
            ]

        prompt = _build_batch_insight_prompt(datasets, score_breakdowns, query)

        try:
            response = await self._llm.complete(
                system_prompt=_BATCH_INSIGHT_SYSTEM,
                user_prompt=prompt,
                max_tokens=2500,
                temperature=0.1,
            )

            if getattr(response, "is_degraded", False):
                raise ValueError("LLM response is degraded")

            # Parse JSON array from response
            raw_text = response.raw_text.strip()
            raw_text = _strip_json_fences(raw_text)
            parsed = json.loads(raw_text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed)}")

            insights: list[DatasetInsight] = []
            for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns)):
                raw_ds = ds.get("raw", ds)
                dataset_id = (
                    ds.get("canonical_id")
                    or raw_ds.get("canonical_id")
                    or f"rank_{i+1}"
                )
                item = (
                    next((p for p in parsed if p.get("dataset_id") == dataset_id), None)
                    or next((p for p in parsed if p.get("dataset_id") == f"rank_{i+1}"), None)
                    or next((p for p in parsed if p.get("dataset_id") == f"dataset_{i+1}"), None)
                    or (parsed[i] if i < len(parsed) else None)
                )
                if item is None:
                    insights.append(_build_fallback_dataset_insight(ds, bd, i+1, query))
                    continue
                insights.append(DatasetInsight(
                    dataset_id=dataset_id,
                    rank=i + 1,
                    why_ranked=item.get("why_ranked", ""),
                    strengths=item.get("strengths", []),
                    weaknesses=item.get("weaknesses", []),
                    metadata_gaps=detect_metadata_gaps(ds),
                    usability_notes=item.get("usability_notes", []),
                    score_narrative=item.get("score_narrative", ""),
                    confidence_note=item.get("confidence_note", ""),
                    fallback_used=False,
                    generated_at=datetime.now(tz=timezone.utc),
                ))
            return insights

        except Exception as exc:
            self._logger.warning(
                "batch_insight_llm_failed",
                extra={"error": type(exc).__name__, "detail": str(exc)[:200]},
            )
            return [
                _build_fallback_dataset_insight(ds, bd, i+1, query)
                for i, (ds, bd) in enumerate(zip(datasets, score_breakdowns))
            ]

    async def _generate_dataset_insight(
        self,
        dataset_dict: dict[str, Any],
        score_breakdown: dict[str, Any],
        rank: int,
        total_results: int,
        query: str,
    ) -> DatasetInsight:
        """Generate insight for one dataset. Falls back gracefully."""
        raw = dataset_dict.get("raw", dataset_dict)
        dataset_id = (
            dataset_dict.get("canonical_id")
            or raw.get("canonical_id")
            or f"rank_{rank}"
        )

        llm_ready = self._llm is not None and hasattr(self._llm, "complete")
        if not llm_ready:
            return _build_fallback_dataset_insight(
                dataset_dict=dataset_dict,
                score_breakdown=score_breakdown,
                rank=rank,
                query=query,
            )

        prompt = _build_dataset_insight_prompt(
            dataset_dict=dataset_dict,
            score_breakdown=score_breakdown,
            rank=rank,
            total_results=total_results,
            query=query,
        )

        try:
            response = await self._llm.complete(
                system_prompt=_DATASET_INSIGHT_SYSTEM,
                user_prompt=prompt,
                max_tokens=800,
                temperature=0.1,
            )
            return _parse_dataset_insight_response(
                raw_text=response.raw_text,
                dataset_id=dataset_id,
                rank=rank,
                dataset_dict=dataset_dict,
                score_breakdown=score_breakdown,
                query=query,
            )
        except Exception as exc:
            import traceback as _tb
            print(f"\n[ENGINE INSIGHT FAILED rank={rank}] {type(exc).__name__}: {exc}")
            _tb.print_exc()
            self._logger.warning(
                "dataset_insight_llm_failed",
                extra={
                    "rank": rank,
                    "dataset_id": dataset_id,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                },
                exc_info=True,
            )
            return _build_fallback_dataset_insight(
                dataset_dict=dataset_dict,
                score_breakdown=score_breakdown,
                rank=rank,
                query=query,
            )

    async def _generate_research_context(
        self,
        query: str,
        top_datasets: list[dict[str, Any]],
        total_candidates: int,
        confidence_level: str,
    ) -> ResearchContext:
        """Generate query-level context. Falls back gracefully."""
        llm_ready = self._llm is not None and hasattr(self._llm, "complete")
        if not llm_ready:
            return _build_fallback_research_context(
                query=query,
                top_datasets=top_datasets,
                total_candidates=total_candidates,
                confidence_level=confidence_level,
            )

        prompt = _build_research_context_prompt(
            query=query,
            top_datasets=top_datasets,
            total_candidates=total_candidates,
            confidence_level=confidence_level,
        )

        try:
            response = await self._llm.complete(
                system_prompt=_RESEARCH_CONTEXT_SYSTEM,
                user_prompt=prompt,
                max_tokens=600,
                temperature=0.15,
            )
            return _parse_research_context_response(
                raw_text=response.raw_text,
                query=query,
                top_datasets=top_datasets,
                total_candidates=total_candidates,
                confidence_level=confidence_level,
            )
        except Exception as exc:
            self._logger.warning(
                "research_context_llm_failed",
                extra={"query": query[:60], "error": type(exc).__name__},
            )
            return _build_fallback_research_context(
                query=query,
                top_datasets=top_datasets,
                total_candidates=total_candidates,
                confidence_level=confidence_level,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

_parse_logger = logging.getLogger("datascout.llm.research.parser")


def _strip_json_fences(text: str) -> str:
    """
    Strip markdown code fences and preamble text from LLM response robustly.

    Handles:
    - ```json ... ``` fences
    - Plain ``` fences
    - Preamble text like "Here is the JSON:" before the actual JSON
    - Trailing text after the JSON closes
    """
    import re
    text = text.strip()

    # Remove opening fence: ```json, ``` json, ```JSON, just ```
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text, count=1).strip()
    # Remove closing fence
    text = re.sub(r"```\s*$", "", text).strip()

    # If still has preamble text before JSON array or object, extract JSON
    # Find first [ or { — that's where JSON starts
    array_start = text.find("[")
    obj_start   = text.find("{")

    if array_start == -1 and obj_start == -1:
        return text  # no JSON found, return as-is and let caller handle

    if array_start != -1 and (obj_start == -1 or array_start < obj_start):
        # JSON array — find matching closing ]
        text = text[array_start:]
        # Find last ] to strip trailing text
        last_bracket = text.rfind("]")
        if last_bracket != -1:
            text = text[:last_bracket + 1]
    elif obj_start != -1:
        # JSON object — find matching closing }
        text = text[obj_start:]
        last_brace = text.rfind("}")
        if last_brace != -1:
            text = text[:last_brace + 1]

    return text.strip()


def _parse_dataset_insight_response(
    raw_text: str,
    dataset_id: str,
    rank: int,
    dataset_dict: dict[str, Any],
    score_breakdown: dict[str, Any],
    query: str,
) -> DatasetInsight:
    """Parse LLM response into DatasetInsight. Falls back on any error."""
    text = _strip_json_fences(raw_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as je:
        _parse_logger.warning(
            "dataset_insight_parse_failed",
            extra={"error": str(je), "raw_preview": raw_text[:300]},
        )
        return _build_fallback_dataset_insight(
            dataset_dict=dataset_dict,
            score_breakdown=score_breakdown,
            rank=rank,
            query=query,
        )

    def safe_str(v: Any, default: str) -> str:
        return str(v).strip() if v and str(v).strip() else default

    def safe_list(v: Any, default: str) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()][:5]
        return [default]

    return DatasetInsight(
        dataset_id=dataset_id,
        rank=rank,
        why_ranked=safe_str(data.get("why_ranked"), "Ranked based on evaluator score alignment."),
        strengths=safe_list(data.get("strengths"), "Covers required domain"),
        weaknesses=safe_list(data.get("weaknesses"), "Review before production use"),
        metadata_gaps=detect_metadata_gaps(dataset_dict),
        usability_notes=safe_list(data.get("usability_notes"), "Perform EDA before training"),
        score_narrative=safe_str(data.get("score_narrative"), "Scored across task relevance, quality, popularity, freshness, and query match."),
        confidence_note=safe_str(data.get("confidence_note"), "Confidence reflects score gap between top results."),
        is_llm_generated=True,
        fallback_used=False,
        generated_at=datetime.now(tz=timezone.utc),
    )


def _parse_research_context_response(
    raw_text: str,
    query: str,
    top_datasets: list[dict[str, Any]],
    total_candidates: int,
    confidence_level: str,
) -> ResearchContext:
    """Parse LLM response into ResearchContext. Falls back on any error."""
    text = _strip_json_fences(raw_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as je:
        _parse_logger.warning(
            "research_context_parse_failed",
            extra={"error": str(je), "raw_preview": raw_text[:300]},
        )
        return _build_fallback_research_context(
            query=query,
            top_datasets=top_datasets,
            total_candidates=total_candidates,
            confidence_level=confidence_level,
        )

    def safe_str(v: Any, default: str) -> str:
        return str(v).strip() if v and str(v).strip() else default

    def safe_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()][:5]
        return []

    return ResearchContext(
        query=query,
        ecosystem_summary=safe_str(data.get("ecosystem_summary"), f"Found datasets for '{query}'."),
        coverage_observations=safe_list(data.get("coverage_observations")),
        quality_trends=safe_list(data.get("quality_trends")),
        follow_up_searches=safe_list(data.get("follow_up_searches")),
        research_gaps=safe_list(data.get("research_gaps")),
        provider_notes=safe_list(data.get("provider_notes")),
        total_candidates=total_candidates,
        confidence_level=confidence_level,
        is_llm_generated=True,
        fallback_used=False,
        generated_at=datetime.now(tz=timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(size_bytes: Optional[int]) -> str:
    if not size_bytes:
        return "unknown"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    return f"{size_bytes / 1024 ** 3:.2f} GB"