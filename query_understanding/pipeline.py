"""
datascout.query_understanding.pipeline
────────────────────────────────────────
QueryUnderstandingPipeline — the single unified entry point that orchestrates
all query understanding components (cleaner → parser → enricher) in one call.

WHY THIS FILE EXISTS:
  search_v2.py needs one import point:

      from datascout.query_understanding.pipeline import QueryUnderstandingPipeline
      qu = QueryUnderstandingPipeline()
      enrichment = await qu.process(raw_query)
      expanded_query = enrichment.expanded_query

  Previously each stage (clean, parse, enrich) was called separately.
  This module wraps them so search_v2.py stays clean and the stages remain
  independently testable.

PIPELINE FLOW:
  1. QueryParser.parse(raw_query)
     → cleaned_text, task_type, modality, keywords, confidence
  2. QueryEnricher.enrich(cleaned_text)
     → expanded_query, semantic_hints, provider_weights, domains_detected

OUTPUT (QueryPipelineResult):
  expanded_query   — enriched text for BM25/adapter calls         (str)
  task_type        — detected ML task, None if UNKNOWN             (Optional[TaskType])
  modality         — detected data modality, None if UNKNOWN       (Optional[Modality])
  keywords         — clean keyword list from QueryParser           (list[str])
  provider_weights — per-source boost hints from enricher          (dict)
  expansion_terms  — terms added by the enricher                   (list[str])
  domains_detected — semantic domains matched                      (list[str])
  parse_result     — full QueryParseResult for diagnostics         (Optional)
  original_query   — original raw user string                      (str)

RESILIENCE:
  - Never raises to caller — wraps everything in try/except
  - On any failure returns a minimal result with expanded_query = raw_query
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from datascout.query_understanding.enricher import EnricherOutput, QueryEnricher
from datascout.query_understanding.parser import QueryParseResult, QueryParser
from datascout.query_understanding.task_types import Modality, TaskType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryPipelineResult:
    """
    Combined output of the full query understanding pipeline.

    This is the object that ``search_v2.py`` receives from
    ``QueryUnderstandingPipeline.process()`` and passes downstream to:
      - Adapter search calls (``expanded_query``)
      - EvaluatorPipeline (``task_type``, ``modality``, ``keywords``)
    """

    # ── Core field — always populated ────────────────────────────────────────
    expanded_query: str
    """Enriched search text — BM25/adapter queries use this, not raw input."""

    # ── For EvaluatorPipeline ─────────────────────────────────────────────────
    task_type: Optional[TaskType] = None
    """Detected ML task type.  None if QueryParser confidence is too low."""

    modality: Optional[Modality] = None
    """Detected data modality.  None if QueryParser confidence is too low."""

    keywords: list[str] = field(default_factory=list)
    """Clean keyword list (stop-words removed) from QueryParser."""

    # ── Provider routing ──────────────────────────────────────────────────────
    provider_weights: dict[str, float] = field(
        default_factory=lambda: {"huggingface": 1.0, "kaggle": 1.0, "openml": 1.0}
    )
    """
    Per-source score multipliers from QueryEnricher.
    Adapters can use these to prioritise or de-prioritise specific sources.
    Default is 1.0 for all (no bias).
    """

    # ── Enrichment metadata ───────────────────────────────────────────────────
    expansion_terms: list[str] = field(default_factory=list)
    """Terms added by the enricher (for explainability/debugging)."""

    domains_detected: list[str] = field(default_factory=list)
    """Semantic domains that triggered enrichment, e.g. ``['speech recognition']``."""

    # ── Full sub-results for diagnostics ─────────────────────────────────────
    parse_result: Optional[QueryParseResult] = None
    """Full output from QueryParser — preserved for diagnostics."""

    enricher_output: Optional[EnricherOutput] = None
    """Full output from QueryEnricher — preserved for diagnostics."""

    # ── Original query ────────────────────────────────────────────────────────
    original_query: str = ""
    """Verbatim raw user input — for display and logging."""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────


class QueryUnderstandingPipeline:
    """
    Orchestrates query cleaning, parsing, and enrichment.

    Designed for use from async FastAPI route handlers.  The actual
    processing is CPU-bound (pure Python regex/dict lookups), so
    ``process()`` runs synchronous logic inside ``run_in_executor`` to
    avoid blocking the event loop.

    Usage::

        pipeline = QueryUnderstandingPipeline()
        enrichment = await pipeline.process("crop disease image classification")
        # enrichment.expanded_query == "crop disease image classification plant pathology..."
        # enrichment.task_type     == TaskType.IMAGE_CLASSIFICATION
        # enrichment.keywords      == ["crop", "disease", "image", "classification"]
    """

    def __init__(self) -> None:
        """
        Instantiate pipeline components.

        Components are cheap to instantiate (no model loading) so it is
        fine to create a new pipeline per request.
        """
        self._parser: QueryParser = QueryParser()
        self._enricher: QueryEnricher = QueryEnricher()

    # ──────────────────────────────────────────────────────────────────────────
    # Public async entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def process(self, raw_query: str) -> QueryPipelineResult:
        """
        Run the full query understanding pipeline asynchronously.

        Runs in a thread-pool executor so the event loop is never blocked.
        Never raises — returns a minimal fallback result on any error.

        Parameters
        ----------
        raw_query:
            Raw user query string as submitted.  May contain typos,
            domain jargon, or mixed-language tokens.

        Returns
        -------
        QueryPipelineResult:
            Always valid and non-None.  On internal failure the result
            has ``expanded_query == raw_query`` and all other fields
            at their defaults (None / empty).
        """
        if not raw_query or not raw_query.strip():
            return QueryPipelineResult(
                expanded_query="",
                original_query=raw_query or "",
            )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._process_sync, raw_query)
            return result
        except Exception as exc:
            logger.warning(
                "query_pipeline_error",
                extra={"error": str(exc)[:200], "query": raw_query[:80]},
            )
            return QueryPipelineResult(
                expanded_query=raw_query,
                original_query=raw_query,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Synchronous pipeline — runs inside executor
    # ──────────────────────────────────────────────────────────────────────────

    def _process_sync(self, raw_query: str) -> QueryPipelineResult:
        """
        Execute all pipeline stages synchronously.

        This is the implementation called from the executor.  It is
        intentionally kept free of any async primitives.

        Parameters
        ----------
        raw_query:
            Raw user input string.

        Returns
        -------
        QueryPipelineResult:
            Populated result.  Never raises.
        """
        # ── Stage 1: Parse ────────────────────────────────────────────────────
        parse_result: Optional[QueryParseResult] = None
        cleaned_text: str = raw_query
        task_type: Optional[TaskType] = None
        modality: Optional[Modality] = None
        keywords: list[str] = []

        try:
            parse_result = self._parser.parse(raw_query)
            cleaned_text = parse_result.cleaned_text or raw_query
            keywords = list(parse_result.keywords or [])

            # Only propagate task/modality if confidence is non-trivial
            if parse_result.has_task:
                task_type = parse_result.task_type
            if parse_result.has_modality:
                modality = parse_result.modality

            logger.debug(
                "query_parser_done",
                extra={
                    "task": task_type.value if task_type else "unknown",
                    "modality": modality.value if modality else "unknown",
                    "keywords": keywords[:5],
                },
            )
        except Exception as exc:
            logger.warning(
                "query_parser_failed",
                extra={"error": str(exc)[:120], "query": raw_query[:80]},
            )
            # Keep defaults — continue to enrichment with raw query

        # ── Stage 2: Enrich ───────────────────────────────────────────────────
        enricher_out: Optional[EnricherOutput] = None
        expanded_query: str = cleaned_text

        try:
            enricher_out = self._enricher.enrich(cleaned_text)
            expanded_query = enricher_out.expanded_query or cleaned_text

            logger.debug(
                "query_enricher_done",
                extra={
                    "domains": enricher_out.domains_detected,
                    "terms_added": len(enricher_out.expansion_terms),
                    "expanded": expanded_query[:80],
                },
            )
        except Exception as exc:
            logger.warning(
                "query_enricher_failed",
                extra={"error": str(exc)[:120], "query": cleaned_text[:80]},
            )
            # Keep defaults — return parsed result without enrichment

        # ── Stage 3: Compose result ───────────────────────────────────────────
        return QueryPipelineResult(
            expanded_query=expanded_query,
            task_type=task_type,
            modality=modality,
            keywords=keywords,
            provider_weights=dict(enricher_out.provider_weights) if enricher_out else {
                "huggingface": 1.0,
                "kaggle": 1.0,
                "openml": 1.0,
            },
            expansion_terms=list(enricher_out.expansion_terms) if enricher_out else [],
            domains_detected=list(enricher_out.domains_detected) if enricher_out else [],
            parse_result=parse_result,
            enricher_output=enricher_out,
            original_query=raw_query,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "QueryUnderstandingPipeline",
    "QueryPipelineResult",
]