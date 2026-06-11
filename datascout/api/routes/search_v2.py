"""
datascout.api.routes.search_v2
────────────────────────────────
Main search pipeline router — V2.

PIPELINE ORDER (per request):
  1. Parse + validate SearchQuery
  2. Query understanding (clean, expand, detect task/modality)
  3. ScoutAgent — agentic retrieval with retry + broadening (Task 14)
     ↳ Fast-path: Elasticsearch hybrid search if connected (Task 15)
     ↳ Fallback: direct adapter calls (Kaggle, HuggingFace, OpenML)
  4. Deduplication (SHA-256 fingerprint)
  5. EvaluatorPipeline — score + rank + diagnose
  6. Analysis engine (quality, summary, target detection)
  7. Serialize results — includes ranking explanation (Task 13)
  8. Return JSON response

RESILIENCY:
  - Every stage is wrapped in try/except
  - ScoutAgent records per-source failures as human-friendly messages
  - _no_results_response returns specific per-source error messages (Task 16)
  - Elasticsearch init is optional — falls back to adapters transparently
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from datascout.contracts.requests import SearchQuery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["search"])

# ── Module-level singletons (Task 15) ─────────────────────────────────────────
# Initialised at startup via _init_elasticsearch(); None means ES not available.
_elastic_client: Optional[Any] = None

# ── Gemini intelligence cache ──────────────────────────────────────────────────
# Caches intelligence results by query hash for 5 minutes.
# Prevents re-calling Gemini on repeated identical searches (rate limit fix).
import time as _time
_intel_cache: dict = {}          # {query_hash: (result, timestamp)}
_INTEL_CACHE_TTL: float = 300.0  # 5 minutes
_embedding_engine: Optional[Any] = None  # EmbeddingEngine for query vectors


# ══════════════════════════════════════════════════════════════════════════════
# Task 15 — Elasticsearch initialisation
# ══════════════════════════════════════════════════════════════════════════════

async def _init_elasticsearch(settings: Any) -> None:
    """Initialise ElasticsearchEngine + EmbeddingEngine at application startup.

    Called from ``api/main.py`` lifespan after the database is ready.  Sets
    the module-level ``_elastic_client`` and ``_embedding_engine`` globals so
    that subsequent search requests can use hybrid ES search.

    If ``settings.elastic_enabled`` is ``False`` (the default), returns
    immediately.  All errors are caught and logged as warnings so that a
    misconfigured ES cluster never prevents the API from starting.

    Parameters
    ----------
    settings:
        Application settings object — must expose ``elastic_enabled``,
        ``elasticsearch_url``, ``elasticsearch_index``,
        ``elasticsearch_api_key``, and ``embedding_model``.
    """
    global _elastic_client, _embedding_engine

    if not getattr(settings, "elastic_enabled", False):
        logger.info("elasticsearch_disabled_in_settings")
        return

    url: Optional[str] = getattr(settings, "elasticsearch_url", None)
    if not url:
        logger.warning("elasticsearch_enabled_but_no_url")
        return

    try:
        from datascout.search.elasticsearch_engine import ElasticsearchEngine
        from datascout.search.embedding_engine import EmbeddingEngine

        engine = ElasticsearchEngine(
            url=url,
            index=getattr(settings, "elasticsearch_index", "datascout-datasets"),
            api_key=getattr(settings, "elasticsearch_api_key", None),
        )
        connected = await engine.connect()
        if connected:
            _elastic_client = engine
            logger.info(
                "elasticsearch_connected",
                extra={
                    "url": url,
                    "index": getattr(settings, "elasticsearch_index", "?"),
                },
            )
        else:
            logger.warning("elasticsearch_connect_failed", extra={"url": url})

        model_name: str = getattr(settings, "embedding_model", "all-MiniLM-L6-v2")
        emb = EmbeddingEngine(model_name=model_name)
        emb.load_model()
        _embedding_engine = emb
        logger.info("embedding_engine_loaded", extra={"model": model_name})

    except Exception as exc:
        logger.warning(
            "elasticsearch_init_error",
            extra={"error": str(exc)[:200]},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _deduplicate(datasets: List[Any]) -> List[Any]:
    """Remove duplicate datasets by ``dataset_fingerprint`` then ``canonical_id``.

    Parameters
    ----------
    datasets:
        Flat list of ``RawDataset`` objects.

    Returns
    -------
    list[RawDataset]:
        Deduplicated list — first occurrence wins.
    """
    seen_fingerprints: set[str] = set()
    seen_ids: set[str] = set()
    out: List[Any] = []
    for ds in datasets:
        fp = getattr(ds, "dataset_fingerprint", None)
        cid = getattr(ds, "canonical_id", None)
        if fp and fp in seen_fingerprints:
            continue
        if cid and cid in seen_ids:
            continue
        if fp:
            seen_fingerprints.add(fp)
        if cid:
            seen_ids.add(cid)
        out.append(ds)
    return out


def _confidence_label(scored_count: int, top_score: float) -> str:
    """Derive a simple confidence label from result count and top score."""
    if scored_count == 0:
        return "low"
    if top_score >= 0.70 and scored_count >= 3:
        return "high"
    if top_score >= 0.45 or scored_count >= 2:
        return "medium"
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
# Task 13 — _serialize() with ranking explanation wired in
# ══════════════════════════════════════════════════════════════════════════════

def _serialize(
    ds: Any,
    score: float,
    breakdown: Dict[str, float],
    rank: int,
    diagnostics: Any = None,
    analysis: Any = None,
) -> Dict[str, Any]:
    """Serialise a ``ScoredDataset`` (or bare ``RawDataset``) to an API dict.

    Parameters
    ----------
    ds:
        A ``ScoredDataset`` (has ``.dataset``, ``.breakdown``,
        ``.composite_score``) or a bare ``RawDataset`` (legacy path).
    score:
        Composite float score in [0, 1].
    breakdown:
        Score dimension dict, e.g.
        ``{"task_relevance": 0.8, "quality": 0.7, ...}``.
    rank:
        1-based rank position.
    diagnostics:
        Optional ``DatasetDiagnostics`` from ``EvaluatorPipeline``.
    analysis:
        Optional analysis result dict from the analysis engine.

    Returns
    -------
    dict:
        Fully populated response dict for one dataset result.
    """
    # Support ScoredDataset wrapper and bare RawDataset transparently
    raw_ds = ds.dataset if hasattr(ds, "dataset") else ds
    g = lambda attr, default=None: getattr(raw_ds, attr, default)  # noqa: E731

    def _enum_val(v: Any) -> Any:
        return v.value if hasattr(v, "value") else v

    out: Dict[str, Any] = {
        "rank":               rank,
        "canonical_id":       g("canonical_id", ""),
        "title":              g("title", ""),
        "description":        g("description", ""),
        "source":             g("source", ""),
        "source_url":         g("source_url", ""),
        "tags":               g("tags") or [],
        "task_types":         [_enum_val(t) for t in (g("task_types") or [])],
        "modalities":         [_enum_val(m) for m in (g("modalities") or [])],
        "row_count":          g("row_count"),
        "column_count":       g("column_count"),
        "column_names":       g("column_names"),
        "file_size_bytes":    g("file_size_bytes"),
        "download_count":     g("download_count"),
        "upvote_count":       g("upvote_count"),
        "author":             g("author"),
        "license_type":       _enum_val(g("license_type")),
        "last_updated":       (
            g("last_updated").isoformat() if g("last_updated") is not None else None
        ),
        "metadata_completeness": g("metadata_completeness", 0.0),
        "composite_score":    round(score, 4),
        "score_breakdown":    breakdown,
    }

    # ── Task 13 — Ranking explanation ─────────────────────────────────────────
    try:
        from datascout.evaluation.explainability import build_ranking_explanation

        if hasattr(ds, "breakdown"):
            expl = build_ranking_explanation(
                scored=ds,
                diagnostics=diagnostics,
                rank=rank,
                total_candidates=0,
                analysis=analysis,  # deep analysis signals flow in here
            )
            out["why_ranked_here"] = expl.why_selected
            out["strengths"]       = expl.strengths
            out["weaknesses"]      = expl.weaknesses
            out["bias_warnings"]   = [
                b.description if hasattr(b, "description") else str(b)
                for b in (expl.bias_warnings or [])
            ]
            out["score_labels"]    = expl.score_labels
            out["quality_tier"]    = expl.quality_tier
        else:
            score_pct = round(score * 100, 1)
            out["why_ranked_here"] = (
                f"Ranked #{rank} with a composite score of {score_pct}%."
            )
            out["strengths"]    = []
            out["weaknesses"]   = []
            out["bias_warnings"] = []
            out["score_labels"] = {}
            out["quality_tier"] = "incomplete"

    except Exception:
        out["why_ranked_here"] = f"Ranked #{rank} with composite score {round(score, 3)}."
        out["strengths"]       = []
        out["weaknesses"]      = []
        out["bias_warnings"]   = []
        out["score_labels"]    = {}
        out["quality_tier"]    = "incomplete"

    # ── Diagnostics extras ─────────────────────────────────────────────────────
    if diagnostics is not None:
        try:
            freshness_diag = getattr(diagnostics, "freshness", None)
            out["freshness_days"] = getattr(freshness_diag, "days_since_update", None)
            out["freshness_explanation"] = getattr(freshness_diag, "explanation", "")

            popularity_diag = getattr(diagnostics, "popularity", None)
            out["popularity_explanation"] = getattr(popularity_diag, "explanation", "")

            quality_diag = getattr(diagnostics, "quality", None)
            bias_signals = getattr(quality_diag, "bias_signals", []) or []
            # Merge bias_signals into bias_warnings if not already populated
            if bias_signals and not out.get("bias_warnings"):
                out["bias_warnings"] = [
                    b.description if hasattr(b, "description") else str(b)
                    for b in bias_signals
                ]

        except Exception:
            pass

    # ── Analysis data ──────────────────────────────────────────────────────────
    if analysis is not None:
        try:
            out["analysis"] = analysis
        except Exception:
            pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# ── Gemini no-results explainer ───────────────────────────────────────────────

def _gemini_no_results_message(query: str, sources_searched: list) -> str:
    """
    Deterministic no-results message. Previously called Gemini for this —
    removed to preserve free-tier quota (15 RPM) for the single intelligence
    call that explains why the top-3 datasets were selected.

    NOTE: This function is now synchronous. Callers that previously awaited it
    should call it directly without await. The async wrapper below preserves
    backward compatibility with call sites that use `await`.
    """
    sources_str = ", ".join(s.title() for s in sources_searched) if sources_searched else "Kaggle, HuggingFace, and OpenML"
    words = query.lower().split()
    # Build two alternative phrasings: shorter core + task-type hint
    core_terms = [w for w in words if len(w) > 3][:3]
    shorter = " ".join(core_terms) if core_terms else query
    # Detect likely task type from query for a smarter suggestion
    _task_hints = {
        ("classif", "recognit", "detect"): "classification",
        ("generat", "caption", "synthes"): "generation",
        ("segmen", "mask", "pixel"): "segmentation",
        ("translat", "bilingual", "parallel"): "translation",
        ("speech", "audio", "asr", "wav"): "speech recognition",
        ("regress", "predict", "forecast"): "regression",
    }
    task_hint = "classification"
    for keywords, label in _task_hints.items():
        if any(k in query.lower() for k in keywords):
            task_hint = label
            break

    logger.info("no_results_message_generated", extra={"query": query[:60]})
    return (
        f"No datasets were found for '{query}' across {sources_str}. "
        f"Try broader or alternative terms — for example, '{shorter} data' "
        f"or the specific task type (e.g. '{task_hint}'). "
        f"Also verify your API credentials are set in the .env file."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Task 16 — _no_results_response() with agent trace
# ══════════════════════════════════════════════════════════════════════════════

async def _no_results_response(
    query: str,
    enrichment: Any,
    t_start: float,
    agent_trace: Any = None,
) -> Dict[str, Any]:
    """Build a structured no-results API response.

    When *agent_trace* is supplied and all adapters failed, returns a
    specific per-source failure message in plain language without any
    technical jargon.  Falls back to a generic suggestion otherwise.

    Parameters
    ----------
    query:
        Original user query string.
    enrichment:
        Query enrichment object (may be None if enrichment failed).
    t_start:
        ``time.monotonic()`` at request start — used for latency calc.
    agent_trace:
        Optional ``AgentTrace`` from ScoutAgent.  If all sources failed,
        a specific per-source message is returned immediately.

    Returns
    -------
    dict:
        API response with ``results: []`` and a human-friendly message.
    """
    processing_ms = int((time.monotonic() - t_start) * 1_000)

    # ── All sources failed: specific per-source message ───────────────────────
    if agent_trace is not None:
        failures = getattr(agent_trace, "adapter_failures", [])
        succeeded = getattr(agent_trace, "sources_succeeded", [])

        if failures and not succeeded:
            issue_parts: List[str] = []
            seen_sources: set[str] = set()
            for f in failures:
                if f.source in seen_sources:
                    continue
                seen_sources.add(f.source)

                if f.reason == "auth_failed":
                    issue_parts.append(
                        f"{f.source.title()}: credentials not configured"
                    )
                elif f.reason == "not_installed":
                    issue_parts.append(
                        f"{f.source.title()}: required package not installed"
                    )
                elif f.reason == "rate_limited":
                    issue_parts.append(
                        f"{f.source.title()}: rate limit reached — try again shortly"
                    )
                elif f.reason == "timeout":
                    issue_parts.append(
                        f"{f.source.title()}: service took too long to respond"
                    )
                else:
                    issue_parts.append(
                        f"{f.source.title()}: temporarily unavailable"
                    )

            friendly = (
                f"We searched HuggingFace, Kaggle, and OpenML for '{query}' "
                f"but couldn't connect right now. "
                f"Issues: {'; '.join(issue_parts)}. "
                f"Check your API credentials in the .env file and try again."
            )
            return {
                "query":                 query,
                "results":               [],
                "total_found":           0,
                "returned":              0,
                "confidence":            "low",
                "intelligence_available": False,
                "message":               friendly,
                "agent_message":         friendly,
                "sources_used":          list(succeeded),
                "adapter_failures": [
                    {"source": f.source, "reason": f.reason}
                    for f in failures
                ],
                "processing_time_ms": processing_ms,
            }

        # Some succeeded but returned nothing — Gemini explains humanly
        if succeeded:
            friendly = _gemini_no_results_message(query, list(succeeded))
            return {
                "query":                 query,
                "results":               [],
                "total_found":           0,
                "returned":              0,
                "confidence":            "low",
                "intelligence_available": True,
                "message":               friendly,
                "agent_message":         friendly,
                "sources_used":          list(succeeded),
                "adapter_failures": [
                    {"source": f.source, "reason": f.reason}
                    for f in failures
                ],
                "processing_time_ms": processing_ms,
            }

    # ── Generic fallback — Gemini explains humanly ────────────────────────────
    friendly = _gemini_no_results_message(query, [])
    return {
        "query":                 query,
        "results":               [],
        "total_found":           0,
        "returned":              0,
        "confidence":            "low",
        "intelligence_available": True,
        "message":               friendly,
        "agent_message":         friendly,
        "sources_used":          [],
        "adapter_failures":      [],
        "processing_time_ms":    processing_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task 14 — _run_pipeline() with ScoutAgent retrieval
# ══════════════════════════════════════════════════════════════════════════════

async def _run_pipeline(
    raw_query: str,
    max_results: int,
    request_id: str,
) -> Dict[str, Any]:
    """Execute the full search pipeline for one query.

    Parameters
    ----------
    raw_query:
        Raw user query string as submitted.
    max_results:
        Maximum number of ranked results to return.
    request_id:
        UUID for this request (for log correlation).

    Returns
    -------
    dict:
        Complete API response dict.  Never raises — errors are caught and
        surfaced as user-friendly messages.
    """
    t_start = time.monotonic()
    sources = ["kaggle", "huggingface", "openml"]

    # ── 1. Query understanding ────────────────────────────────────────────────
    enrichment: Any = None
    expanded_query: str = raw_query
    try:
        from datascout.query_understanding.pipeline import QueryUnderstandingPipeline  # noqa: PLC0415
        qu = QueryUnderstandingPipeline()
        enrichment = await qu.process(raw_query)
        expanded_query = getattr(enrichment, "expanded_query", raw_query) or raw_query
        logger.info(
            "query_understood",
            extra={
                "request_id": request_id,
                "original": raw_query[:80],
                "expanded": expanded_query[:80],
            },
        )
    except Exception as exc:
        logger.warning(
            "query_understanding_failed",
            extra={"request_id": request_id, "error": str(exc)[:120]},
        )

    # ── 2. Retrieval — adapters first, then ES hybrid re-ranking ─────────────
    raw_datasets: List[Any] = []
    agent_trace: Any = None
    agent_status_message: str = ""
    retrieval_method: str = "adapter_direct"

    try:
        # ── 2a. Always fetch fresh candidates via ScoutAgent ──────────────────
        from datascout.agents.scout_agent import ScoutAgent  # noqa: PLC0415
        agent = ScoutAgent()
        raw_datasets, agent_trace = await agent.search(
            query=expanded_query,
            sources=sources,
            max_results=max_results,
        )
        logger.info(
            "agent_retrieval_done",
            extra={
                "request_id": request_id,
                "total":      len(raw_datasets),
                "iterations": agent_trace.iterations,
                "broadened":  agent_trace.broadening_applied,
                "failures":   len(agent_trace.adapter_failures),
                "sources_ok": agent_trace.sources_succeeded,
            },
        )
        agent_status_message = agent_trace.to_user_message()

        # ── 2b. Elasticsearch hybrid re-ranking — ACTIVE on every query ───────
        # ES is not a cache — it actively re-ranks adapter candidates using
        # BM25 + dense vector (kNN) Reciprocal Rank Fusion.  This is the core
        # differentiator: semantically irrelevant results (high downloads but
        # wrong topic) are pushed down; semantically aligned results rise.
        es_connected = (
            _elastic_client is not None
            and getattr(_elastic_client, "_connected", False)
            and _embedding_engine is not None
        )
        if es_connected and raw_datasets:
            try:
                logger.info(
                    "elasticsearch_hybrid_rerank_start",
                    extra={
                        "candidates": len(raw_datasets),
                        "query":      expanded_query[:60],
                        "request_id": request_id,
                    },
                )

                # Encode all candidate texts for indexing
                corpus_texts: List[str] = [
                    f"{getattr(ds, 'title', '')} "
                    f"{getattr(ds, 'description', '')[:200]} "
                    f"{' '.join(getattr(ds, 'tags', [])[:8])}"
                    for ds in raw_datasets
                ]
                candidate_embeddings = _embedding_engine.encode_batch(corpus_texts)

                # Index candidates (with embeddings) into ES — synchronous
                # in-request so re-ranking has the latest data
                indexed_count = await _elastic_client.index_datasets(
                    raw_datasets,
                    embeddings=[
                        e if e is not None else [0.0] * 384
                        for e in candidate_embeddings
                    ],
                )

                # Encode the search query for kNN retrieval
                query_emb = _embedding_engine.encode(expanded_query)

                if indexed_count > 0 and query_emb:
                    # Hybrid BM25 + kNN RRF re-rank
                    es_results = await _elastic_client.hybrid_search(
                        query=expanded_query,
                        query_embedding=query_emb,
                        top_k=max_results * 3,
                    )
                    if es_results:
                        raw_datasets = es_results
                        retrieval_method = "elasticsearch_hybrid"
                        logger.info(
                            "elasticsearch_hybrid_rerank_complete",
                            extra={
                                "returned":    len(es_results),
                                "indexed":     indexed_count,
                                "request_id":  request_id,
                                "mode":        "bm25+knn+rrf",
                            },
                        )
                    else:
                        logger.info(
                            "elasticsearch_rerank_empty_keep_adapter_order",
                            extra={"request_id": request_id},
                        )
                else:
                    logger.info(
                        "elasticsearch_skip_no_embeddings",
                        extra={"indexed": indexed_count, "request_id": request_id},
                    )

            except Exception as es_exc:
                logger.warning(
                    "elasticsearch_rerank_error",
                    extra={"error": str(es_exc)[:200], "request_id": request_id},
                )
                # Keep adapter results — ES failure is non-fatal

        elif not es_connected and raw_datasets:
            # ES not running — log clearly so operator knows
            logger.info(
                "elasticsearch_not_connected_adapter_order_used",
                extra={"hint": "Set ELASTIC_ENABLED=true + ELASTICSEARCH_URL to enable hybrid re-ranking"},
            )

    except Exception as exc:
        logger.error(
            "retrieval_stage_error",
            extra={"request_id": request_id, "error": str(exc)[:200]},
        )

    # ── 3. No results ─────────────────────────────────────────────────────────
    if not raw_datasets:
        return await _no_results_response(
            query=raw_query,
            enrichment=enrichment,
            t_start=t_start,
            agent_trace=agent_trace,
        )

    # ── 4. Deduplication ──────────────────────────────────────────────────────
    raw_datasets = _deduplicate(raw_datasets)
    logger.info(
        "deduplication_done",
        extra={"request_id": request_id, "count": len(raw_datasets)},
    )

    # ── 5. EvaluatorPipeline — score + rank + diagnose ────────────────────────
    eval_result: Any = None
    diagnostics_map: Dict[str, Any] = {}
    try:
        from datascout.evaluation.pipeline import EvaluatorPipeline  # noqa: PLC0415
        pipeline = EvaluatorPipeline()
        eval_result = await pipeline.run(
            datasets=raw_datasets,
            query=expanded_query,
            enrichment=enrichment,
        )
        # Build diagnostics lookup by canonical_id
        for diag in getattr(eval_result, "diagnostics", []) or []:
            cid = getattr(diag, "canonical_id", None)
            if cid:
                diagnostics_map[cid] = diag
    except Exception as exc:
        logger.warning(
            "evaluator_pipeline_failed",
            extra={"request_id": request_id, "error": str(exc)[:200]},
        )

    # ── 6. Build ranked list ──────────────────────────────────────────────────
    ranked_datasets: List[Any] = []
    if eval_result is not None:
        ranked_datasets = getattr(eval_result, "ranked", []) or []
    else:
        # Fallback: return raw datasets with zero scores
        from datascout.evaluation.scorer import ScoredDataset, ScoreBreakdown  # noqa: PLC0415
        for ds in raw_datasets:
            bd = ScoreBreakdown(
                task_relevance=0.0, quality=0.0, popularity=0.0,
                freshness=0.0, description_match=0.0, composite=0.0,
            )
            ranked_datasets.append(
                ScoredDataset(dataset=ds, breakdown=bd, composite_score=0.0)
            )

    # ── 6b. Hard relevance gate ───────────────────────────────────────────────
    # Skip gate entirely when Elasticsearch already re-ranked — ES hybrid search
    # (BM25 + kNN + RRF) already guarantees relevance. Applying a keyword gate
    # on top of ES results discards good results with low exact keyword overlap.
    if ranked_datasets and retrieval_method != "elasticsearch_hybrid":
        from datascout.evaluation.scorer import DESCRIPTION_MATCH_GATE  # noqa: PLC0415
        # FIX: Gate lowered to match new DESCRIPTION_MATCH_GATE (0.10).
        # REASON: The old 0.25 gate here was redundant with the scorer gate and
        # was blocking legitimate results whose titles don't exactly repeat the query.
        # All-irrelevant check now only fires for truly zero-match garbage results.
        RELEVANCE_GATE = max(DESCRIPTION_MATCH_GATE, 0.05)  # 1 keyword out of 20 must match
        TOP_SCORE_GATE = 0.05  # composite score must exceed 5% — near-zero means truly junk

        top_dataset = ranked_datasets[0]
        top_composite = float(getattr(top_dataset, "composite_score", 0.0) or 0.0)
        top_bd = getattr(top_dataset, "breakdown", None)
        top_desc_match = float(getattr(top_bd, "description_match", 0.0) or 0.0) if top_bd else 0.0

        all_irrelevant = all(
            (float(getattr(getattr(sd, "breakdown", None), "description_match", 0.0) or 0.0) < RELEVANCE_GATE)
            for sd in ranked_datasets[:5]  # check top-5 only for speed
        )

        if all_irrelevant and top_composite < TOP_SCORE_GATE:
            logger.info(
                "relevance_gate_triggered",
                extra={
                    "request_id":     request_id,
                    "top_composite":  top_composite,
                    "top_desc_match": top_desc_match,
                    "candidates":     len(ranked_datasets),
                    "query":          raw_query[:60],
                },
            )
            # Return Gemini-powered human-friendly no-results instead of garbage
            sources_used = list(getattr(agent_trace, "sources_succeeded", []) or []) if agent_trace else []
            return await _no_results_response(
                query=raw_query,
                enrichment=enrichment,
                t_start=t_start,
                agent_trace=type("_FakeTrace", (), {
                    "adapter_failures": [],
                    "sources_succeeded": sources_used,
                })(),
            )

    # ── 7. Analysis ───────────────────────────────────────────────────────────
    # TWO-LAYER ANALYSIS:
    #   Layer 1 (fast, always runs): QualityScorer + SummaryGenerator + TargetDetector
    #            — metadata only, <1 ms per dataset, deterministic
    #   Layer 2 (deep, async):       AnalysisEngine.analyze()
    #            — downloads sample data, checks class balance, duplicates,
    #              missing values, image quality, geographic bias, and produces
    #              the composite quality_score + suggested_fixes list
    # Both layers run concurrently across all datasets (asyncio.gather).
    # A 28-second outer timeout keeps search response times acceptable even if
    # some source_urls are slow; AnalysisEngine itself has its own per-dataset
    # 20-second timeout and never raises — partial results are always returned.
    analysis_map: Dict[str, Any] = {}
    try:
        from datascout.analysis.quality_scorer import QualityScorer  # noqa: PLC0415
        from datascout.analysis.summary_generator import SummaryGenerator  # noqa: PLC0415
        from datascout.analysis.target_detector import TargetDetector  # noqa: PLC0415
        from datascout.analysis.analysis_engine import create_analysis_engine  # noqa: PLC0415

        qs = QualityScorer()
        sg = SummaryGenerator()
        td = TargetDetector()

        # Deep analysis engine — generous timeout per dataset for image downloads
        # Top 5 get deep analysis (real download + quality checks)
        # Remaining 5 get fast metadata-only analysis
        DEEP_ANALYSIS_COUNT = 2     # FIX v3.8.0: top-2 only — correct candidates after re-rank
        # FIX v3.8.0: PER_DATASET_TIMEOUT raised 300s → 600s for large Kaggle image downloads.
        # Deep analysis runs on the TRUE top-2 after light-analysis re-rank (not before).
        # 2 tasks run simultaneously via asyncio.gather (parallel, not serial).
        # DEEP_BATCH_TIMEOUT = 2 × 600s = 1200s (both run concurrently so elapsed ≈ 600s).
        # NOTE: these timeouts apply to deep analysis only (top 2 datasets).
        # Light analysis still uses 10s per dataset (analysis_engine_meta below).
        PER_DATASET_TIMEOUT = 600.0  # 10 min per dataset — handles very large image zips
        DEEP_BATCH_TIMEOUT  = 1200.0 # 20 min ceiling; both tasks run in parallel ≈600s actual

        analysis_engine_deep = create_analysis_engine(
            sample_rows=50_000,      # FIX v3.7.0 (Issue 6): was 500 rows — far too low.
            timeout_seconds=PER_DATASET_TIMEOUT,
            download_images=True,    # full download: blur/dup/corrupt checks
            deep_mode=True,          # FIX v3.7.0 (Issue 6): raises image cap from 50→2000/class.
            # Root cause: analysis_engine_deep used the same MAX_IMAGES_PER_CLASS=50
            # as the fast/meta engine. For a 5676-image dataset, only ~1900 images were
            # analysed. deep_mode=True enables MAX_IMAGES_PER_CLASS_DEEP=2000/class.
        )
        analysis_engine_meta = create_analysis_engine(
            sample_rows=100,
            timeout_seconds=10.0,    # FIX: raised from 5s → 10s for slow connections
            download_images=False,   # use file-listing API, no download
        )

        async def _analyse_one(sd: Any, deep: bool = False) -> tuple:
            """
            Layer 1 (always) + Layer 2 (deep=True only).
            deep=True  → real data download, image checks, class balance, duplicates
            deep=False → metadata scores only, <100ms
            Never raises.
            """
            raw = sd.dataset if hasattr(sd, "dataset") else sd
            cid: Optional[str] = getattr(raw, "canonical_id", None)
            if not cid:
                return None, None
            try:
                # Layer 1: fast metadata
                q_score       = qs.score(raw)
                summary       = sg.generate(raw)
                col_names     = getattr(raw, "column_names", []) or []
                target_result = td.detect(col_names)

                # Layer 2: deep or metadata-only
                engine = analysis_engine_deep if deep else analysis_engine_meta
                report = await engine.analyze(raw)

                # ── Compute blurry_count from blur_result ─────────────────
                _blur_r = report.blur_result
                _blurry_count: Optional[int] = None
                if _blur_r and _blur_r.images_sampled > 0:
                    _blurry_count = round(
                        (_blur_r.blurry_percentage / 100) * _blur_r.images_sampled
                    )

                # ── Build null_columns with raw count (primary) + percentage ──
                _null_cols = []
                for _col in (report.column_result.columns if report.column_result else []):
                    if _col.missing_percentage > 5.0:
                        _null_cols.append({
                            "name":        _col.name,
                            "null_count":  _col.null_count,                    # raw count — primary
                            "missing_pct": round(_col.missing_percentage, 1),  # percentage — secondary
                        })

                # ── All columns detail (for low-info, type inconsistencies) ─
                _cols_detail = []
                for _col in (report.column_result.columns if report.column_result else []):
                    _cols_detail.append({
                        "name":             _col.name,
                        "type":             _col.col_type.value,
                        "null_count":       _col.null_count,
                        "missing_pct":      round(_col.missing_percentage, 1),
                        "unique_count":     _col.unique_count,
                        "total_count":      _col.total_count,
                        "cardinality_ratio": round(_col.cardinality_ratio, 3),
                        "is_id":            _col.is_likely_id,
                        "is_target":        _col.is_likely_target,
                        "top_categories": (
                            [(str(v), int(c)) for v, c in _col.top_categories[:5]]
                            if _col.top_categories else None
                        ),
                    })

                # ── Column type summary (tabular) ─────────────────────────────
                _col_r = report.column_result
                _columns_summary = {
                    "total":       _col_r.total_columns,
                    "numeric":     _col_r.numeric_columns,
                    "categorical": _col_r.categorical_columns,
                    "text":        _col_r.text_columns,
                    "missing_pct": round(_col_r.overall_missing_percentage, 1),
                } if _col_r else None

                # ── Geographic bias ───────────────────────────────────────────
                _geo = report.geographic_result
                _geo_bias = {
                    "detected": _geo.is_single_region if _geo else False,
                    "regions":  _geo.detected_regions  if _geo else [],
                    "warning":  _geo.warning_message   if _geo else None,
                }

                # ── Total images for image datasets ───────────────────────────
                _total_images: Optional[int] = None
                if report.balance_result and report.balance_result.class_distribution:
                    _total_images = sum(report.balance_result.class_distribution.values())

                # ── Dimension info (image datasets) ───────────────────────────
                _dim = report.dimension_result
                _dimension_info = {
                    "is_consistent":   _dim.is_consistent,
                    "dominant_size":   list(_dim.dominant_size) if _dim.dominant_size else None,
                    "unique_sizes":    _dim.unique_sizes,
                    "undersized_count": _dim.undersized_count,
                    "oversized_count":  _dim.oversized_count,
                    "warning":         _dim.warning_message,
                } if _dim else None

                return cid, {
                    "quality": {
                        "completeness":       round(q_score.completeness, 3),
                        "consistency":        round(q_score.consistency, 3),
                        "validity":           round(q_score.validity, 3),
                        "composite":          round(q_score.composite, 3),
                        "issues":             q_score.issues,
                        "deep_quality_score": report.quality_score,
                        "completeness_score": report.completeness_score,
                        "balance_score":      report.balance_score,
                        "uniqueness_score":   report.uniqueness_score,
                        "suggested_fixes":    report.suggested_fixes,
                        "analysis_partial":   report.is_partial,
                        "analysis_error":     report.error,

                        # ── IMAGE: class distribution ─────────────────────────
                        "class_distribution": (
                            report.balance_result.class_distribution
                            if report.balance_result else None
                        ),
                        "num_classes": (
                            report.balance_result.num_classes
                            if report.balance_result else None
                        ),
                        "total_images": _total_images,
                        # Estimated from description text (no download/auth needed)
                        "total_images_est": report.total_images_est,
                        "num_classes_est":  report.num_classes_est,

                        # ── IMAGE: per-class quality (all raw counts) ─────────
                        "per_class_stats": (
                            [s.to_dict() for s in report.per_class_stats]
                            if report.per_class_stats else None
                        ),

                        # ── IMAGE: aggregate quality signals ──────────────────
                        "duplicate_pct": (
                            report.duplicate_result.duplicate_percentage
                            if report.duplicate_result else None
                        ),
                        "duplicate_count": (                  # raw count
                            report.duplicate_result.duplicate_count
                            if report.duplicate_result else None
                        ),
                        "blur_pct":    _blur_r.blurry_percentage if _blur_r else None,
                        "blurry_count": _blurry_count,        # raw count (from sample)
                        "corrupted_pct": (
                            report.corruption_result.corruption_percentage
                            if report.corruption_result else None
                        ),
                        "corrupted_count": (                  # raw count (from sample)
                            report.corruption_result.corrupted_count
                            if report.corruption_result else None
                        ),
                        "dimension_info": _dimension_info,

                        # ── TABULAR: column-level signals ─────────────────────
                        # null_columns: columns with >5% missing (raw count primary)
                        "null_columns":    _null_cols,
                        # columns_detail: every column for low-info / type checks
                        "columns_detail":  _cols_detail,
                        "columns_summary": _columns_summary,

                        # ── SHARED ────────────────────────────────────────────
                        "geographic_bias": _geo_bias,
                    },
                    "summary": {
                        "one_liner":       summary.one_liner,
                        "size":            summary.size_description,
                        "task_labels":     summary.task_labels,
                        "top_tags":        summary.top_tags,
                        "freshness_label": summary.freshness_label,
                    },
                    "target_column": (
                        {
                            "column_name": target_result.best_candidate.column_name,
                            "confidence":  round(target_result.best_candidate.confidence, 2),
                            "reason":      target_result.best_candidate.reason,
                        }
                        if target_result.best_candidate else None
                    ),
                }
            except Exception as exc:
                logger.warning(
                    "per_dataset_analysis_failed",
                    extra={
                        "request_id":   request_id,
                        "canonical_id": cid,
                        "deep":         deep,
                        "error":        str(exc)[:200],
                    },
                    exc_info=True,
                )
                return cid, None

        # ── Run analysis on ALL candidates before final ranking ────────────────
        # IMPORTANT: analysis must happen BEFORE final ranking, not after.
        # Otherwise: ranking decides who gets analyzed, which is circular.
        # Correct flow: analyze all → use real quality scores → re-rank → show.
        #
        # FIX v3.5.0 — CONCURRENCY: Deep analysis is launched as a background
        # future IMMEDIATELY after this comment, before light analysis starts.
        # Light and deep now run in parallel instead of sequentially.
        # Previously deep started only after the 120-second light gather
        # completed, adding up to 120s of unnecessary serial latency.
        #
        # Strategy: all datasets get fast lightweight analysis in parallel with
        # deep analysis of the pre-ranked top candidates.
        datasets_slice = ranked_datasets[:max_results]

        # FIX v3.8.0: deep candidates selected AFTER light-analysis re-rank (see below).
        # Pre-selecting by metadata rank was wrong — the re-rank could move a dataset
        # from position 6 to position 1, but deep analysis had already been wasted on
        # the original top-3. Correct flow: run light on all → re-rank → launch
        # deep on the true top-2 simultaneously via asyncio.gather (parallel).
        def _select_deep_candidates(ranked: list, n: int) -> list:
            """Return top-n by current rank order after re-rank."""
            return ranked[:n]

        # All datasets: fast analysis (light check — no full download)
        # This gives real signals: missing values, basic balance from metadata,
        # duplicate check on tabular, class count from tags.
        all_tasks = [
            _analyse_one(sd, deep=False)
            for sd in datasets_slice
        ]

        try:
            _all_results = await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=120.0,  # FIX: raised from 60s → 120s
                                # REASON: paginated dataset_list_files makes
                                # multiple API calls per dataset; 10 concurrent
                                # runs easily exceed 60s on slow connections.
            )
        except asyncio.TimeoutError:
            logger.warning(
                "analysis_timeout",
                extra={"request_id": request_id, "timeout_s": 120},
            )
            _all_results = []

        for _item in _all_results:
            if isinstance(_item, Exception):
                logger.warning(
                    "analysis_gather_exception",
                    extra={"request_id": request_id, "error": str(_item)[:200]},
                )
                continue
            _cid, _data = _item
            if _cid and _data is not None:
                analysis_map[_cid] = _data

        # ── Re-rank using real quality signals ────────────────────────────────
        # Now that ALL datasets have quality scores, re-rank them.
        # A dataset ranked 6th by metadata may have better real quality than #1.
        def _quality_adjusted_score(sd: Any) -> float:
            raw = sd.dataset if hasattr(sd, "dataset") else sd
            cid = getattr(raw, "canonical_id", None)
            base = float(getattr(sd, "composite_score", 0.0) or 0.0)
            if not cid or cid not in analysis_map:
                return base
            q = analysis_map[cid].get("quality", {})
            deep_score    = q.get("deep_quality_score")
            balance       = q.get("balance_score")
            uniqueness    = q.get("uniqueness_score")
            completeness  = q.get("completeness_score")

            # Image scale signal: use extracted or real total_images count.
            # Larger datasets have higher training value — this is a real
            # quality differentiator that works even without downloading.
            _total_img = q.get("total_images") or q.get("total_images_est") or 0
            image_scale: Optional[float] = None
            if _total_img > 0:
                if   _total_img >= 50_000: image_scale = 0.90
                elif _total_img >= 10_000: image_scale = 0.75
                elif _total_img >=  1_000: image_scale = 0.55
                elif _total_img >=    100: image_scale = 0.35
                else:                      image_scale = 0.15

            # Build quality adjustment: weighted average of real signals.
            # Only use signals that are actually available (not None).
            signals = []
            if deep_score    is not None: signals.append((deep_score / 100.0,    0.35))
            if balance       is not None: signals.append((balance / 100.0,       0.25))
            if uniqueness    is not None: signals.append((uniqueness / 100.0,    0.20))
            if completeness  is not None: signals.append((completeness / 100.0,  0.10))
            if image_scale   is not None: signals.append((image_scale,           0.25))

            if not signals:
                return base
            total_w = sum(w for _, w in signals)
            quality_adj = sum(s * w for s, w in signals) / total_w
            # Blend: 50% original metadata rank + 50% real quality
            return round((base * 0.50) + (quality_adj * 0.50), 4)

        # Re-sort using quality-adjusted scores (light analysis signals)
        ranked_datasets = sorted(
            ranked_datasets,
            key=_quality_adjusted_score,
            reverse=True,
        )
        logger.info(
            "quality_rerank_complete",
            extra={
                "request_id": request_id,
                "datasets_reranked": len(ranked_datasets),
                "had_analysis": len(analysis_map),
            },
        )

        # ── FIX v3.8.0: select deep candidates from TRUE top-2 after re-rank ────
        # Now that light analysis has re-ranked datasets, pick the real top-2.
        # Launch BOTH deep tasks simultaneously (asyncio.gather = parallel, not serial).
        deep_candidates = _select_deep_candidates(ranked_datasets, DEEP_ANALYSIS_COUNT)
        logger.info(
            "deep_analysis_candidates",
            extra={
                "request_id": request_id,
                "count": len(deep_candidates),
                "sources": [
                    getattr(
                        (sd.dataset if hasattr(sd, "dataset") else sd),
                        "source", "?"
                    )
                    for sd in deep_candidates
                ],
            },
        )

        try:
            # Both tasks start at the same time — asyncio.gather is parallel.
            # Elapsed wall-clock ≈ max(task1, task2), not task1 + task2.
            _deep_results = await asyncio.wait_for(
                asyncio.gather(
                    *[_analyse_one(sd, deep=True) for sd in deep_candidates],
                    return_exceptions=True,
                ),
                timeout=DEEP_BATCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "deep_analysis_timeout",
                extra={"request_id": request_id, "timeout_s": DEEP_BATCH_TIMEOUT},
            )
            _deep_results = []

        # Update analysis_map with deep results (overwrite shallow light results)
        for _item in _deep_results:
            if isinstance(_item, Exception):
                continue
            _cid, _data = _item
            if _cid and _data is not None:
                analysis_map[_cid] = _data

        # ── FIX v3.5.0: Re-rank again using deep analysis scores ───────────────
        # Deep results have overwritten analysis_map for the top-3 candidates.
        # Re-sorting here ensures those richer signals (real quality_score,
        # balance_score, uniqueness_score from actual downloaded data) affect the
        # final rank positions. Previously this second sort was missing — deep
        # scores improved the "analysis" field in the response but never moved
        # a dataset up or down in the final ordered list.
        ranked_datasets = sorted(
            ranked_datasets,
            key=_quality_adjusted_score,
            reverse=True,
        )
        logger.info(
            "deep_rerank_complete",
            extra={
                "request_id": request_id,
                "datasets_reranked": len(ranked_datasets),
                "had_deep_analysis": sum(
                    1 for _sd in ranked_datasets[:DEEP_ANALYSIS_COUNT]
                    if getattr(
                        (_sd.dataset if hasattr(_sd, "dataset") else _sd),
                        "canonical_id", None
                    ) in analysis_map
                ),
            },
        )

        # ── FIX v3.5.0: Write quality-adjusted score back to ScoredDataset ────
        # composite_score on each ScoredDataset still holds the raw metadata-only
        # value from EvaluatorPipeline. The API response was showing that stale
        # score even though ranking used the quality-blended value. Writeback here
        # so the displayed composite_score matches what ordering actually used.
        for _sd in ranked_datasets:
            _adj = _quality_adjusted_score(_sd)
            try:
                object.__setattr__(_sd, "composite_score", _adj)
            except (AttributeError, TypeError):
                try:
                    _sd.composite_score = _adj  # type: ignore[attr-defined]
                except Exception:
                    pass  # best-effort

        # ── FIX v3.5.0: Source diversity enforcement on final result set ───────
        # After quality-adjusted re-sort, positions 4-10 are in pure score order.
        # Kaggle's structural popularity advantage (10x–100x more downloads) means
        # all top-10 slots can be Kaggle even with the 30% ranker penalty.
        # Enforce at least 2 results per source that contributed candidates,
        # promoting them from lower positions while preserving intra-source order.
        def _enforce_source_diversity(datasets: list, min_per_source: int = 2) -> list:
            if not datasets:
                return datasets
            by_source: dict[str, list] = {}
            for _sd in datasets:
                _raw = _sd.dataset if hasattr(_sd, "dataset") else _sd
                _src = getattr(_raw, "source", "unknown") or "unknown"
                by_source.setdefault(_src, []).append(_sd)

            result: list = []
            source_filled: dict[str, int] = {s: 0 for s in by_source}

            # Pass 1: guarantee min_per_source slots per source
            for _src, _bucket in by_source.items():
                needed = max(0, min_per_source - source_filled[_src])
                for _item in _bucket[:needed]:
                    if id(_item) not in {id(x) for x in result}:
                        result.append(_item)
                        source_filled[_src] += 1

            # Pass 2: fill remaining positions in quality order
            result_ids = {id(x) for x in result}
            for _item in datasets:
                if id(_item) not in result_ids:
                    result.append(_item)
                    result_ids.add(id(_item))

            return result[:len(datasets)]

        _sources_with_results = set(
            getattr((sd.dataset if hasattr(sd, "dataset") else sd), "source", "")
            for sd in ranked_datasets
        ) - {""}
        if len(_sources_with_results) > 1:
            ranked_datasets = _enforce_source_diversity(ranked_datasets, min_per_source=2)
            logger.info(
                "source_diversity_enforced",
                extra={
                    "request_id": request_id,
                    "sources": list(_sources_with_results),
                    "final_count": len(ranked_datasets),
                },
            )

    except Exception as exc:
        logger.warning(
            "analysis_stage_failed",
            extra={"request_id": request_id, "error": str(exc)[:120]},
        )

    # ── 8. Serialize results (Task 13 — ranking explanation injected) ─────────
    serialized: List[Dict[str, Any]] = []
    for rank_idx, sd in enumerate(ranked_datasets[:max_results], start=1):
        raw = sd.dataset if hasattr(sd, "dataset") else sd
        cid = getattr(raw, "canonical_id", "")
        score = float(getattr(sd, "composite_score", 0.0) or 0.0)

        # Build breakdown dict
        bd = getattr(sd, "breakdown", None)
        if bd is not None:
            breakdown = {
                "task_relevance":    round(getattr(bd, "task_relevance", 0.0) or 0.0, 4),
                "quality":           round(getattr(bd, "quality", 0.0) or 0.0, 4),
                "popularity":        round(getattr(bd, "popularity", 0.0) or 0.0, 4),
                "freshness":         round(getattr(bd, "freshness", 0.0) or 0.0, 4),
                "description_match": round(getattr(bd, "description_match", 0.0) or 0.0, 4),
            }
        else:
            breakdown = {}

        diag = diagnostics_map.get(cid)
        analysis = analysis_map.get(cid)

        serialized.append(
            _serialize(
                ds=sd,
                score=score,
                breakdown=breakdown,
                rank=rank_idx,
                diagnostics=diag,
                analysis=analysis,
            )
        )

    # ── 9. Persist search record ───────────────────────────────────────────────
    processing_ms = int((time.monotonic() - t_start) * 1_000)
    try:
        from datascout.storage.repositories.search_repository import (  # noqa: PLC0415
            SearchRecord, SearchRepository,
        )
        repo = SearchRepository()
        sources_used = (
            list(agent_trace.sources_succeeded)
            if agent_trace else sources
        )
        record = SearchRecord(
            query=raw_query,
            results_count=len(serialized),
            sources_used=sources_used,
            confidence=_confidence_label(
                len(serialized),
                serialized[0]["composite_score"] if serialized else 0.0,
            ),
            processing_time_ms=processing_ms,
            request_id=request_id,
        )
        asyncio.create_task(repo.save_search(record))
    except Exception:
        pass

    # ── 10. Assemble final response ────────────────────────────────────────────
    top_score = serialized[0]["composite_score"] if serialized else 0.0
    confidence = _confidence_label(len(serialized), top_score)

    # ── FIX v3.7.0: intelligence_available must NOT be hardcoded True.
    # Root cause: UI badge "Gemini AI" was always shown because this field was always
    # True regardless of whether Gemini succeeded, was rate-limited, or fell back to
    # a deterministic mock/template. We default to False here and flip it to True
    # only when a real LLM (not Mock/template) produced the insights.
    _intelligence_available: bool = False
    _intelligence_provider: str = "template"

    result: Dict[str, Any] = {
        "query":                 raw_query,
        "expanded_query":        expanded_query if expanded_query != raw_query else None,
        "results":               serialized,
        "total_found":           len(raw_datasets),
        "returned":              len(serialized),
        "confidence":            confidence,
        "intelligence_available": False,      # updated after LLM call below
        "intelligence_provider":  "template", # updated after LLM call below
        "processing_time_ms":    processing_ms,
        "request_id":            request_id,
        "retrieval_method":      retrieval_method,
    }

    # ── 10a. LLM Intelligence — Gemini explains why each dataset ranked ────────
    # ResearchIntelligenceEngine calls Gemini (with Mock fallback) to generate
    # human-readable why_ranked, strengths, weaknesses, and research summary.
    # Falls back to deterministic templates if Gemini is unavailable — never fails.
    llm_dataset_insights: dict[str, dict] = {}
    llm_summary: str = ""
    llm_follow_ups: List[str] = []
    llm_gaps: List[str] = []

    try:
        from datascout.llm.research.intelligence_engine import ResearchIntelligenceEngine  # noqa: PLC0415
        from datascout.llm.llm__init__ import build_fallback_chain  # noqa: PLC0415

        llm_chain = build_fallback_chain(include_mock=True)
        engine = ResearchIntelligenceEngine(llm_provider=llm_chain)

        # Build the ranked_datasets dicts the engine expects.
        # ranked_datasets items are ScoredDataset objects — extract plain dicts
        # because intelligence_engine calls .get() expecting dict, not Pydantic model.
        engine_datasets = []
        engine_breakdowns = []
        for sd in ranked_datasets[:10]:
            # Handle both ScoredDataset wrapper and bare RawDataset
            ds_obj = getattr(sd, "dataset", sd)
            # ScoredDataset stores the breakdown directly as .breakdown (not .score)
            breakdown_obj = getattr(sd, "breakdown", None)

            # Score breakdown → plain dict
            bd: dict = {}
            if breakdown_obj is not None:
                if hasattr(breakdown_obj, "to_dict"):
                    bd = breakdown_obj.to_dict()
                elif hasattr(breakdown_obj, "model_dump"):
                    bd = breakdown_obj.model_dump()
                elif isinstance(breakdown_obj, dict):
                    bd = breakdown_obj
                else:
                    bd = {
                        "task_relevance":    getattr(breakdown_obj, "task_relevance", 0.0),
                        "quality":           getattr(breakdown_obj, "quality", 0.0),
                        "popularity":        getattr(breakdown_obj, "popularity", 0.0),
                        "freshness":         getattr(breakdown_obj, "freshness", 0.0),
                        "description_match": getattr(breakdown_obj, "description_match", 0.0),
                    }

            composite = getattr(breakdown_obj, "composite", 0.0) if breakdown_obj else 0.0

            # Convert ds_obj (Pydantic RawDataset) to plain dict for the engine
            def _attr(name: str, default=None):
                v = getattr(ds_obj, name, default)
                return v if v is not None else default

            engine_datasets.append({
                "canonical_id":      _attr("canonical_id", ""),
                "title":             _attr("title", ""),
                "source":            _attr("source", ""),
                "description_short": _attr("description_short", ""),
                "tags_primary":      list(_attr("tags_primary", []) or []),
                "row_count":         _attr("row_count"),
                "download_count":    _attr("download_count"),
                "license_name":      _attr("license_name"),
                "last_updated":      str(_attr("last_updated", "") or ""),
                "composite_score":   composite,
                "raw":               {},  # engine uses this only for fallback label
            })
            engine_breakdowns.append(bd)

        # Check cache first — avoids re-calling Gemini for repeated queries
        _cache_key = f"{expanded_query}:{len(engine_datasets)}"
        _cached = _intel_cache.get(_cache_key)
        if _cached and (_time.monotonic() - _cached[1]) < _INTEL_CACHE_TTL:
            intel_result = _cached[0]
            logger.info("intel_cache_hit", extra={"query": expanded_query[:60]})
        else:
            intel_result = await engine.generate_intelligence(
                query=expanded_query,
                ranked_datasets=engine_datasets,
                score_breakdowns=engine_breakdowns,
                total_candidates=len(raw_datasets),
                confidence_level=confidence,
                max_explain=min(len(engine_datasets), 3),  # FIX v3.5.1: spec requires top-3 only; was 5, wasting free-tier tokens
            )
            # Cache the result
            _intel_cache[_cache_key] = (intel_result, _time.monotonic())
            # Keep cache small — evict entries older than TTL
            _expired = [k for k, v in _intel_cache.items()
                       if _time.monotonic() - v[1] > _INTEL_CACHE_TTL]
            for k in _expired:
                _intel_cache.pop(k, None)

        # Map dataset_id → insight dict for per-card use
        for insight in intel_result.dataset_insights:
            llm_dataset_insights[insight.dataset_id] = insight.to_dict()

        rc = intel_result.research_context
        llm_summary = getattr(rc, "ecosystem_summary", "") or ""
        llm_follow_ups = list(getattr(rc, "follow_up_searches", []) or [])[:5]
        llm_gaps = list(getattr(rc, "research_gaps", []) or [])[:5]

        # ── FIX v3.7.0: Determine actual provider used and set intelligence flags.
        # pipeline_degraded=True means Gemini was rate-limited/failed and all
        # insights were generated by the deterministic fallback/mock template.
        # We must NOT label those as "Gemini AI" in the UI.
        #
        # Strategy: check pipeline_degraded on intel_result. If False, Gemini
        # succeeded for at least the top-3 explanations. If True, fallback was used.
        # Also inspect individual insight.fallback_used flags for per-card accuracy.
        _pipeline_degraded = getattr(intel_result, "pipeline_degraded", True)
        if not _pipeline_degraded:
            _intelligence_available = True
            _intelligence_provider = "gemini"
        else:
            # All insights were produced by deterministic templates — do not claim Gemini
            _intelligence_available = False
            _intelligence_provider = "template"

        # Update the result dict with the real values now that we know them
        result["intelligence_available"] = _intelligence_available
        result["intelligence_provider"]  = _intelligence_provider

        # Annotate each insight with whether it was LLM-generated or template
        for insight in intel_result.dataset_insights:
            cid = getattr(insight, "dataset_id", "")
            if cid in llm_dataset_insights:
                llm_dataset_insights[cid]["is_llm_generated"] = (
                    not getattr(insight, "fallback_used", True)
                )

        logger.info(
            "llm_intelligence_generated",
            extra={
                "request_id": request_id,
                "latency_ms": intel_result.llm_latency_ms,
                "degraded": intel_result.pipeline_degraded,
                "provider": _intelligence_provider,
            },
        )
    except Exception as exc:
        import traceback as _tb
        print(f"\n[LLM INTELLIGENCE FAILED] {type(exc).__name__}: {exc}")
        _tb.print_exc()
        logger.warning(
            "llm_intelligence_failed",
            extra={"request_id": request_id, "error": str(exc)[:500]},
            exc_info=True,
        )

    # Merge LLM insights into serialized datasets
    for ds in serialized:
        cid = ds.get("canonical_id", "") or ds.get("dataset_id", "")
        if cid in llm_dataset_insights:
            ins = llm_dataset_insights[cid]
            ds["why_ranked_here"] = ins.get("why_ranked") or ds.get("why_ranked_here", "")
            ds["strengths"]       = ins.get("strengths") or ds.get("strengths", [])
            ds["weaknesses"]      = ins.get("weaknesses") or ds.get("weaknesses", [])

    # Build dataset_insights list for the intelligence panel
    dataset_insights = [
        {
            "dataset_id":      ds.get("canonical_id", ""),
            "why_ranked":      ds.get("why_ranked_here", ""),
            "strengths":       ds.get("strengths", []),
            "weaknesses":      ds.get("weaknesses", []),
            "score_narrative": ds.get("why_ranked_here", ""),
        }
        for ds in serialized
    ]

    # Use LLM summary if available, else build a static fallback
    from collections import Counter as _Counter  # noqa: PLC0415
    src_counts = _Counter(ds.get("source", "") for ds in serialized)
    src_parts  = [f"{cnt} from {src.title()}" for src, cnt in src_counts.most_common()]
    static_summary = (
        f"Found {len(serialized)} relevant datasets from {len(raw_datasets)} candidates "
        f"({', '.join(src_parts)})."
        if src_parts else
        f"Found {len(serialized)} datasets from {len(raw_datasets)} candidates."
    )
    if retrieval_method == "elasticsearch_hybrid":
        static_summary += " Retrieved via Elastic hybrid search (BM25 + semantic vector)."
    summary = llm_summary or static_summary

    # Collect follow-up suggestions — prefer LLM-generated, fall back to enrichment keywords
    followups: List[str] = llm_follow_ups[:]
    if not followups and enrichment is not None:
        kw = getattr(enrichment, "keywords", []) or []
        domains = getattr(enrichment, "domains_detected", []) or []
        if kw:
            followups.append(" ".join(kw[:3]) + " benchmark")
        if domains:
            followups.extend(d + " dataset" for d in domains[:2])

    # Collect metadata gaps — prefer LLM-generated, fall back to bias warnings
    gaps: List[str] = llm_gaps[:]
    if not gaps:
        seen_gaps: set = set()
        for ds in serialized:
            for w in (ds.get("bias_warnings") or []):
                if w not in seen_gaps:
                    gaps.append(w)
                    seen_gaps.add(w)

    # Build human-friendly infrastructure notes for the UI
    infra_notes: List[str] = []
    if retrieval_method != "elasticsearch_hybrid":
        infra_notes.append(
            "Semantic re-ranking is off — start Elasticsearch and set "
            "ELASTIC_ENABLED=true in your .env for better result ordering."
        )
    if agent_trace is not None:
        failed_sources = {f.source for f in getattr(agent_trace, "adapter_failures", [])}
        succeeded_sources = set(getattr(agent_trace, "sources_succeeded", []))
        for src in failed_sources - succeeded_sources:
            failures_for_src = [
                f for f in getattr(agent_trace, "adapter_failures", [])
                if f.source == src
            ]
            if failures_for_src:
                reason = failures_for_src[0].reason
                if reason == "auth_failed":
                    infra_notes.append(
                        f"{src.title()} is not connected — add your {src.upper()}_USERNAME "
                        f"and {src.upper()}_KEY to .env to include its datasets."
                    )
                elif reason == "not_installed":
                    infra_notes.append(
                        f"{src.title()} adapter needs 'pip install {src}' to work."
                    )
                elif reason == "rate_limited":
                    infra_notes.append(
                        f"{src.title()} hit its rate limit — results exclude that source for now."
                    )
                elif reason == "timeout":
                    infra_notes.append(
                        f"{src.title()} took too long to respond and was skipped this time."
                    )
                else:
                    infra_notes.append(
                        f"{src.title()} was temporarily unavailable — results may be incomplete."
                    )

    result["intelligence"] = {
        "summary":              summary,
        "dataset_insights":     dataset_insights,
        "metadata_gaps":        gaps[:5],
        "follow_up_searches":   followups[:5],
        "ecosystem_observation": (
            "Results ranked using semantic + keyword hybrid search for higher relevance."
            if retrieval_method == "elasticsearch_hybrid"
            else "Results ranked by relevance, quality, freshness, and popularity signals."
        ),
        "retrieval_method":     retrieval_method,
        "infrastructure_notes": infra_notes,
    }

    # Task 14 — agent status fields
    if agent_trace is not None:
        result["agent_message"]    = agent_status_message
        result["sources_used"]     = list(getattr(agent_trace, "sources_succeeded", []))
        result["adapter_failures"] = [
            {"source": f.source, "reason": f.reason}
            for f in getattr(agent_trace, "adapter_failures", [])
        ]
        result["broadened"]        = getattr(agent_trace, "broadening_applied", False)
        result["query_used"]       = getattr(agent_trace, "query_used", raw_query)
    else:
        result["agent_message"]    = ""
        result["sources_used"]     = sources
        result["adapter_failures"] = []
        result["broadened"]        = False
        result["query_used"]       = expanded_query

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/search")
async def search_datasets(
    query: SearchQuery,
    request: Request,
) -> JSONResponse:
    """Search for ML datasets matching *query*.

    Parameters
    ----------
    query:
        ``SearchQuery`` body with ``raw_query: str`` and
        ``max_results: int`` (default 10, max 50).

    Returns
    -------
    JSONResponse:
        Ranked list of datasets with per-dataset explanations.

    Raises
    ------
    HTTPException 400:
        If the query string is empty.
    """
    raw_query = (query.raw_query or "").strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="raw_query must not be empty.")

    # ── Refinement detection ─────────────────────────────────────────────────
    # If user typed a follow-up like "show me more", "similar ones", "different source"
    # merge it with the previous query so the pipeline gets the full context.
    previous_query = (getattr(query, "previous_query", None) or "").strip()
    if previous_query:
        _refinement_phrases = (
            "more", "show more", "i want more", "give me more",
            "similar", "more like this", "different", "other options",
            "show me", "find more", "next", "again", "retry",
        )
        is_pure_refinement = raw_query.lower() in _refinement_phrases or len(raw_query.split()) <= 3
        if is_pure_refinement:
            # Merge: treat raw_query as refinement instruction on top of previous
            raw_query = f"{previous_query} — {raw_query}"
        elif len(raw_query.split()) < 6:
            # Short query — prepend previous context so query understanding has enough signal
            raw_query = f"{previous_query} {raw_query}"

    max_results = max(1, min(int(query.max_results or 10), 50))
    request_id = str(uuid.uuid4())

    logger.info(
        "search_request_received",
        extra={
            "request_id": request_id,
            "query":      raw_query[:120],
            "max_results": max_results,
        },
    )

    try:
        result = await _run_pipeline(
            raw_query=raw_query,
            max_results=max_results,
            request_id=request_id,
        )
    except Exception as exc:
        logger.error(
            "search_pipeline_unhandled_error",
            extra={"request_id": request_id, "error": str(exc)},
            exc_info=True,
        )
        result = {
            "query":             raw_query,
            "results":           [],
            "total_found":       0,
            "returned":          0,
            "confidence":        "low",
            "message":           "An unexpected error occurred. Please try again.",
            "agent_message":     "",
            "sources_used":      [],
            "adapter_failures":  [],
            "request_id":        request_id,
            "processing_time_ms": 0,
        }

    return JSONResponse(content=result)


@router.get("/search")
async def search_datasets_get(
    q: str,
    max_results: int = 10,
    request: Request = None,
) -> JSONResponse:
    """GET convenience alias for ``POST /search``.

    Parameters
    ----------
    q:
        URL query parameter — the search query string.
    max_results:
        Number of results to return (default 10, max 50).
    """
    from datascout.contracts.requests import SearchQuery as SQ  # noqa: PLC0415
    return await search_datasets(
        query=SQ(raw_query=q, max_results=max_results),
        request=request,
    )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.

    Returns
    -------
    dict:
        Service status, Elasticsearch connectivity, and uptime info.
    """
    es_status = "disabled"
    if _elastic_client is not None:
        es_status = "connected" if getattr(_elastic_client, "_connected", False) else "disconnected"

    return {
        "status":      "ok",
        "elasticsearch": es_status,
        "embedding_engine": "loaded" if _embedding_engine is not None else "not_loaded",
    }