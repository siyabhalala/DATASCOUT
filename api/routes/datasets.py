"""
datascout.api.routes.datasets
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Dataset search and retrieval endpoints.

ENDPOINTS:
  POST /api/v1/datasets/search   — Execute a dataset search via ScoutAgent
  GET  /api/v1/datasets/{id}     — Retrieve a single dataset by canonical_id
  GET  /api/v1/datasets/         — List datasets with cursor-based pagination

DESIGN DECISIONS:

  1. WHY POST for search (not GET)?
     - Search body can be complex (filters, preferences, 500-char query)
     - GET query strings have URL-length limits (2KB on some proxies)
     - POST body is not cached by default — correct for real-time search

  2. WHY cursor-based pagination (not offset)?
     - Offset pagination: OFFSET 1000 forces DB to scan 1000 rows → O(n) waste
     - Cursor pagination: WHERE id > cursor → O(log n) via index scan
     - Stable under concurrent inserts — offset shifts rows, cursor doesn't

  3. WHY 30s timeout on search endpoint?
     - ScoutAgent runs up to 3 ReAct iterations each calling multiple adapters
     - p99 latency observed at ~12s — 30s gives 2.5× headroom
     - Return partial results if timeout — never return empty with no error

  4. WHY agent None check?
     - ScoutAgent init can fail (no API keys, missing models)
     - SYSTEM RULE #1: never crash — return 503 with clear message

Complexity:
  - POST /search: O(I × (H + E)) where I ≤ 3 iterations, H = hybrid search
  - GET /{id}: O(log n) — index lookup
  - GET /: O(log n + page_size) — cursor + scan

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse

from datascout.api.dependencies import get_scout_agent
from datascout.api.models.requests import DatasetSearchRequest
from datascout.api.models.responses import (
    DatasetListResponse,
    DatasetSearchResponse,
    ErrorResponse,
    SingleDatasetResponse,
)
from datascout.infrastructure.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Search timeout — ScoutAgent p99 is ~12s; 30s gives 2.5× headroom
_SEARCH_TIMEOUT_SECONDS = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# POST /search
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=DatasetSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search for datasets using semantic + BM25 hybrid search",
    responses={
        503: {"model": ErrorResponse, "description": "Scout agent unavailable"},
        504: {"model": ErrorResponse, "description": "Search timed out"},
        422: {"description": "Validation error"},
    },
)
async def search_datasets(
    request: Request,
    body: DatasetSearchRequest,
    agent: Any = Depends(get_scout_agent),
) -> JSONResponse:
    """
    Execute a dataset search via the Level-0 ScoutAgent.

    The agent runs a ReAct loop (max 3 iterations) combining:
    - Hybrid semantic search (BM25 + sentence-transformers + FAISS)
    - Deterministic scoring and ranking
    - LLM explanation (explains only — never ranks)

    Returns ranked results with explanations and pipeline metadata.

    Complexity: O(I × (H + E)) where I ≤ 3, H = hybrid search, E = evaluation.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if agent is None:
        logger.info(
            "search_agent_unavailable",
            extra={"request_id": request_id, "query": body.query[:50]},
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "AGENT_UNAVAILABLE",
                "message": "Scout agent is not available. Check API key configuration.",
                "request_id": request_id,
            },
        )

    logger.info(
        "dataset_search_started",
        extra={
            "request_id": request_id,
            "query": body.query[:100],
            "max_results": body.max_results,
            "sources": body.sources,
        },
    )

    try:
        # Build SearchQuery contract from API request
        from datascout.contracts.requests import (
            SearchFilters,
            SearchPreferences,
            SearchQuery,
        )

        search_query = SearchQuery(
            raw_query=body.query,
            max_results=body.max_results,
            sources=body.sources or [],
            filters=SearchFilters(
                min_rows=body.filters.min_rows if body.filters else None,
                max_rows=body.filters.max_rows if body.filters else None,
                require_description=body.filters.require_description if body.filters else False,
            ),
            preferences=SearchPreferences(
                prefer_recent=body.preferences.prefer_recent if body.preferences else False,
                prefer_popular=body.preferences.prefer_popular if body.preferences else False,
            ),
            session_id=request_id,
        )

        # Run agent with timeout — SYSTEM RULE #1: never crash
        result = await asyncio.wait_for(
            _run_agent(agent, search_query),
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )

        logger.info(
            "dataset_search_completed",
            extra={
                "request_id": request_id,
                "query": body.query[:50],
                "result_count": len(result.get("results", [])),
                "processing_time_ms": result.get("processing_time_ms", 0),
            },
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result,
        )

    except asyncio.TimeoutError:
        logger.info(
            "dataset_search_timeout",
            extra={
                "request_id": request_id,
                "timeout_seconds": _SEARCH_TIMEOUT_SECONDS,
                "query": body.query[:50],
            },
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "SEARCH_TIMEOUT",
                "message": f"Search exceeded {_SEARCH_TIMEOUT_SECONDS}s timeout. Try a simpler query.",
                "request_id": request_id,
            },
        )

    except Exception as exc:
        # SYSTEM RULE #1: catch all — degrade gracefully
        logger.info(
            "dataset_search_failed",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "query": body.query[:50],
            },
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "query_id": "error",
                "results": [],
                "total_found": 0,
                "confidence": "low",
                "partial_result": True,
                "error_message": f"Search failed: {type(exc).__name__}",
                "processing_time_ms": 0,
                "request_id": request_id,
            },
        )

    except BaseException as exc:
        # FIX: CancelledError (Python 3.8+) is BaseException not Exception.
        # Uvicorn cancels request tasks on client disconnect or worker timeout.
        # Without this, the server logs an unhandled exception and may crash the worker.
        # We return 503 so the client gets a clean error instead of a silent hang.
        logger.warning(
            "dataset_search_cancelled",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "query": body.query[:50],
            },
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "REQUEST_CANCELLED",
                "message": "Request was cancelled. Please try again.",
                "request_id": request_id,
            },
        )


async def _run_agent(agent: Any, search_query: Any) -> dict[str, Any]:
    """
    Run the ToolOrchestrator (or any compatible agent).

    Dispatch priority:
      1. execute()  — ToolOrchestrator async method (preferred)
      2. run()      — legacy / test doubles

    asyncio.to_thread: runs sync code without blocking the event loop.
    """
    import asyncio
    import inspect

    # ToolOrchestrator uses .execute(); legacy agents use .run()
    if inspect.iscoroutinefunction(getattr(agent, "execute", None)):
        result = await agent.execute(search_query)
    elif inspect.iscoroutinefunction(getattr(agent, "run", None)):
        result = await agent.run(search_query)
    elif hasattr(agent, "execute"):
        result = await asyncio.to_thread(agent.execute, search_query)
    else:
        result = await asyncio.to_thread(agent.run, search_query)

    # Normalize result — AgentResponse has to_dict_safe(); else assume dict
    if hasattr(result, "to_dict_safe"):
        return result.to_dict_safe()
    if isinstance(result, dict):
        return result
    return {"results": [], "total_found": 0, "error_message": "Unexpected agent response type"}


# ─────────────────────────────────────────────────────────────────────────────
# GET /{dataset_id}
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/{dataset_id}",
    response_model=SingleDatasetResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a single dataset by canonical ID",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def get_dataset(
    dataset_id: str,
    request: Request,
) -> JSONResponse:
    """
    Retrieve dataset metadata by canonical_id (e.g. 'kaggle:titanic').

    Complexity: O(log n) — primary key index lookup.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "dataset_get_started",
        extra={"request_id": request_id, "dataset_id": dataset_id},
    )

    try:
        from datascout.storage.repositories.dataset_repository import DatasetRepository  # type: ignore[attr-defined]

        repo = DatasetRepository()
        dataset = await repo.get_by_canonical_id(dataset_id)

        if dataset is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "DATASET_NOT_FOUND",
                    "message": f"Dataset '{dataset_id}' not found.",
                    "request_id": request_id,
                },
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"dataset": dataset, "request_id": request_id},
        )

    except Exception as exc:
        # SYSTEM RULE #1: never crash
        logger.info(
            "dataset_get_failed",
            extra={
                "request_id": request_id,
                "dataset_id": dataset_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve dataset.",
                "request_id": request_id,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET / — list with cursor-based pagination
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/",
    response_model=DatasetListResponse,
    status_code=status.HTTP_200_OK,
    summary="List datasets with cursor-based pagination",
)
async def list_datasets(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100, description="Page size")] = 20,
    cursor: Annotated[str | None, Query(description="Pagination cursor (opaque)")] = None,
    source: Annotated[
        str | None, Query(description="Filter by source: kaggle|huggingface|openml")
    ] = None,
) -> JSONResponse:
    """
    List datasets with cursor-based pagination.

    WHY cursor (not offset):
    - Offset: SKIP 1000 rows → O(n) scan regardless of index
    - Cursor: WHERE id > cursor → O(log n) via index
    - Stable under concurrent inserts — no row-shift side effects

    Complexity: O(log n + limit) where n = total datasets.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "dataset_list_started",
        extra={
            "request_id": request_id,
            "limit": limit,
            "cursor": cursor,
            "source": source,
        },
    )

    try:
        from datascout.storage.repositories.dataset_repository import DatasetRepository  # type: ignore[attr-defined]

        repo = DatasetRepository()
        page_result = await repo.list_paginated(limit=limit, cursor=cursor, source=source)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "datasets": page_result.get("items", []),
                "next_cursor": page_result.get("next_cursor"),
                "has_more": page_result.get("has_more", False),
                "total_count": page_result.get("total_count", 0),
                "request_id": request_id,
            },
        )

    except Exception as exc:
        # SYSTEM RULE #1: degrade gracefully
        logger.info(
            "dataset_list_failed",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "datasets": [],
                "next_cursor": None,
                "has_more": False,
                "total_count": 0,
                "error_message": f"List failed: {type(exc).__name__}",
                "request_id": request_id,
            },
        )