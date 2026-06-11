"""
datascout.infrastructure.warmup
────────────────────────────────────────────────────────
Non-blocking demo warmup — fires pre-configured queries
in the background after startup to prime retrieval caches.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("datascout.infrastructure.warmup")


def schedule_warmup(agent: Any, queries: Optional[list[str]] = None) -> None:
    """
    Schedule background warmup queries. Non-blocking.
    If the agent or event loop is unavailable, silently skips.
    """
    queries = queries or [
        "image classification dataset",
        "sentiment analysis NLP",
        "tabular classification",
    ]

    async def _run_warmup() -> None:
        from datascout.contracts.requests import SearchQuery

        for q in queries:
            try:
                await asyncio.sleep(1)  # Stagger requests
                sq = SearchQuery(raw_query=q, max_results=5)
                await agent.execute(sq)
                logger.info("warmup_query_complete", extra={"query": q})
            except Exception as exc:
                logger.debug("warmup_query_failed", extra={"query": q, "error": str(exc)[:60]})

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_warmup())
        logger.info("warmup_scheduled", extra={"query_count": len(queries)})
    except Exception as exc:
        logger.warning("warmup_schedule_failed", extra={"error": str(exc)[:60]})