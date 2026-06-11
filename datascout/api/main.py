"""
datascout.api.main
───────────────────
FastAPI application factory and lifespan.

STARTUP ORDER:
  1. Database — create async engine, run pending Alembic migrations (if any)
  2. ToolOrchestrator — initialise adapter health checks
  3. Elasticsearch + EmbeddingEngine — optional, only if ELASTIC_ENABLED=true

SHUTDOWN ORDER (reverse):
  1. Close Elasticsearch connection
  2. Close DB connections
  
"""
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://datascout-ecru.vercel.app/",  # your deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    All initialisation steps are wrapped in individual try/except blocks so
    that a failure in one step (e.g. Elasticsearch unreachable) never prevents
    the application from starting.

    Parameters
    ----------
    app:
        The FastAPI application instance.
    """
    # ── 1. Settings ───────────────────────────────────────────────────────────
    settings = None
    try:
        from datascout.infrastructure.config.settings import get_settings  # noqa: PLC0415
        settings = get_settings()
        logger.info(
            "settings_loaded",
            extra={
                "environment": getattr(settings, "environment", "unknown"),
                "app_name":    getattr(settings, "app_name", "DATASCOUT"),
            },
        )
    except Exception as exc:
        logger.warning("settings_load_failed", extra={"error": str(exc)[:120]})

    # ── 2. Database ───────────────────────────────────────────────────────────
    try:
        from datascout.storage.database import init_db  # noqa: PLC0415
        await init_db()
        logger.info("database_initialised")
    except ImportError:
        logger.debug("storage.database not found — skipping DB init")
    except Exception as exc:
        logger.warning("database_init_failed", extra={"error": str(exc)[:200]})

    # ── 3. ToolOrchestrator ───────────────────────────────────────────────────
    try:
        import inspect as _inspect
        from datascout.orchestration.tool_orchestrator import ToolOrchestrator  # noqa: PLC0415
        orchestrator = ToolOrchestrator()
        if hasattr(orchestrator, "initialize"):
            _init = orchestrator.initialize()
            if _inspect.isawaitable(_init):
                await _init
        app.state.orchestrator = orchestrator
        logger.info("tool_orchestrator_initialised")
    except ImportError:
        logger.debug("ToolOrchestrator not found — skipping")
    except Exception as exc:
        logger.warning(
            "tool_orchestrator_init_failed", extra={"error": str(exc)[:200]}
        )

    # ── 4. Elasticsearch + EmbeddingEngine (Task 15) ──────────────────────────
    try:
        from datascout.api.routes.search_v2 import _init_elasticsearch  # noqa: PLC0415
        if settings is not None:
            await _init_elasticsearch(settings)
        else:
            logger.warning("elasticsearch_init_skipped_no_settings")
    except Exception as exc:
        logger.warning(
            "elasticsearch_startup_skipped",
            extra={"error": str(exc)[:120]},
        )

    # ── 5. Configuration health check — surface missing keys at startup ─────
    _warn = []
    if settings:
        if not getattr(settings, 'google_api_key', None):
            _warn.append('GOOGLE_API_KEY not set — Gemini LLM disabled; template fallback will be used')
        if not getattr(settings, 'kaggle_username', None) or not getattr(settings, 'kaggle_key', None):
            _warn.append('KAGGLE_USERNAME / KAGGLE_KEY not set — Kaggle adapter disabled')
        if not getattr(settings, 'hf_token', None):
            logger.info('startup_info', extra={'msg': 'HF_TOKEN not set — public HuggingFace datasets only (gated datasets will fail)'})
    for w in _warn:
        logger.warning('startup_config_warning', extra={'msg': w})
    if _warn:
        logger.warning('startup_missing_keys_hint', extra={
            'msg': 'Copy .env.example to .env and fill in your API keys to enable all features'
        })

    logger.info('datascout_startup_complete')

    yield  # ← Application serves requests here

    # ══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ══════════════════════════════════════════════════════════════════════════

    # Close Elasticsearch connection
    try:
        from datascout.api.routes import search_v2 as _sv2  # noqa: PLC0415
        if getattr(_sv2, "_elastic_client", None) is not None:
            await _sv2._elastic_client.disconnect()
            logger.info("elasticsearch_disconnected_on_shutdown")
    except Exception as exc:
        logger.warning(
            "elasticsearch_shutdown_error", extra={"error": str(exc)[:80]}
        )

    # Close DB connections
    try:
        from datascout.storage.database import close_db  # noqa: PLC0415
        await close_db()
        logger.info("database_closed")
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("database_close_failed", extra={"error": str(exc)[:80]})

    logger.info("datascout_shutdown_complete")


# ══════════════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI:
        Fully configured application instance, ready for ``uvicorn``.
    """
    app = FastAPI(
        title="DATASCOUT",
        description=(
            "Agentic ML dataset discovery and recommendation system. "
            "Searches HuggingFace, Kaggle, and OpenML with hybrid "
            "semantic + keyword ranking."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    try:
        from datascout.api.routes.search_v2 import router as search_router  # noqa: PLC0415
        app.include_router(search_router)
        logger.debug("search_v2_router_registered")
    except Exception as exc:
        logger.error("failed_to_register_search_router", extra={"error": str(exc)})

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

app = create_app()

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "datascout.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=os.environ.get("ENVIRONMENT", "production") == "development",
    )