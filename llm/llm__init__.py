"""
datascout.llm
─────────────────────────────────────────────────────────────────────────────
LLM layer public API — Gemini only (hackathon requirement).

FIX (v3.3.0): build_fallback_chain now creates THREE providers:
  1. GeminiProvider(gemini-2.5-flash)       — primary, 15 RPM free tier
  2. GeminiProvider(gemini-2.5-flash-lite)  — fallback, 30 RPM, SEPARATE bucket
  3. MockLLMProvider                        — always works, returns templates

Flash and Flash-Lite have completely separate rate limit quotas at Google.
When Flash hits 15 RPM, Flash-Lite still has its own 30 RPM available.
This effectively triples throughput with zero extra cost and stays 100%
Gemini — fully hackathon compliant.

Both deprecated models (2.0-flash, 1.5-flash) are auto-upgraded in
GeminiProvider._resolve_model() so the .env doesn't need to change.
"""

from __future__ import annotations

import logging
from typing import Optional

from datascout.infrastructure.config.settings import get_settings

from .base import BaseLLMProvider, FallbackChain, LLMResponse, compute_cost_usd, make_request_id
from .gemini_provider import GeminiProvider
from .mock_provider import MockLLMProvider
from .prompts.evaluator import EvaluationResult, ResultSetEvaluator
from .prompts.explainer import DatasetExplainer

logger = logging.getLogger("datascout.llm")


def build_fallback_chain(include_mock: bool = True) -> FallbackChain:
    """
    Build standard production fallback chain.

    Chain order:
      1. gemini-2.5-flash      (primary — best quality, 15 RPM free)
      2. gemini-2.5-flash-lite (fallback — separate 30 RPM bucket, still Gemini)
      3. MockLLMProvider       (last resort — returns template-based insights)

    FallbackChain calls is_available() before each attempt.
    When Flash is rate-limited, its is_available() returns False immediately
    — no waiting, no retrying — and the chain moves to Flash-Lite.
    When Flash-Lite is also rate-limited, Mock takes over instantly.
    Users always get a response. The only difference is explanation quality.
    """
    settings = get_settings()
    providers: list[BaseLLMProvider] = []

    # ── Provider 1: Gemini 2.0 Flash Lite (primary — stable free tier) ─────────
    # gemini-1.5-flash is deprecated as of 2026. gemini-2.0-flash-lite is the
    # current stable free-tier model with 1500 RPD / 30 RPM on free plan.
    flash = GeminiProvider(
        api_key=settings.google_api_key,
        model="gemini-2.5-flash",  
        timeout=settings.llm_timeout,
    )
    providers.append(flash)
    logger.info("llm_chain_provider_added", extra={
        "provider": "gemini", "model": "gemini-2.5-flash",
        "available": flash.is_available(), "role": "primary",
    })

    # ── Provider 2: Gemini 2.0 Flash (second free-tier slot, separate RPM bucket) ─
    # gemini-2.0-flash is the non-lite version — slightly higher quality,
    # still free tier with separate RPM quota from flash-lite.
    flash_v2 = GeminiProvider(
        api_key=settings.google_api_key,
        model="gemini-2.0-flash",        # separate RPM bucket from flash-lite
        timeout=settings.llm_timeout,
    )
    providers.append(flash_v2)
    logger.info("llm_chain_provider_added", extra={
        "provider": "gemini", "model": "gemini-2.0-flash",
        "available": flash_v2.is_available(), "role": "gemini_fallback",
    })

    # ── Provider 3: Mock (always available last resort) ───────────────────────
    if include_mock:
        mock = MockLLMProvider(simulated_latency_ms=0)
        providers.append(mock)
        logger.info("llm_chain_provider_added", extra={
            "provider": "mock", "available": True, "role": "last_resort",
        })

    return FallbackChain(providers=providers)


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "FallbackChain",
    "GeminiProvider",
    "MockLLMProvider",
    "DatasetExplainer",
    "ResultSetEvaluator",
    "EvaluationResult",
    "build_fallback_chain",
    "compute_cost_usd",
    "make_request_id",
]