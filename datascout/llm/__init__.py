"""
datascout.llm
─────────────────────────────────────────────────────────────────────────────
LLM layer public API — Gemini only.

Exports:
- BaseLLMProvider:  Abstract provider interface
- LLMResponse:      Raw provider output (pre-parsing)
- FallbackChain:    Ordered fallback chain (Gemini → Mock)
- GeminiProvider:   Google Gemini implementation
- MockLLMProvider:  Deterministic mock for tests and fallback
- DatasetExplainer:    High-level explanation orchestrator
- ResultSetEvaluator:  High-level evaluation orchestrator
- EvaluationResult:    Evaluation output contract
- build_fallback_chain: Factory for standard production chain
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
    Factory: build standard production fallback chain.

    Order: Gemini → Mock (if include_mock=True)
    """
    settings = get_settings()
    providers: list[BaseLLMProvider] = []

    gemini = GeminiProvider(
        api_key=settings.google_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
    providers.append(gemini)
    logger.info("llm_chain_provider_added", extra={
        "provider": "gemini", "available": gemini.is_available(), "model": gemini.model_name,
    })

    if include_mock:
        mock = MockLLMProvider(simulated_latency_ms=0)
        providers.append(mock)
        logger.info("llm_chain_provider_added", extra={"provider": "mock", "available": True})

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