"""
datascout.llm.base
──────────────────────────────────────────────────────────────────────────────
Abstract LLM provider contract + fallback chain.

FIX (v3.1.0): FallbackChain now returns a DEGRADED LLMResponse instead of
raising LLMFallbackExhaustedError when all providers fail.

WHY: Search results are fully computed and ranked BEFORE the LLM runs.
The LLM only adds "why ranked here" explanation text. Raising a 500 error
discards valid results just because an explanation couldn't be generated.
Callers check response.is_degraded and render a "AI insights unavailable"
message instead.
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict

from datascout.contracts.errors.exceptions import LLMFallbackExhaustedError
from datascout.contracts.responses import LLMMetadata

logger = logging.getLogger("datascout.llm.base")


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_ms: int
    request_id: str
    called_at: datetime
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    is_degraded: bool = False  # True when all providers failed — no explanation available

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_llm_metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name, tokens_used=self.total_tokens,
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd, fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason, duration_ms=self.duration_ms,
            request_id=self.request_id, called_at=self.called_at,
        )


class BaseLLMProvider(ABC):
    """LLM NEVER ranks or scores — only explains and evaluates."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str,
                       max_tokens: int = 1024, temperature: float = 0.1) -> LLMResponse: ...

    async def explain(self, system_prompt: str, user_prompt: str,
                      max_tokens: int = 1024) -> LLMResponse:
        return await self.complete(system_prompt=system_prompt, user_prompt=user_prompt,
                                   max_tokens=max_tokens, temperature=0.1)

    async def evaluate(self, system_prompt: str, user_prompt: str,
                       max_tokens: int = 512) -> LLMResponse:
        return await self.complete(system_prompt=system_prompt, user_prompt=user_prompt,
                                   max_tokens=max_tokens, temperature=0.0)


def _make_degraded_response(request_id: str, reason: str) -> LLMResponse:
    """Return graceful empty response when all providers exhausted."""
    return LLMResponse(
        raw_text="", model_name="none",
        prompt_tokens=0, completion_tokens=0, cost_usd=0.0, duration_ms=0,
        request_id=request_id, called_at=datetime.now(timezone.utc),
        fallback_used=True, fallback_reason=reason, is_degraded=True,
    )


class FallbackChain:
    """
    Ordered provider fallback chain. Returns degraded response (not raises)
    when all providers fail, so search results are never discarded.
    """

    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self._providers = providers
        self._logger = logging.getLogger("datascout.llm.fallback_chain")

    @property
    def providers(self) -> list[BaseLLMProvider]:
        return list(self._providers)

    async def complete(self, system_prompt: str, user_prompt: str,
                       max_tokens: int = 1024, temperature: float = 0.1) -> LLMResponse:
        req_id = make_request_id()
        tried: list[str] = []
        last_reason = "no providers available"

        for provider in self._providers:
            if not provider.is_available():
                tried.append(f"{provider.provider_name}:unavailable")
                continue
            try:
                self._logger.info("llm_provider_attempting", extra={
                    "provider": provider.provider_name, "model": provider.model_name,
                    "is_fallback": len(tried) > 0,
                })
                response = await provider.complete(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    max_tokens=max_tokens, temperature=temperature,
                )
                if tried:
                    response = response.model_copy(update={
                        "fallback_used": True,
                        "fallback_reason": f"primary_failed:{tried[0].split(':')[0]}",
                    })
                self._logger.info("llm_provider_success", extra={
                    "provider": provider.provider_name, "tokens": response.total_tokens,
                })
                return response
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                tried.append(f"{provider.provider_name}:{type(exc).__name__}")
                self._logger.warning("llm_provider_failed", extra={
                    "provider": provider.provider_name,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc)[:200],
                    "tried_so_far": tried,
                })
                # Print to stdout so it's visible in uvicorn console
                print(f"\n[LLM FALLBACK] {provider.provider_name}/{provider.model_name} failed: {type(exc).__name__}: {str(exc)[:200]}")

        # All providers failed — return degraded, never raise
        tried_names = [t.split(":")[0] for t in tried]
        self._logger.error("llm_all_providers_failed",
                           extra={"tried": tried, "last_reason": last_reason})
        return _make_degraded_response(req_id, f"all_failed:{','.join(tried_names)}")

    async def explain(self, system_prompt: str, user_prompt: str,
                      max_tokens: int = 1024) -> LLMResponse:
        return await self.complete(system_prompt, user_prompt, max_tokens, 0.1)

    async def evaluate(self, system_prompt: str, user_prompt: str,
                       max_tokens: int = 512) -> LLMResponse:
        return await self.complete(system_prompt, user_prompt, max_tokens, 0.0)


COST_PER_TOKEN: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-4o":                   {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":              {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "mock":                     {"input": 0.0,               "output": 0.0},
}


def compute_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_TOKEN.get(model_name, {"input": 0.0, "output": 0.0})
    return round(prompt_tokens * rates["input"] + completion_tokens * rates["output"], 8)


def make_request_id() -> str:
    return f"llm-{uuid.uuid4().hex[:16]}"