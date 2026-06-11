"""
datascout.llm.gemini_provider
──────────────────────────────────────────────────────────────────────────────
Google Gemini LLM provider via direct httpx calls.

FIXES:
  v3.2.0 — module-level rate-limit backoff (_rate_limit_until)
  v3.4.0 — switched to gemini-1.5-flash (2.5-flash causes 503 overload errors)
           fallback to gemini-1.5-flash-8b (separate quota, lighter + faster)
           per-model rate limit tracking so flash and flash-lite track independently
           exponential backoff on retries instead of fixed 65s wait
  v3.8.0 — BUG FIX: 503 retry success path returned a raw tuple (text, dict)
           instead of a proper LLMResponse object.  FallbackChain and
           ResearchIntelligenceEngine both call .raw_text / .model_name on the
           return value, so the tuple caused an AttributeError crash on every
           successful 503 retry.  Fixed by building a full LLMResponse, matching
           the happy-path return at the bottom of complete().
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from datascout.contracts.errors.exceptions import (
    LLMContextOverflowError, LLMError, LLMRateLimitError, LLMTimeoutError,
)
from datascout.infrastructure.config.settings import get_settings
from .base import BaseLLMProvider, LLMResponse, make_request_id

logger = logging.getLogger("datascout.llm.gemini_provider")

_GEMINI_BASE_URL         = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL           = "gemini-2.0-flash-lite"   # FIX: 1.5-flash is deprecated; 2.0-flash-lite is stable free-tier
# _FALLBACK_MODEL = deprecated; use FallbackChain in build_fallback_chain() instead
_GEMINI_MAX_INPUT_TOKENS = 950_000
_CHARS_PER_TOKEN_EST     = 4
_SUCCESS_FINISH_REASONS  = {"STOP", "MAX_TOKENS"}

# ── Per-model rate-limit tracking ─────────────────────────────────────────────
# Each model has its own rate limit bucket at Google.
# Track them separately so flash-lite can work while flash is rate-limited.
_model_rate_limit_until: dict[str, float] = {}
_rate_limit_lock = asyncio.Lock()


def _is_model_rate_limited(model: str) -> bool:
    until = _model_rate_limit_until.get(model, 0.0)
    return until > 0 and time.monotonic() < until


def _seconds_until_available(model: str) -> float:
    until = _model_rate_limit_until.get(model, 0.0)
    return max(0.0, until - time.monotonic())


async def _set_model_rate_limit(model: str, retry_after_s: int) -> None:
    async with _rate_limit_lock:
        backoff = max(retry_after_s, 60) + 5   # always at least 65s
        new_until = time.monotonic() + backoff
        existing = _model_rate_limit_until.get(model, 0.0)
        if new_until > existing:
            _model_rate_limit_until[model] = new_until
            logger.warning(
                "gemini_rate_limit_backoff_set",
                extra={"model": model, "backoff_s": backoff,
                       "available_at": datetime.fromtimestamp(
                           time.time() + backoff, tz=timezone.utc
                       ).strftime("%H:%M:%S UTC")},
            )


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider. NEVER ranks or scores — only generates explanations.

    Tracks rate limits per model instance so Flash and Flash-Lite can be
    used as independent providers in the FallbackChain without interfering
    with each other's rate limit state.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._settings = get_settings()
        self._api_key  = api_key or getattr(self._settings, "google_api_key", None)

        # Resolve model — upgrade deprecated models automatically
        raw_model = model or getattr(self._settings, "llm_model", _DEFAULT_MODEL)
        self._model = self._resolve_model(raw_model)

        self._timeout = (
            timeout if timeout is not None
            else getattr(self._settings, "llm_timeout", 25.0)
        )
        logger.info(
            "gemini_provider_initialized",
            extra={"model": self._model, "available": self.is_available()},
        )

    @staticmethod
    def _resolve_model(raw: str) -> str:
        """
        Upgrade deprecated model names automatically.
        gemini-2.0-flash was deprecated Feb 2026, retired March 3 2026.
        gemini-1.5-flash is also deprecated.
        """
        DEPRECATED = {
            # Only the experimental variants are fully retired
            "gemini-2.0-flash-exp":      "gemini-2.0-flash",       # -exp retired; use stable
            "gemini-1.5-flash":          "gemini-2.0-flash-lite",  # 1.5 line deprecated
            "gemini-1.5-flash-8b":       "gemini-2.0-flash-lite",  # 1.5 line deprecated
            "gemini-1.5-pro":            "gemini-2.0-flash",       # 1.5 line deprecated
            "gemini-pro":                "gemini-2.0-flash-lite",  # very old, deprecated
        }
        resolved = DEPRECATED.get(raw, raw)
        if resolved != raw:
            logger.warning(
                "gemini_deprecated_model_upgraded",
                extra={"from": raw, "to": resolved},
            )
        if not resolved.startswith("gemini"):
            return _DEFAULT_MODEL
        return resolved

    def is_available(self) -> bool:
        """
        Returns False when:
        - API key not configured
        - This model is currently in its rate-limit backoff window

        FallbackChain calls this before attempting the request.
        When Flash is rate-limited, Flash-Lite's is_available() returns True
        because they track separately — so the chain falls through correctly.
        """
        if not self._api_key:
            return False
        if _is_model_rate_limited(self._model):
            remaining = round(_seconds_until_available(self._model), 1)
            logger.debug(
                "gemini_skipped_rate_limited",
                extra={"model": self._model, "remaining_s": remaining},
            )
            return False
        return True

    @property
    def provider_name(self) -> str:
        return f"gemini_{self._model.replace('-', '_')}"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMError("GEMINI_NO_API_KEY", "GOOGLE_API_KEY not configured.")

        # is_available() already checked by FallbackChain, but guard here too
        if _is_model_rate_limited(self._model):
            raise LLMRateLimitError(
                f"{self._model} rate-limited for "
                f"{_seconds_until_available(self._model):.0f}s more."
            )

        req_id = request_id or make_request_id()
        start  = time.perf_counter()

        # Preflight context check
        est_tokens = (len(user_prompt) + len(system_prompt or "")) // _CHARS_PER_TOKEN_EST
        if est_tokens > _GEMINI_MAX_INPUT_TOKENS:
            raise LLMContextOverflowError(
                f"Prompt ~{est_tokens} tokens exceeds safe limit."
            )

        payload = self._build_payload(user_prompt, system_prompt, max_tokens, temperature)
        url = (
            f"{_GEMINI_BASE_URL}/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self._model} timed out after {self._timeout}s"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError("GEMINI_CONNECTION_ERROR", str(exc)) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        # ── Rate limit ────────────────────────────────────────────────────────
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            await _set_model_rate_limit(self._model, retry_after)
            raise LLMRateLimitError(
                f"{self._model} rate limit hit. Retry after {retry_after}s."
            )

        # ── Bad request ───────────────────────────────────────────────────────
        if response.status_code == 400:
            msg = _safe_json(response).get("error", {}).get("message", "Bad request")
            if "token" in msg.lower() or "context" in msg.lower():
                raise LLMContextOverflowError(f"Context overflow: {msg}")
            raise LLMError("GEMINI_BAD_REQUEST", msg)

        # ── Model not found — auto-upgrade ────────────────────────────────────
        if response.status_code == 404:
            logger.warning(
                "gemini_model_not_found",
                extra={"model": self._model, "upgrade_to": _DEFAULT_MODEL},
            )
            self._model = _DEFAULT_MODEL
            url = (
                f"{_GEMINI_BASE_URL}/models/{self._model}:generateContent"
                f"?key={self._api_key}"
            )
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
            except httpx.RequestError as exc:
                raise LLMError("GEMINI_CONNECTION_ERROR", str(exc)) from exc

        if response.status_code not in (200, 201):
            msg = _safe_json(response).get("error", {}).get("message", response.text[:200])

            # ── 503 Service Unavailable — model overloaded, retry with backoff ──
            # Gemini returns 503 when the model is temporarily overloaded.
            # Unlike 429 (rate limit), 503 should be retried immediately with
            # short exponential backoff — not treated as a hard failure.
            if response.status_code == 503:
                for _attempt in range(3):
                    _wait = 2 ** _attempt        # 1s, 2s, 4s
                    logger.warning(
                        "gemini_503_retry",
                        extra={
                            "model": self._model,
                            "attempt": _attempt + 1,
                            "wait_s": _wait,
                            "msg": msg[:80],
                        },
                    )
                    await asyncio.sleep(_wait)
                    try:
                        async with httpx.AsyncClient(timeout=self._timeout) as _client:
                            _resp = await _client.post(url, json=payload)
                        if _resp.status_code == 200:
                            response = _resp
                            body = _safe_json(response)
                            text, input_tokens, output_tokens = self._parse_response(body, req_id)
                            # FIX v3.8.0: was `return text, {...}` — a raw tuple.
                            # FallbackChain and ResearchIntelligenceEngine access
                            # .raw_text / .model_name on the return value, so a tuple
                            # caused AttributeError on every successful 503 retry.
                            # Now we build a proper LLMResponse, identical to the
                            # happy-path return at the bottom of complete().
                            latency_ms = int((time.perf_counter() - start) * 1000)
                            cost_usd = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
                            logger.info(
                                "gemini_request_completed",
                                extra={
                                    "request_id": req_id, "model": self._model,
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "latency_ms": latency_ms,
                                    "cost_usd": round(cost_usd, 6),
                                    "via_503_retry": True,
                                    "attempt": _attempt + 1,
                                },
                            )
                            return LLMResponse(
                                raw_text=text, model_name=self._model,
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                cost_usd=cost_usd,
                                duration_ms=latency_ms,
                                request_id=req_id,
                                called_at=datetime.now(timezone.utc),
                            )
                        elif _resp.status_code != 503:
                            # Different error — stop retrying, fall through to generic handler
                            response = _resp
                            msg = _safe_json(_resp).get("error", {}).get("message", _resp.text[:200])
                            break
                        # still 503 — keep retrying
                        msg = _safe_json(_resp).get("error", {}).get("message", _resp.text[:200])
                    except httpx.RequestError:
                        pass  # network error — try again
                # All retries exhausted
                raise LLMError(
                    "GEMINI_SERVICE_UNAVAILABLE",
                    f"Gemini model overloaded after 3 retries: {msg}",
                )

            raise LLMError(f"GEMINI_HTTP_{response.status_code}", msg)

        # ── Parse response ────────────────────────────────────────────────────
        body = _safe_json(response)
        text, input_tokens, output_tokens = self._parse_response(body, req_id)
        # Flash pricing: $0.075/1M input, $0.30/1M output
        cost_usd = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000

        logger.info(
            "gemini_request_completed",
            extra={
                "request_id": req_id, "model": self._model,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "latency_ms": latency_ms, "cost_usd": round(cost_usd, 6),
            },
        )
        return LLMResponse(
            raw_text=text, model_name=self._model,
            prompt_tokens=input_tokens, completion_tokens=output_tokens,
            cost_usd=cost_usd, duration_ms=latency_ms,
            request_id=req_id, called_at=datetime.now(timezone.utc),
        )

    def _build_payload(
        self, user_prompt: str, system_prompt: Optional[str],
        max_tokens: int, temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            ],
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _parse_response(
        self, body: dict[str, Any], req_id: str
    ) -> tuple[str, int, int]:
        candidates = body.get("candidates", [])
        if not candidates:
            raise LLMError("GEMINI_EMPTY_CANDIDATES", "No candidates returned.")
        candidate    = candidates[0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if finish_reason == "SAFETY":
            raise LLMError("GEMINI_SAFETY_BLOCK", "Response blocked for safety reasons.")
        if finish_reason not in _SUCCESS_FINISH_REASONS:
            logger.warning(
                "gemini_unexpected_finish_reason",
                extra={"request_id": req_id, "finish_reason": finish_reason},
            )
        parts = candidate.get("content", {}).get("parts", [])
        text  = "".join(p.get("text", "") for p in parts if "text" in p).strip()
        if not text:
            raise LLMError(
                "GEMINI_EMPTY_TEXT",
                f"Empty text (finishReason={finish_reason}).",
            )
        usage = body.get("usageMetadata", {})
        return (
            text,
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
        )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {}