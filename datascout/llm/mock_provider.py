"""
datascout.llm.mock_provider
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Deterministic mock LLM provider for tests + fallback.

SYSTEM DESIGN DECISIONS:

  1. WHY deterministic output over random mock?
     - Tests must be reproducible — random output breaks assertions
     - Fallback scenario in production: template output is better than crash
     - Determinism means: same prompt_hash → same mock response always

  2. WHY mock generates realistic StructuredExplanation-shaped output?
     - Downstream parsers (explainer.py) parse the raw_text
     - If mock returns garbage, parser tests are invalid
     - Mock output must survive the same parsing pipeline as real LLM output

  3. WHY configurable latency simulation?
     - Unit tests: latency_ms=0 for speed
     - Integration tests: latency_ms=100 to catch timeout handling
     - Performance tests: latency_ms=3000 to simulate slow LLM

  4. WHY always is_available() = True?
     - Mock is the last-resort fallback — it must never be unavailable
     - If mock is unavailable, LLMFallbackExhaustedError fires — pipeline fails
     - No external dependency: if Python runs, mock runs

  5. WHY configurable failure injection?
     - Test fallback chain: set should_fail=True on Claude → verify OpenAI used
     - Test LLMFallbackExhaustedError: set should_fail=True on all providers
     - Production: should_fail always False

PERFORMANCE ANALYSIS:
  - complete(): O(1) — no I/O, pure in-memory
  - With simulated_latency_ms: adds exactly that many ms (asyncio.sleep)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from datascout.contracts.errors.exceptions import LLMError

from .base import BaseLLMProvider, LLMResponse, make_request_id

logger = logging.getLogger("datascout.llm.mock_provider")

# Template structured explanation — valid JSON that explainer.py can parse
# Template for DatasetInsight — keys must match _parse_dataset_insight_response expectations
_MOCK_DATASET_INSIGHT_TEMPLATE = """\
{{
  "why_ranked": "Ranked #{rank} based on strong task alignment (composite score: {score_hint}). This dataset closely matches your query and has good metadata coverage.",
  "strengths": [
    "High task relevance score — covers the core domain of your query",
    "Adequate dataset size for training a baseline model",
    "Recently updated — data reflects current state of the field"
  ],
  "weaknesses": [
    "Review license terms before commercial or production deployment",
    "Perform exploratory data analysis to verify class balance"
  ],
  "usability_notes": [
    "Download and inspect a sample before committing to full training",
    "Check train/test split availability before starting experiments"
  ],
  "score_narrative": "Composite score {score_hint} — strongest signals are task relevance and data quality. Query match reflects keyword overlap with dataset description.",
  "confidence_note": "Medium confidence — review the top 3 results before deciding. Score gap between candidates is moderate."
}}"""

# Template for ResearchContext — keys must match _parse_research_context_response expectations
_MOCK_RESEARCH_CONTEXT_TEMPLATE = """\
{{
  "ecosystem_summary": "Found {total} candidate datasets for your query. Results span multiple providers. Top datasets show reasonable metadata coverage and task alignment.",
  "coverage_observations": [
    "Multiple dataset sources available — compare across providers for best fit",
    "Most top results have adequate metadata for initial evaluation"
  ],
  "quality_trends": [
    "Check license clarity — some datasets have unspecified licensing",
    "Dataset size varies significantly — verify minimum row count for your model"
  ],
  "follow_up_searches": [
    "annotated {query_word} dataset",
    "{query_word} benchmark",
    "{query_word} evaluation labels"
  ],
  "research_gaps": [
    "Verify task type labels are present in the actual data files",
    "Check for temporal coverage if your use case is time-sensitive"
  ],
  "provider_notes": [
    "Results from Kaggle, HuggingFace, and OpenML — each has different metadata completeness"
  ]
}}"""


# Template for BATCHED insights — returns a JSON array, one object per dataset.
# Used by intelligence_engine._generate_batch_insights() which sends all datasets
# in one prompt and expects a JSON array back. Mock must match this format exactly.
_MOCK_BATCH_INSIGHT_TEMPLATE = """[{items}]"""

_MOCK_BATCH_ITEM_TEMPLATE = """{{
  "dataset_id": "{dataset_id}",
  "why_ranked": "Ranked #{rank} based on strong task alignment (composite score: {score_hint}). This dataset closely matches your query and has good metadata coverage.",
  "strengths": [
    "High task relevance score — covers the core domain of your query",
    "Good metadata coverage for initial evaluation",
    "Recently updated — data reflects current state of the field"
  ],
  "weaknesses": [
    "Review license terms before commercial or production deployment",
    "Perform exploratory data analysis to verify class balance"
  ],
  "usability_notes": [
    "Download and inspect a sample before committing to full training",
    "Check train/test split availability before starting experiments"
  ],
  "score_narrative": "Composite score {score_hint} — strongest signals are task relevance and data quality. Query match reflects keyword overlap with dataset description.",
  "confidence_note": "Medium confidence — review the top 3 results before deciding."
}}"""



# Combined template — used by _generate_combined() which issues ONE call
# returning both dataset_insights array AND ecosystem fields.
_MOCK_COMBINED_TEMPLATE_HEAD = """{{"dataset_insights": [{items}],"""
_MOCK_COMBINED_TEMPLATE_TAIL = """
  "ecosystem_summary": "Found {total} candidate datasets for your query. Results span {source_count} source(s). Top datasets show reasonable metadata coverage and task alignment.",
  "follow_up_searches": [
    "annotated {query_word} dataset",
    "{query_word} benchmark",
    "{query_word} evaluation labels"
  ],
  "research_gaps": [
    "Verify task type labels are present in the actual data files",
    "Check for temporal coverage if your use case is time-sensitive"
  ]
}}"""
# Legacy evaluation template kept for backward compatibility
_MOCK_EVALUATION_TEMPLATE = """\
{{
  "quality_sufficient": true,
  "quality_score": 0.68,
  "issues": [],
  "recommendation": "Results meet minimum quality threshold.",
  "reasoning": "Mock evaluation template."
}}"""


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock LLM provider.

    Use cases:
    1. Unit tests — fast, zero-latency, fully deterministic
    2. Integration tests — configurable latency + failure injection
    3. Production fallback — template output is better than pipeline crash
    4. CI/CD — no API keys required in test environment
    """

    def __init__(
        self,
        simulated_latency_ms: int = 0,
        should_fail: bool = False,
        fail_with: Optional[type] = None,
    ) -> None:
        """
        Args:
            simulated_latency_ms: Artificial delay in ms (0 = instant for tests).
            should_fail:          If True, complete() raises an LLMError.
            fail_with:            Exception class to raise when should_fail=True.
                                  Defaults to LLMError.
        """
        self._latency_ms = simulated_latency_ms
        self._should_fail = should_fail
        self._fail_with = fail_with or LLMError
        self._call_count: int = 0
        self._logger = logging.getLogger("datascout.llm.mock_provider")

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        """Always True — mock has no external dependencies. O(1)."""
        return True

    @property
    def call_count(self) -> int:
        """Test helper: how many times complete() was called."""
        return self._call_count

    def reset(self) -> None:
        """Test helper: reset call counter and failure state."""
        self._call_count = 0

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Return deterministic mock response.
        O(1) — no I/O.

        Raises:
            LLMError (or configured fail_with type): if should_fail=True.
        """
        self._call_count += 1

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        if self._should_fail:
            # Construct with correct signature per exception type
            exc_cls = self._fail_with
            exc_name = type.__name__ if isinstance(self._fail_with, type) else type(self._fail_with).__name__
            cls_name = self._fail_with.__name__ if isinstance(self._fail_with, type) else type(self._fail_with).__name__
            if cls_name == "LLMTimeoutError":
                raise self._fail_with(model="mock", timeout_s=30.0)
            elif cls_name == "LLMRateLimitError":
                raise self._fail_with(model="mock", retry_after=60.0)
            elif cls_name == "LLMContextOverflowError":
                raise self._fail_with(model="mock", tokens=200001, max_tokens=180000)
            elif cls_name == "LLMFallbackExhaustedError":
                raise self._fail_with(models=["mock"])
            else:
                # LLMError base or unknown subclass
                raise self._fail_with(
                    message=f"Mock provider configured to fail (call #{self._call_count})",
                    model="mock",
                )

        # Deterministic hash for reproducible output per unique prompt — O(prompt_len)
        query_hash = hashlib.sha256(
            (system_prompt[:100] + user_prompt[:100]).encode()
        ).hexdigest()[:8]

        # Detect call type from system prompt keywords.
        # Order matters: be specific first to avoid false positives.
        sys_lower = system_prompt.lower()
        user_lower = user_prompt.lower()

        # Combined call — _generate_combined() returns dataset_insights + ecosystem in one call.
        # Must be detected FIRST because its system prompt contains both batch and ecosystem keywords.
        is_combined = (
            "dataset_insights" in sys_lower
            and ("ecosystem_summary" in sys_lower or "follow_up_searches" in sys_lower)
        )
        # Batch insight: called by _generate_batch_insights — expects JSON array
        # Detected by the batch system prompt marker
        is_batch_insight = (
            not is_combined
            and (
                "json array" in sys_lower
                or "one object per dataset" in sys_lower
                or ("list," in sys_lower and "array" in sys_lower)
            )
        )
        # Research context: looks at the ecosystem/landscape across multiple results
        is_research_context = (
            not is_combined
            and not is_batch_insight
            and (
                "ecosystem" in sys_lower
                or "research context" in sys_lower
                or ("landscape" in sys_lower and "dataset" in sys_lower)
                or ("coverage" in sys_lower and "gaps" in sys_lower)
            )
        )
        # Evaluation: explicit quality scoring call (NOT the insight explainer)
        is_evaluation = (
            not is_batch_insight
            and not is_research_context
            and ("quality_sufficient" in sys_lower or "quality score" in sys_lower
                 or ("evaluat" in sys_lower and "rank" not in sys_lower))
        )
        # Default: single dataset insight
        is_dataset_insight = not is_combined and not is_batch_insight and not is_evaluation and not is_research_context

        score_hint = "0.73"

        # Extract query preview for more realistic mock text
        import re as _re
        # Try to extract from "Query: <text>" pattern first
        query_match = _re.search(r"query[:\s]+([a-z][a-z\s]{5,60}?)(?:[.\n,]|$)", user_lower)
        if query_match:
            query_preview = query_match.group(1).strip()
        else:
            query_preview = "your use case"
        # Pick the longest meaningful word (skip stopwords)
        stopwords = {"about", "based", "using", "with", "from", "that", "this", "which", "have", "your"}
        words = [w for w in query_preview.split() if len(w) > 4 and w not in stopwords]
        query_word = max(words, key=len) if words else "plant-disease"

        # Extract rank if present
        rank_match = _re.search(r"rank[ed #]*(\d+)", user_lower)
        rank = rank_match.group(1) if rank_match else "1"

        # Extract total candidates
        total_match = _re.search(r"(\d+)\s+candidate", user_lower)
        total = total_match.group(1) if total_match else "40"

        if is_combined:
            import re as _re3
            ids = _re3.findall(r"id:\s*([^\)]+)\)", user_prompt)
            if not ids:
                ids = [f"rank_{i+1}" for i in range(3)]
            items = []
            for i, did in enumerate(ids):
                items.append(_MOCK_BATCH_ITEM_TEMPLATE.format(
                    dataset_id=did.strip(),
                    rank=i + 1,
                    score_hint=score_hint,
                ))
            source_match = _re.search(r"sources:\s*([^\n]+)", user_lower)
            source_count = source_match.group(1).strip() if source_match else "multiple"
            head = _MOCK_COMBINED_TEMPLATE_HEAD.format(items=",\n".join(items))
            tail = _MOCK_COMBINED_TEMPLATE_TAIL.format(
                total=total,
                source_count=source_count,
                query_word=query_word,
            )
            raw_text = head + tail
        elif is_evaluation:
            raw_text = _MOCK_EVALUATION_TEMPLATE.format(query_hash=query_hash)
        elif is_research_context:
            raw_text = _MOCK_RESEARCH_CONTEXT_TEMPLATE.format(
                query=query_preview[:40],
                query_word=query_word,
                total=total,
            )
        elif is_batch_insight:
            # Extract dataset IDs from user prompt (DATASET #1 (id: ...) pattern)
            import re as _re2
            ids = _re2.findall(r"id:\s*([^\)]+)\)", user_prompt)
            if not ids:
                # Fallback — generate dummy ids
                ids = [f"rank_{i+1}" for i in range(5)]
            items = []
            for i, did in enumerate(ids):
                items.append(_MOCK_BATCH_ITEM_TEMPLATE.format(
                    dataset_id=did.strip(),
                    rank=i + 1,
                    score_hint=score_hint,
                ))
            raw_text = _MOCK_BATCH_INSIGHT_TEMPLATE.format(items=",\n".join(items))
        else:
            # Default: single dataset insight
            raw_text = _MOCK_DATASET_INSIGHT_TEMPLATE.format(
                rank=rank,
                score_hint=score_hint,
                query_preview=query_preview[:40],
            )

        request_id = make_request_id()
        prompt_tokens = (len(system_prompt) + len(user_prompt)) // 4
        completion_tokens = len(raw_text) // 4

        self._logger.debug(
            "mock_llm_complete",
            extra={
                "request_id": request_id,
                "call_count": self._call_count,
                "is_evaluation": is_evaluation,
                "query_hash": query_hash,
                "simulated_latency_ms": self._latency_ms,
            },
        )

        # FIX v3.5.0: set is_degraded=True and fallback_used=True on mock responses.
        # Previously mock returned is_degraded=False, making intelligence_engine
        # accept template text as genuine Gemini output — operators had no signal
        # that Gemini was never called. With is_degraded=True the engine falls to
        # its deterministic fallback path and pipeline_degraded=True is surfaced
        # in the API response, alerting operators to fix the Gemini key/model.
        return LLMResponse(
            raw_text=raw_text,
            model_name="mock",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,
            duration_ms=self._latency_ms,
            request_id=request_id,
            called_at=datetime.now(tz=timezone.utc),
            fallback_used=True,
            fallback_reason="mock_provider_active",
            is_degraded=True,
        )