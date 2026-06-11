"""
datascout.agents.scout_agent
─────────────────────────────
ScoutAgent — the decision-making layer above the retrieval pipeline.

AGENT LOOP (ReAct-style, 3 max iterations):
  1. THINK:   Analyse query, decide which sources to use, estimate difficulty
  2. ACT:     Call adapters in parallel via _attempt_retrieval()
  3. OBSERVE: Count results; if < MIN_RESULTS_THRESHOLD and iterations
              remain → broaden and retry
  4. RETURN:  Pass results to EvaluatorPipeline

QUERY BROADENING STRATEGY (when results < min_threshold):
  - Iteration 1: Use original query, all requested sources
  - Iteration 2: Strip domain-specific terms (words > 10 chars), keep top 3
  - Iteration 3: Keep only the first 2 words of the original query

AGENT STATE:
  - Tracks which sources were tried and which succeeded
  - Records adapter failures with human-readable reason codes
  - Produces AgentTrace for transparency in API responses

This agent does NOT do multi-hop reasoning or tool calling in v1.
It is a focused retry-with-broadening loop.  LLM involvement is optional
and only used for query broadening (falls back to rule-based if LLM
unavailable).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from datascout.contracts.errors.exceptions import (
    AdapterAuthError,
    AdapterConnectionError,
    AdapterRateLimitError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterFailure",
    "AgentTrace",
    "ScoutAgent",
]


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AdapterFailure:
    """Record of a single adapter's failed attempt.

    Attributes
    ----------
    source:
        The source name that failed (e.g. ``"kaggle"``, ``"huggingface"``).
    reason:
        Machine-readable reason code.  One of: ``"auth_failed"``,
        ``"timeout"``, ``"rate_limited"``, ``"not_installed"``,
        ``"unknown"``.
    error_message:
        Raw exception message, truncated to 300 chars for log safety.
    attempted_at:
        UTC timestamp when the attempt was made.
    """

    source: str
    reason: str
    error_message: str
    attempted_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass
class AgentTrace:
    """Full audit trail for a single ScoutAgent.search() call.

    This object is attached to every API response so that users (and
    developers) can understand why they got the results they did without
    needing to inspect server logs.

    Attributes
    ----------
    query_original:
        The raw query string as submitted by the caller.
    query_used:
        The (possibly broadened) query string that was actually sent to the
        adapters in the final iteration.
    iterations:
        Number of agent loop iterations executed (1–MAX_ITERATIONS).
    sources_attempted:
        All source names that were attempted at least once.
    sources_succeeded:
        Source names that returned at least one result.
    adapter_failures:
        Failure records for each adapter+iteration that produced an error.
    total_retrieved:
        Total number of raw datasets returned across all succeeded sources.
    broadening_applied:
        ``True`` if the query was broadened at least once.
    broadening_reason:
        Human-readable explanation of why broadening was applied, or ``""``
        if it was not.
    agent_version:
        Semver string for the agent implementation.
    """

    query_original: str
    query_used: str
    iterations: int
    sources_attempted: List[str] = field(default_factory=list)
    sources_succeeded: List[str] = field(default_factory=list)
    adapter_failures: List[AdapterFailure] = field(default_factory=list)
    total_retrieved: int = 0
    broadening_applied: bool = False
    broadening_reason: str = ""
    agent_version: str = "1.0.0"

    def to_user_message(self) -> str:
        """Generate a human-friendly status message for the API response.

        Rules
        -----
        - All sources failed → explain each source's issue in plain language.
        - Some sources failed → name the ones that worked and note issues.
        - Broadening was applied → tell the user what query was used.
        - Everything worked → return ``""`` (no need to explain success).

        The message **never** uses technical jargon such as "adapter",
        "circuit breaker", "timeout exception", or "503".

        Returns
        -------
        str:
            User-facing message string, or ``""`` if nothing went wrong.
        """
        parts: List[str] = []

        if self.broadening_applied and self.broadening_reason:
            parts.append(
                f"Your search was expanded to \"{self.query_used}\" "
                f"to find more results."
            )

        if self.adapter_failures:
            failed_sources = {f.source for f in self.adapter_failures}
            succeeded = set(self.sources_succeeded)
            all_failed = len(succeeded) == 0 and len(failed_sources) > 0

            issue_parts: List[str] = []
            # Deduplicate failures by source (use the most recent for each)
            seen: dict[str, AdapterFailure] = {}
            for f in self.adapter_failures:
                seen[f.source] = f  # last failure for each source wins
            for src, f in seen.items():
                if f.reason == "auth_failed":
                    issue_parts.append(
                        f"{src.title()}: API credentials are not configured"
                    )
                elif f.reason == "not_installed":
                    issue_parts.append(
                        f"{src.title()}: required package is not installed"
                    )
                elif f.reason == "rate_limited":
                    issue_parts.append(
                        f"{src.title()}: request limit reached — try again shortly"
                    )
                elif f.reason == "timeout":
                    issue_parts.append(
                        f"{src.title()}: service took too long to respond"
                    )
                else:
                    issue_parts.append(f"{src.title()}: temporarily unavailable")

            if all_failed:
                sources_tried = ", ".join(
                    s.title() for s in sorted(failed_sources)
                )
                issues_str = "; ".join(issue_parts)
                parts.append(
                    f"We searched {sources_tried} but could not connect right now. "
                    f"Issues: {issues_str}. "
                    f"Check your API credentials and try again."
                )
            elif issue_parts:
                succeeded_str = ", ".join(s.title() for s in sorted(succeeded))
                issues_str = "; ".join(issue_parts)
                parts.append(
                    f"Results are from {succeeded_str}. "
                    f"Some sources were unavailable: {issues_str}."
                )

        return " ".join(parts).strip()


# ── ScoutAgent ────────────────────────────────────────────────────────────────


class ScoutAgent:
    """Agentic retrieval layer with retry and query broadening.

    ``ScoutAgent`` runs a simple ReAct-style loop:

    1. **Think** — analyse the query and plan which sources to hit.
    2. **Act** — call all requested adapters in parallel.
    3. **Observe** — if too few results, broaden the query and retry.
    4. **Return** — hand results to the evaluation pipeline.

    The agent degrades gracefully at every step.  If all adapters fail, it
    returns ``([], trace)`` where ``trace.to_user_message()`` gives a
    friendly explanation.

    Example
    -------
    >>> agent = ScoutAgent()
    >>> datasets, trace = await agent.search(
    ...     query="tabular classification benchmark",
    ...     sources=["kaggle", "huggingface", "openml"],
    ...     max_results=10,
    ... )
    """

    MIN_RESULTS_THRESHOLD: int = 3
    MAX_ITERATIONS: int = 3

    def __init__(self) -> None:
        """Initialise the agent.

        No dependencies are injected here — everything is imported lazily so
        that missing optional packages (e.g. kaggle, huggingface_hub) only
        cause individual source failures, not a startup crash.
        """
        self._last_adapter_error: str = ""

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        sources: List[str],
        max_results: int,
    ) -> Tuple[List[Any], AgentTrace]:
        """Run the agentic search loop and return results plus trace.

        Parameters
        ----------
        query:
            Raw search query from the user.
        sources:
            List of source names to search (e.g. ``["kaggle", "huggingface"]``).
        max_results:
            Maximum total number of raw datasets to return.

        Returns
        -------
        tuple[list[RawDataset], AgentTrace]:
            Always returned; never raises.  ``datasets`` may be empty if all
            sources failed; ``trace.to_user_message()`` explains why.
        """
        trace = AgentTrace(
            query_original=query,
            query_used=query,
            iterations=0,
        )
        datasets: List[Any] = []
        current_query = query

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            trace.iterations = iteration

            logger.info(
                "scout_agent_iteration",
                extra={
                    "iteration": iteration,
                    "query": current_query[:80],
                    "sources": sources,
                },
            )

            # ACT
            batch = await self._attempt_retrieval(
                query=current_query,
                sources=sources,
                max_results=max_results,
                trace=trace,
            )
            datasets = batch

            # OBSERVE
            if len(datasets) >= self.MIN_RESULTS_THRESHOLD:
                logger.info(
                    "scout_agent_threshold_met",
                    extra={"count": len(datasets), "iteration": iteration},
                )
                break

            if iteration < self.MAX_ITERATIONS:
                broadened = self._broaden_query(query, iteration + 1)
                if broadened != current_query:
                    logger.info(
                        "scout_agent_broadening",
                        extra={
                            "from": current_query[:80],
                            "to": broadened[:80],
                            "iteration": iteration + 1,
                        },
                    )
                    trace.broadening_applied = True
                    trace.broadening_reason = (
                        f"Too few results ({len(datasets)}) — "
                        f"broadened query from '{current_query}' to '{broadened}'"
                    )
                    trace.query_used = broadened
                    current_query = broadened
                else:
                    # Query cannot be broadened further — stop early
                    break
            else:
                logger.info(
                    "scout_agent_max_iterations_reached",
                    extra={"final_count": len(datasets)},
                )

        trace.total_retrieved = len(datasets)
        logger.info(
            "scout_agent_done",
            extra={
                "total": len(datasets),
                "iterations": trace.iterations,
                "succeeded": trace.sources_succeeded,
                "failures": len(trace.adapter_failures),
            },
        )
        return datasets, trace

    # ──────────────────────────────────────────────────────────────────────
    # Private methods
    # ──────────────────────────────────────────────────────────────────────

    async def _attempt_retrieval(
        self,
        query: str,
        sources: List[str],
        max_results: int,
        trace: AgentTrace,
    ) -> List[Any]:
        """One retrieval attempt across all *sources* in parallel.

        Parameters
        ----------
        query:
            The (possibly broadened) query to send to adapters.
        sources:
            Source names to query.
        max_results:
            Per-source result limit passed to each adapter.
        trace:
            Mutable trace object; failures are appended here.

        Returns
        -------
        list[RawDataset]:
            Deduplicated results from all sources that responded.
        """
        per_source_limit = max(max_results, 20)
        tasks: List[asyncio.Task] = []

        for source in sources:
            if source not in trace.sources_attempted:
                trace.sources_attempted.append(source)
            adapter = self._make_adapter(source)
            if adapter is None:
                failure_reason = self._classify_error(self._last_adapter_error)
                trace.adapter_failures.append(
                    AdapterFailure(
                        source=source,
                        reason=failure_reason,
                        error_message=self._last_adapter_error[:300],
                    )
                )
                continue
            tasks.append(
                asyncio.create_task(
                    self._safe_search(adapter, query, per_source_limit, source, trace),
                    name=f"scout_{source}",
                )
            )

        if not tasks:
            return []

        results_nested = await asyncio.gather(*tasks, return_exceptions=False)
        all_datasets: List[Any] = []
        for batch in results_nested:
            all_datasets.extend(batch)

        deduped = self._deduplicate(all_datasets)

        # Record succeeded sources
        for source in sources:
            # A source succeeded if it contributed ≥1 dataset
            source_ids = {
                ds.source for ds in deduped if hasattr(ds, "source")
            }
            if source in source_ids and source not in trace.sources_succeeded:
                trace.sources_succeeded.append(source)

        return deduped

    def _broaden_query(self, query: str, iteration: int) -> str:
        """Return a broadened version of *query* for a given *iteration*.

        Strategy
        --------
        - Iteration 2: Remove words longer than 10 characters (domain
          jargon) and keep the first 3 remaining words.
        - Iteration 3+: Keep only the first 2 words of the original query.

        Parameters
        ----------
        query:
            Original query string.
        iteration:
            Current iteration number (used to select strategy).

        Returns
        -------
        str:
            Broadened query, or the original if no broadening is possible
            (e.g. single-word query).
        """
        words = query.strip().split()
        if len(words) <= 1:
            return query

        if iteration == 2:
            # Remove domain-specific long words, keep top 3
            short_words = [w for w in words if len(w) <= 10]
            if not short_words:
                short_words = words  # fallback — keep all
            return " ".join(short_words[:3])

        # Iteration 3+: first 2 words only
        return " ".join(words[:2])

    def _make_adapter(self, source: str) -> Optional[Any]:
        """Lazily import and instantiate the adapter for *source*.

        Distinguishes between import errors (package not installed),
        authentication errors, and other errors so that the correct reason
        code can be attached to the failure record.

        Parameters
        ----------
        source:
            Source name: ``"kaggle"``, ``"huggingface"``, or ``"openml"``.

        Returns
        -------
        adapter instance or ``None``:
            ``None`` on any error.  ``self._last_adapter_error`` is set to
            the error message for use by the caller.
        """
        self._last_adapter_error = ""
        try:
            if source == "kaggle":
                from datascout.adapters.kaggle_adapter import KaggleAdapter  # noqa: PLC0415
                return KaggleAdapter()
            if source == "huggingface":
                from datascout.adapters.huggingface_adapter import HuggingFaceAdapter  # noqa: PLC0415
                return HuggingFaceAdapter()
            if source == "openml":
                from datascout.adapters.openml_adapter import OpenMLAdapter  # noqa: PLC0415
                return OpenMLAdapter()
            self._last_adapter_error = f"Unknown source: {source}"
            return None
        except ImportError as exc:
            self._last_adapter_error = f"ImportError: {exc}"
            logger.warning(
                "scout_agent_adapter_import_failed",
                extra={"source": source, "error": str(exc)[:200]},
            )
            return None
        except Exception as exc:
            self._last_adapter_error = str(exc)
            logger.warning(
                "scout_agent_adapter_instantiate_failed",
                extra={"source": source, "error": str(exc)[:200]},
            )
            return None

    async def _safe_search(
        self,
        adapter: Any,
        query: str,
        limit: int,
        source: str,
        trace: AgentTrace,
    ) -> List[Any]:
        """Call ``adapter.search()`` safely, catching and classifying errors.

        Parameters
        ----------
        adapter:
            An instantiated adapter object with a ``search()`` coroutine.
        query:
            Query string to pass to the adapter.
        limit:
            Maximum number of results to request.
        source:
            Source name, used for failure attribution.
        trace:
            Mutable trace; failures are appended here.

        Returns
        -------
        list[RawDataset]:
            Results from the adapter, or ``[]`` on error.
        """
        try:
            # Adapters expect a SearchQuery contract object, not raw kwargs
            from datascout.contracts.requests import SearchQuery as AdapterQuery  # noqa: PLC0415
            sq = AdapterQuery(raw_query=query, max_results=limit)

            # Support both async and sync adapters.
            # No outer timeout here — each adapter manages its own timeout internally
            # (KaggleAdapter, HuggingFaceAdapter, OpenMLAdapter all use asyncio.wait_for
            # with their own limits). Adding a second outer timeout causes CancelledError
            # to escape before the adapter's AdapterTimeoutError is raised, breaking
            # the error classification and circuit breaker logic.
            if asyncio.iscoroutinefunction(adapter.search):
                results = await adapter.search(sq)
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: adapter.search(sq),
                )
            return results or []
        except Exception as exc:
            reason = self._classify_error(str(exc), exc)
            trace.adapter_failures.append(
                AdapterFailure(
                    source=source,
                    reason=reason,
                    error_message=str(exc)[:300],
                )
            )
            import traceback as _tb
            print(f"\n[ADAPTER FAILURE] source={source} reason={reason}")
            print(f"[ADAPTER FAILURE] error={str(exc)[:500]}")
            _tb.print_exc()
            logger.warning(
                "scout_agent_adapter_search_failed",
                extra={"source": source, "reason": reason, "error": str(exc)[:500]},
                exc_info=True,
            )
            return []

    def _classify_error(
        self,
        message: str,
        exc: Optional[Exception] = None,
    ) -> str:
        """Map an exception to a machine-readable reason code.

        Parameters
        ----------
        message:
            String representation of the error.
        exc:
            Original exception object (used for isinstance checks).

        Returns
        -------
        str:
            One of: ``"auth_failed"``, ``"timeout"``, ``"rate_limited"``,
            ``"not_installed"``, ``"unknown"``.
        """
        # Check typed adapter exceptions first — most precise classification.
        if exc is not None:
            if isinstance(exc, AdapterAuthError):
                return "auth_failed"
            if isinstance(exc, AdapterTimeoutError) or isinstance(exc, asyncio.TimeoutError):
                return "timeout"
            if isinstance(exc, AdapterRateLimitError):
                return "rate_limited"
            if isinstance(exc, AdapterConnectionError):
                # Connection error with "not installed" detail = package missing
                detail = str(exc).lower()
                if "not installed" in detail or "no module" in detail:
                    return "not_installed"
                return "unknown"
            if isinstance(exc, ImportError):
                return "not_installed"

        msg_lower = message.lower()

        if any(k in msg_lower for k in ("401", "unauthorized", "credentials", "forbidden", "403", "authentication failed")):
            return "auth_failed"
        if any(k in msg_lower for k in ("timeout", "timed out", "read timeout")):
            return "timeout"
        if any(k in msg_lower for k in ("429", "rate limit", "too many requests")):
            return "rate_limited"
        if "importerror" in msg_lower or "no module named" in msg_lower or "not installed" in msg_lower:
            return "not_installed"

        return "unknown"

    def _deduplicate(self, datasets: List[Any]) -> List[Any]:
        """Remove duplicate datasets by ``canonical_id``.

        Parameters
        ----------
        datasets:
            Flat list of ``RawDataset`` objects, possibly containing
            duplicates from multiple sources.

        Returns
        -------
        list[RawDataset]:
            Deduplicated list; first occurrence of each ``canonical_id`` wins.
        """
        seen: set[str] = set()
        out: List[Any] = []
        for ds in datasets:
            cid = getattr(ds, "canonical_id", None)
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            out.append(ds)
        return out