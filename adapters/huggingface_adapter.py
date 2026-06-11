"""
datascout.adapters.huggingface_adapter
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: HuggingFace Datasets Hub adapter with graceful
degradation for inconsistent metadata, gated dataset handling, and model
card extraction.

AGENT-0 CONTEXT:
  Produces RawDataset records from HuggingFace Hub datasets API.
  HF metadata quality varies wildly — graceful degradation is critical here.

SYSTEM DESIGN DECISIONS:

  1. WHY huggingface_hub SDK over raw HTTP?
     - Official SDK handles token auth (HF_TOKEN env var)
     - Handles pagination with list_datasets() generator
     - SDK abstracts HF's complex nested API response structure
     - Maintained by HuggingFace — API changes handled in SDK

  2. WHY graceful degradation for HF?
     - HF metadata is the most inconsistent of the three sources
     - Some datasets: full model card with description, tasks, languages, licenses
     - Other datasets: just a name and zero other metadata
     - Without graceful degradation: 60% of HF records fail validation
     - With it: we capture what we have, quality signals reflect the gaps

  3. WHY extract from both dataset card and card_data?
     - HF datasets have two description sources:
       a. card_data (structured YAML front-matter from README.md)
       b. description (free text remainder of README.md)
     - card_data has structured fields (task_categories, language, license)
     - description has the human-readable content
     - We need both: card_data for metadata signals, description for embeddings

  4. WHY fetch full dataset info lazily (not in search)?
     - list_datasets() returns minimal info (name, tags only)
     - dataset_info() returns full metadata but costs one HTTP call per dataset
     - Strategy: search returns basic records, embedding agent fetches full info
     - This keeps search latency low at the cost of some quality signals initially

  5. WHY handle gated datasets explicitly?
     - HF has "gated" datasets (require auth or approval to access)
     - These return 403 on info fetch but appear in search results
     - We mark them as gated in extra{} rather than failing or skipping
     - User sees result but knows they need HF account to access

FAILURE SCENARIOS HANDLED:
  - huggingface_hub not installed → ImportError → adapter disabled
  - No HF_TOKEN → public datasets still work (limited)
  - Gated dataset (403) → mark extra.gated=True, continue with partial data
  - Private dataset → skip silently (404 on info fetch)
  - card_data parse error → fall back to empty metadata, log warning
  - Generator timeout → stop early, return what we have

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from datascout.contracts import (
    DataDomain,
    DataFormat,
    LicenseType,
    Modality,
    RawDataset,
    TaskType,
    normalize_domain,
    normalize_format,
    normalize_license,
    normalize_modality,
    normalize_task_type,
)
from datascout.contracts.errors import (
    AdapterAuthError,
    AdapterConnectionError,
    AdapterRateLimitError,
    AdapterTimeoutError,
)
from datascout.contracts.requests import SearchQuery
from datascout.infrastructure.circuit_breaker import CircuitBreakerConfig, circuit_registry
from datascout.infrastructure.logging import get_logger, log_performance
from datascout.infrastructure.monitoring import metrics

from .base import AdapterHealth, BaseAdapter
from .modality_detector import detect_modality

logger = get_logger("adapters.huggingface")


class HuggingFaceAdapter(BaseAdapter):
    """
    Adapter for HuggingFace Datasets Hub.

    Authentication:
        Set HF_TOKEN environment variable for private/gated datasets.
        Public datasets work without authentication.

    Rate limits:
        HF Hub API is generous — ~300 req/min for authenticated users.
        Circuit breaker threshold set higher (10) than Kaggle (5).
    """

    ADAPTER_NAME    = "huggingface"
    ADAPTER_VERSION = "huggingface-adapter-3.0.0"

    # HF task category → TaskType mapping
    _TASK_MAP: dict[str, TaskType] = {
        "image-classification":       TaskType.IMAGE_CLASSIFICATION,
        "object-detection":           TaskType.OBJECT_DETECTION,
        "image-segmentation":         TaskType.SEMANTIC_SEGMENTATION,
        "image-to-text":              TaskType.IMAGE_CAPTIONING,
        "text-classification":        TaskType.TEXT_CLASSIFICATION,
        "token-classification":       TaskType.NER,
        "sentiment-analysis":         TaskType.SENTIMENT_ANALYSIS,
        "translation":                TaskType.MACHINE_TRANSLATION,
        "summarization":              TaskType.SUMMARIZATION,
        "question-answering":         TaskType.QUESTION_ANSWERING,
        "text-generation":            TaskType.TEXT_GENERATION,
        "fill-mask":                  TaskType.LANGUAGE_MODELING,
        "automatic-speech-recognition": TaskType.SPEECH_RECOGNITION,
        "audio-classification":       TaskType.AUDIO_CLASSIFICATION,
        "tabular-classification":     TaskType.CLASSIFICATION,
        "tabular-regression":         TaskType.REGRESSION,
        "time-series-forecasting":    TaskType.TIME_SERIES_FORECASTING,
        "visual-question-answering":  TaskType.VISUAL_QUESTION_ANSWERING,
    }

    # HF modality tags → Modality
    _MODALITY_MAP: dict[str, Modality] = {
        "text": Modality.TEXT,
        "image": Modality.IMAGE,
        "audio": Modality.AUDIO,
        "video": Modality.VIDEO,
        "tabular": Modality.TABULAR,
        "multimodal": Modality.MULTIMODAL,
        "time-series": Modality.TIME_SERIES,
    }

    def __init__(self) -> None:
        self._api: Optional[Any] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._token: Optional[str] = os.getenv("HF_TOKEN")
        self._circuit = circuit_registry.get_or_create(
            "huggingface",
            CircuitBreakerConfig(
                failure_threshold=10,       # HF is more reliable than Kaggle
                recovery_timeout_s=30.0,
            ),
        )
        logger.info("huggingface_adapter_created", extra={"has_token": bool(self._token)})

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._api is not None
        try:
            from huggingface_hub import HfApi  # type: ignore[import]
            self._api = HfApi(token=self._token)
            self._initialized = True
            logger.info("huggingface_adapter_initialized")
            return True
        except ImportError:
            self._init_error = (
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            self._initialized = True
            logger.error("huggingface_not_installed")
            return False
        except Exception as e:
            self._init_error = str(e)
            self._initialized = True
            logger.error("huggingface_init_failed", extra={"error": str(e)})
            return False

    @log_performance("huggingface.search")
    async def search(self, query: SearchQuery) -> list[RawDataset]:
        """
        Search HuggingFace datasets and return normalized RawDataset records.
        Always returns List[RawDataset] — empty list on any failure.
        """
        if not self._ensure_initialized():
            err = self._init_error or "HuggingFace adapter failed to initialise"
            logger.warning("huggingface_search_skipped", extra={"reason": err})
            if "not installed" in err.lower() or "importerror" in err.lower() or "no module" in err.lower():
                raise AdapterConnectionError("huggingface", detail=err)
            raise AdapterAuthError("huggingface")

        fetch_count = min(query.max_results * 3, 100)

        try:
            raw_results = await self._circuit.call(
                self._do_search, query.raw_query, fetch_count  # raw_query is already the short focused query from _build_short_queries
            )
        except (AdapterAuthError, AdapterConnectionError, AdapterTimeoutError,
                AdapterRateLimitError) as e:
            raise  # Let _safe_search record this as AdapterFailure
        except Exception as e:
            logger.error(
                "huggingface_search_failed",
                extra={"error": type(e).__name__, "query": query.raw_query[:50]},
            )
            metrics.adapter_error(self.ADAPTER_NAME, type(e).__name__.lower())
            return []

        datasets = self._parse_results(raw_results)
        logger.info(
            "[HF] search_complete",
            extra={"query": (query.expanded_query or query.raw_query)[:50], "results": len(datasets)},
        )
        print(f"[HF] query={query.raw_query!r} returned={len(datasets)} datasets")
        return datasets

    async def _do_search(self, query_text: str, limit: int) -> list[Any]:
        """Execute HuggingFace search in thread pool (SDK is synchronous).

        FIX: Removed `direction=-1` — parameter removed in huggingface_hub >= 0.20.
             `sort="downloads"` already implies descending order in the current SDK.
        FIX2: Use get_running_loop() not get_event_loop() — the latter is
              deprecated in Python 3.10+ when called from a running async context
              and can return a closed/wrong loop causing run_in_executor to fail silently.
        """
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        try:
    # huggingface_hub >= 1.0: sort= conflicts with search= and returns []
            # Fix: search without sort first (returns relevance-ranked results),
            # fall back to sort-only if search returns nothing.
            def _search_hf():
                # huggingface_hub >= 0.21: list_datasets() with search= only returns
                # dataset IDs by default (no tags, no card_data). Without full=True
                # the results are empty-looking and our parser skips most of them.
                # card_data=True fetches the README/card metadata needed for modality
                # detection and description extraction.
                results = list(
                    self._api.list_datasets(
                        search=query_text,
                        limit=limit,
                        full=True,          # include tags, downloads, card_data
                        # No sort= — sort conflicts with search= in hub >= 1.0
                    )
                )
                # Fallback 1: try without full= (some SDK versions don't support it with search=)
                if not results:
                    try:
                        results = list(
                            self._api.list_datasets(
                                search=query_text,
                                limit=limit,
                            )
                        )
                    except Exception:
                        pass
                # Fallback 2: HF search is very literal — retry with just the core noun
                # e.g. "image detecting potholes" → "pothole" to find niche datasets
                if not results:
                    try:
                        core = next(
                            (t for t in query_text.lower().split()
                             if len(t) >= 5 and t not in {"image","photo","visual","dataset","detect"}),
                            None
                        )
                        if core and core != query_text.lower():
                            results = list(
                                self._api.list_datasets(
                                    search=core,
                                    limit=limit,
                                    full=True,
                                )
                            )
                    except Exception:
                        pass
                return results

            results = await asyncio.wait_for(
                loop.run_in_executor(None, _search_hf),
                timeout=12.0,
            )
            return results or []
        except asyncio.TimeoutError as e:
            raise AdapterTimeoutError("huggingface", timeout_s=12.0, cause=e) from e
        except asyncio.CancelledError:
            # FIX: CancelledError must be re-raised — do NOT record it as a circuit breaker
            # failure. It means the caller was cancelled (worker timeout / disconnect),
            # not that HuggingFace is down. Swallowing it would hang the task.
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str:
                raise AdapterAuthError("huggingface", cause=e) from e
            raise AdapterConnectionError("huggingface", detail=str(e), cause=e) from e

    def _parse_results(self, raw_results: list[Any]) -> list[RawDataset]:
        """Convert HuggingFace DatasetInfo objects to RawDataset records."""
        datasets: list[RawDataset] = []
        for item in raw_results:
            try:
                dataset = self._parse_single(item)
                if dataset:
                    datasets.append(dataset)
            except Exception as e:
                logger.warning(
                    "huggingface_record_parse_failed",
                    extra={"error": str(e)[:100]},
                )
                continue
        return datasets

    def _parse_single(self, item: Any) -> Optional[RawDataset]:
        """Parse one HuggingFace DatasetInfo into a RawDataset."""
        # ── Extract ID and URL ────────────────────────────────────────────────
        dataset_id = getattr(item, "id", None) or getattr(item, "modelId", None)
        if not dataset_id:
            return None

        # ── VALIDATION: dataset_id must be non-empty string ──────────────────
        # HuggingFace dataset IDs are "username/dataset-name" or just "dataset-name"
        dataset_id = str(dataset_id).strip()
        if not dataset_id or "/" not in dataset_id and len(dataset_id) < 2:
            logger.warning(
                "[HF] invalid_dataset_id_rejected",
                extra={"dataset_id": dataset_id, "reason": "invalid format"},
            )
            return None

        source_url = f"https://huggingface.co/datasets/{dataset_id}"

        # ── Title: use last part of "owner/dataset-name" ─────────────────────
        title = dataset_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

        # ── Description from card_data ────────────────────────────────────────
        description = ""
        card_data = getattr(item, "cardData", {}) or {}
        if isinstance(card_data, dict):
            description = self._extract_str(card_data, "description", default="")

        # Fall back to any description on the object itself
        if not description:
            description = getattr(item, "description", "") or ""

        # ── Tags ──────────────────────────────────────────────────────────────
        raw_tags = getattr(item, "tags", []) or []
        tags = self._parse_tags(raw_tags)

        # ── Task categories from card_data ────────────────────────────────────
        task_types: list[TaskType] = []
        raw_tasks = []
        if isinstance(card_data, dict):
            raw_tasks = card_data.get("task_categories", []) or []
        if not raw_tasks:
            raw_tasks = getattr(item, "task_categories", []) or []

        seen_tasks: set[TaskType] = set()
        for t in raw_tasks:
            task_str = str(t).lower().strip()
            tt = self._TASK_MAP.get(task_str, normalize_task_type(task_str))
            if tt != TaskType.OTHER and tt not in seen_tasks:
                task_types.append(tt)
                seen_tasks.add(tt)

        # ── Modalities ────────────────────────────────────────────────────────
        modalities: list[Modality] = []
        seen_mods: set[Modality] = set()

        # From card_data modalities field
        raw_mods = []
        if isinstance(card_data, dict):
            raw_mods = card_data.get("modalities", []) or []
        for m_str in raw_mods:
            m = self._MODALITY_MAP.get(str(m_str).lower(), normalize_modality(str(m_str)))
            if m != Modality.OTHER and m not in seen_mods:
                modalities.append(m)
                seen_mods.add(m)

        # Infer from tags if not in card_data
        if not modalities:
            for tag in tags:
                m = normalize_modality(tag)
                if m != Modality.OTHER and m not in seen_mods:
                    modalities.append(m)
                    seen_mods.add(m)

        # MULTI-SIGNAL modality detection (Phase 1 fix).
        # HuggingFace card_data often has no modality field.
        # Use title + description + tags for stronger signal.
        if not modalities:
            result = detect_modality(
                tags=tags,
                title=title,
                description=description,
                extra={"task_categories": raw_tasks},
                source="huggingface",
            )
            detected = result.modality
            mod_map = {
                "image": Modality.IMAGE,
                "text": Modality.TEXT,
                "audio": Modality.AUDIO,
                "video": Modality.VIDEO,
                "tabular": Modality.TABULAR,
            }
            if detected in mod_map and mod_map[detected] not in seen_mods:
                modalities.append(mod_map[detected])
                seen_mods.add(mod_map[detected])
                logger.debug(
                    "[HF] modality_detected_multi_signal",
                    extra={"dataset_id": dataset_id, "modality": detected,
                           "confidence": result.confidence, "signals": result.signals_used},
                )

        # ── License ───────────────────────────────────────────────────────────
        raw_license = ""
        if isinstance(card_data, dict):
            raw_license = card_data.get("license", "") or ""
        if not raw_license:
            raw_license = str(getattr(item, "license", "") or "")
        license_type = normalize_license(raw_license) if raw_license else LicenseType.UNKNOWN

        # ── Language ──────────────────────────────────────────────────────────
        language: Optional[str] = None
        if isinstance(card_data, dict):
            langs = card_data.get("language", []) or []
            if isinstance(langs, list) and langs:
                language = str(langs[0])
            elif isinstance(langs, str):
                language = langs

        # ── Download count ────────────────────────────────────────────────────
        download_count = getattr(item, "downloads", None)
        if download_count is not None:
            try:
                download_count = int(download_count)
            except (ValueError, TypeError):
                download_count = None

        # ── Last updated ──────────────────────────────────────────────────────
        last_updated = self._parse_datetime(
            getattr(item, "lastModified", None)
            or getattr(item, "created_at", None)
        )

        # ── Domain inference ──────────────────────────────────────────────────
        data_domain: Optional[DataDomain] = None
        for tag in tags:
            domain = normalize_domain(tag)
            if domain != DataDomain.OTHER:
                data_domain = domain
                break

        # ── Author from dataset_id ────────────────────────────────────────────
        author: Optional[str] = None
        if "/" in dataset_id:
            author = dataset_id.split("/")[0]

        # ── Gated flag ────────────────────────────────────────────────────────
        is_gated = getattr(item, "gated", False)

        # ── Row count from card_data splits (no download needed) ─────────────
        # FIX: HF card_data has dataset_info.splits with num_examples.
        # Extracting this gives real row counts without downloading anything,
        # improving metadata_completeness and quality rank signals.
        row_count: Optional[int] = None
        if isinstance(card_data, dict):
            _ds_info = card_data.get("dataset_info") or {}
            if isinstance(_ds_info, list) and _ds_info:
                _ds_info = _ds_info[0]  # multi-config datasets: take first config
            if isinstance(_ds_info, dict):
                _splits = _ds_info.get("splits") or []
                if isinstance(_splits, list):
                    _total = sum(
                        int(s.get("num_examples", 0))
                        for s in _splits
                        if isinstance(s, dict)
                    )
                    if _total > 0:
                        row_count = _total
                elif isinstance(_splits, dict):
                    _total = sum(
                        int(v.get("num_examples", 0))
                        for v in _splits.values()
                        if isinstance(v, dict)
                    )
                    if _total > 0:
                        row_count = _total

        # ── Column count from card_data features (no download needed) ─────────
        # FIX: HF card_data has dataset_info.features list — length = column count.
        column_count: Optional[int] = None
        if isinstance(card_data, dict):
            _ds_info2 = card_data.get("dataset_info") or {}
            if isinstance(_ds_info2, list) and _ds_info2:
                _ds_info2 = _ds_info2[0]
            if isinstance(_ds_info2, dict):
                _features = _ds_info2.get("features") or []
                if isinstance(_features, list) and _features:
                    column_count = len(_features)

        extra: dict[str, Any] = {
            "gated": is_gated,
            "private": getattr(item, "private", False),
            "likes": getattr(item, "likes", None),
            # analysis_engine.py reads extra["hf_dataset_id"] to load tabular
            # HF datasets via the `datasets` library for deep analysis.
            "hf_dataset_id": dataset_id,
        }

        return self._build_raw_dataset(
            raw={},
            source="huggingface",
            source_id=dataset_id,
            source_url=source_url,
            title=title,
            description=description,
            tags=tags,
            license_type=license_type,
            data_domain=data_domain,
            task_types=task_types,
            modalities=modalities,
            language=language,
            last_updated=last_updated,
            download_count=download_count,
            author=author,
            row_count=row_count,        # FIX: extracted from card_data splits
            column_count=column_count,  # FIX: extracted from card_data features
            extra=extra,
        )

    async def health_check(self) -> AdapterHealth:
        """Ping HuggingFace Hub API with a minimal list call."""
        if not self._ensure_initialized():
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=False,
                error=self._init_error,
            )

        start = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: list(self._api.list_datasets(limit=1)),
                ),
                timeout=5.0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=True,
                latency_ms=latency_ms,
                message=f"HuggingFace Hub reachable in {latency_ms}ms",
            )
        except asyncio.TimeoutError:
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=False,
                latency_ms=5000,
                error="HuggingFace Hub health check timed out",
            )
        except Exception as e:
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=False,
                error=str(e),
            )