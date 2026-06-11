"""
datascout.adapters.kaggle_adapter
─────────────────────────────────────────────────────
Production Kaggle Datasets adapter.

Supports kaggle SDK 1.x (KaggleApiExtended) and 2.x (KaggleApi) transparently.

Authentication:
    Option A — file:   ~/.kaggle/kaggle.json  {"username":"...","key":"..."}
    Option B — env:    KAGGLE_USERNAME + KAGGLE_KEY

Author:  Principal Engineer
Version: 3.1.0  (kaggle 1.x / 2.x dual-compat)
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
    AdapterServerError,
    AdapterTimeoutError,
)
from datascout.contracts.requests import SearchQuery
from datascout.infrastructure.circuit_breaker import CircuitBreakerConfig, circuit_registry
from datascout.infrastructure.config.settings import get_settings
from datascout.infrastructure.logging import get_logger, log_performance
from datascout.infrastructure.monitoring import metrics

from .base import AdapterHealth, BaseAdapter
from .modality_detector import detect_modality

logger = get_logger("adapters.kaggle")


class KaggleAdapter(BaseAdapter):
    """
    Adapter for Kaggle Datasets API.

    Dual-compatible with kaggle SDK 1.x and 2.x:
      - 1.x: class KaggleApiExtended, method dataset_list(search=, page_size=, sort_by=)
      - 2.x: class KaggleApi,         method dataset_list(search=, page=,      sort_by=)

    The _ensure_initialized() method detects which version is installed and
    stores the correct callable in self._dataset_list_fn so search logic
    doesn't need version branches.
    """

    ADAPTER_NAME    = "kaggle"
    ADAPTER_VERSION = "kaggle-adapter-3.1.0"

    _LICENSE_MAP: dict[str, LicenseType] = {
        "CC0-1.0":              LicenseType.CC0,
        "CC BY 4.0":            LicenseType.CC_BY,
        "CC BY-SA 4.0":         LicenseType.CC_BY_SA,
        "CC BY-NC 4.0":         LicenseType.CC_BY_NC,
        "CC BY-NC-SA 4.0":      LicenseType.CC_BY_NC_SA,
        "CC BY-ND 4.0":         LicenseType.CC_BY_ND,
        "Attribution 4.0 International (CC BY 4.0)": LicenseType.CC_BY,
        "unknown":              LicenseType.UNKNOWN,
        "other":                LicenseType.UNKNOWN,
        "GPL 2":                LicenseType.GPL_2,
        "GPL 3":                LicenseType.GPL_3,
    }

    def __init__(self) -> None:
        self._api: Optional[Any]               = None
        self._sdk_version: Optional[int]        = None   # 1 or 2
        self._initialized                       = False
        self._init_error: Optional[str]         = None
        self._circuit = circuit_registry.get_or_create(
            "kaggle",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60.0),
        )
        logger.info("kaggle_adapter_created")

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALISATION  (sync, lazy)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> bool:
        """
        Lazy-init the Kaggle API client.
        Detects SDK version (1.x vs 2.x) and authenticates.
        Returns True if ready, False on any failure.
        """
        if self._initialized:
            return self._api is not None

        try:
            api, version = self._import_and_authenticate()
            self._api         = api
            self._sdk_version = version
            self._initialized = True
            logger.info("kaggle_adapter_authenticated", extra={"sdk_version": version})
            return True

        except ImportError as e:
            self._init_error  = f"kaggle package not installed or broken: {e}"
            self._initialized = True
            # Log as ERROR not CRITICAL — ImportError is a setup issue, not a crash
            logger.error("kaggle_not_installed", extra={"error": self._init_error})
            return False

        except Exception as e:
            self._init_error  = str(e)
            self._initialized = True
            logger.critical("kaggle_auth_failed", extra={"error": str(e)}, exc_info=True)
            return False

    def _import_and_authenticate(self) -> tuple[Any, int]:
        """
        Import kaggle SDK and authenticate.  Returns (api_instance, sdk_major_version).

        kaggle 1.x:  from kaggle.api.kaggle_api_extended import KaggleApiExtended
                     api = KaggleApiExtended(); api.authenticate()

        kaggle 2.x:  from kaggle.api.kaggle_api_extended import KaggleApi
                     api = KaggleApi(); api.authenticate()

        WHY not just `import kaggle; kaggle.api.authenticate()`?
        - kaggle 1.x __init__.py calls authenticate() at import time.
          If credentials are wrong, the ImportError-like error fires before
          we even reach authenticate() — and we can't catch it cleanly.
        - Direct class import + explicit authenticate() gives us clean error handling.

        IMPORTANT: pydantic-settings reads .env into the Settings model but does NOT
        set os.environ.  The kaggle SDK calls os.environ.get("KAGGLE_USERNAME") directly,
        so we must inject the credentials before authenticate() is called.
        """
        import os
        # ── Inject credentials from settings into os.environ ─────────────────
        # Only set if not already present (respects user's system kaggle.json)
        settings = get_settings()
        kaggle_user = getattr(settings, "kaggle_username", None) or os.environ.get("KAGGLE_USERNAME", "")
        kaggle_key  = getattr(settings, "kaggle_key", None) or os.environ.get("KAGGLE_KEY", "")
        if kaggle_user:
            os.environ["KAGGLE_USERNAME"] = kaggle_user
        if kaggle_key:
            os.environ["KAGGLE_KEY"] = kaggle_key

        # ── Try 1.x path first (KaggleApiExtended) ───────────────────────────
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore[import]
            api = KaggleApiExtended()
            api.authenticate()
            return api, 1
        except ImportError:
            pass   # Class doesn't exist in this version — try 2.x
        except Exception:
            raise  # Auth or other real error

        # ── Try 2.x path (KaggleApi) ─────────────────────────────────────────
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import]
        api = KaggleApi()
        api.authenticate()
        return api, 2

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    @log_performance("kaggle.search")
    async def search(self, query: SearchQuery) -> list[RawDataset]:
        if not self._ensure_initialized():
            err = self._init_error or "Kaggle adapter failed to initialise"
            logger.warning("kaggle_search_skipped", extra={"reason": err})
            # Raise typed errors so ScoutAgent can surface them in the UI.
            # "not installed" → ImportError-like message → AdapterConnectionError
            # "auth"         → credentials missing     → AdapterAuthError
            if "not installed" in err.lower() or "importerror" in err.lower() or "no module" in err.lower():
                raise AdapterConnectionError("kaggle", detail=err)
            raise AdapterAuthError("kaggle")

        fetch_count = min(query.max_results * 3, 150)

        try:
            raw_results = await self._circuit.call(
                self._do_search,
                query.raw_query,  # raw_query = short focused query from _build_short_queries
                fetch_count,
            )
        except (AdapterAuthError, AdapterConnectionError, AdapterTimeoutError,
                AdapterRateLimitError, AdapterServerError) as e:
            # Re-raise typed adapter errors so _safe_search in ScoutAgent
            # can record them as AdapterFailure and surface them in the UI.
            raise
        except Exception as e:
            logger.error(
                "kaggle_search_failed",
                extra={"error_type": type(e).__name__, "error": str(e)[:200],
                       "query": query.raw_query[:50]},
                exc_info=True,
            )
            metrics.adapter_error(self.ADAPTER_NAME, type(e).__name__.lower())
            return []

        datasets = self._parse_results(raw_results)
        logger.info(
            "kaggle_search_complete",
            extra={"query": (query.expanded_query or query.raw_query)[:50], "results": len(datasets),},
        )
        return datasets

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: API CALL
    # ─────────────────────────────────────────────────────────────────────────

    async def _do_search(self, query_text: str, page_size: int) -> list[Any]:
        """Call Kaggle dataset_list in a thread pool (SDK is synchronous)."""
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._call_dataset_list(query_text, page_size),
                ),
                timeout=15.0,
            )
            return result or []

        except asyncio.TimeoutError as e:
            raise AdapterTimeoutError("kaggle", timeout_s=15.0, cause=e) from e
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["401", "403", "forbidden", "unauthorized"]):
                raise AdapterAuthError("kaggle", cause=e) from e
            if any(x in err for x in ["429", "rate limit"]):
                raise AdapterRateLimitError("kaggle", retry_after=60.0, cause=e) from e
            if any(x in err for x in ["500", "502", "503", "504"]):
                raise AdapterServerError("kaggle", status_code=503, cause=e) from e
            raise AdapterConnectionError("kaggle", detail=str(e), cause=e) from e

    def _call_dataset_list(self, query_text: str, page_size: int) -> list[Any]:
        """
        Call the correct dataset_list() signature for the detected SDK version.

        kaggle 1.x: dataset_list(search=, page_size=, sort_by=)
        kaggle 2.x: dataset_list(search=, page=,      sort_by=)
                    Returns list[ApiDataset] — no page_size param.
                    We fetch multiple pages to approximate page_size.
        """
        if self._sdk_version == 1:
            return self._api.dataset_list(
                search=query_text,
                page_size=page_size,
                sort_by="votes",       # votes = community quality signal, not trending noise
            ) or []

        # kaggle 2.x — no page_size, fetch multiple pages
        results: list[Any] = []
        page = 1
        per_page = 20   # Kaggle 2.x default page size
        pages_needed = max(1, (page_size + per_page - 1) // per_page)

        for _ in range(pages_needed):
            batch = self._api.dataset_list(
                search=query_text,
                page=page,
                sort_by="votes",       # votes = community quality signal, not trending noise
            ) or []
            results.extend(batch)
            if len(batch) < per_page:
                break   # No more pages
            page += 1

        # ── Client-side keyword re-rank ──────────────────────────────────────
        # Kaggle API sorts by votes/trending — not by query relevance.
        # Re-rank by keyword match score BEFORE returning to scorer.
        # Uses short meaningful tokens only (skip stopwords + short words).
        _STOP = {"i", "m", "a", "an", "the", "for", "in", "on", "with",
                 "and", "or", "is", "are", "to", "that", "this", "of",
                 "my", "we", "our", "im", "building", "want", "need",
                 "using", "use", "can", "how", "what", "get", "make"}
        if query_text and results:
            qtokens = set(
                t for t in query_text.lower().replace("-", " ").split()
                if len(t) >= 3 and t not in _STOP
            )
            def _kw_score(item: Any) -> int:
                title = ""
                tags  = ""
                desc  = ""
                try:
                    title = (getattr(item, "title", "") or "").lower()
                    tags  = " ".join(
                        getattr(item, "tags", []) or []
                    ).lower()
                    desc  = (
                        getattr(item, "subtitle", "")
                        or getattr(item, "description", "")
                        or ""
                    ).lower()
                except Exception:
                    pass
                text = f"{title} {tags} {desc}"
                return sum(1 for t in qtokens if len(t) > 2 and t in text)
            results = sorted(results, key=_kw_score, reverse=True)

        return results[:page_size]

    # ─────────────────────────────────────────────────────────────────────────
    # PARSING
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_results(self, raw_results: list[Any]) -> list[RawDataset]:
        datasets: list[RawDataset] = []
        for item in raw_results:
            try:
                raw = self._sdk_object_to_dict(item)
                dataset = self._parse_single(raw)
                if dataset:
                    datasets.append(dataset)
            except Exception as e:
                logger.warning(
                    "kaggle_record_parse_failed",
                    extra={"error": str(e)[:100], "item": str(item)[:50]},
                )
        return datasets

    def _sdk_object_to_dict(self, obj: Any) -> dict[str, Any]:
        """
        Convert a Kaggle SDK object to a plain dict.

        Handles both 1.x objects (attribute-based) and 2.x ApiDataset objects
        (also attribute-based but with different field names).

        Field mapping (1.x name → 2.x name → our canonical name):
          ref           → ref              → ref
          title         → title            → title
          subtitle      → subtitle         → subtitle
          description   → description      → description
          totalBytes    → (not present)    → file_size
          lastUpdated   → last_updated     → last_updated
          downloadCount → download_count   → download_count
          voteCount     → (not present)    → vote_count
          ownerName     → owner_name       → owner_name
          ownerRef      → owner_ref        → owner_ref
          usabilityRating→(not present)    → usability_rating
          tags          → tags             → tags
          licenseName   → license_name     → license_name
        """
        if isinstance(obj, dict):
            return obj

        result: dict[str, Any] = {}

        # Try both 1.x and 2.x attribute names for each field
        field_candidates = [
            # (our_key, [1.x_attr, 2.x_attr, ...])
            ("ref",              ["ref"]),
            ("title",            ["title"]),
            ("subtitle",         ["subtitle"]),
            ("description",      ["description"]),
            ("lastUpdated",      ["lastUpdated", "last_updated"]),
            ("downloadCount",    ["downloadCount", "download_count"]),
            ("totalBytes",       ["totalBytes", "total_bytes"]),
            ("voteCount",        ["voteCount", "vote_count"]),
            ("ownerName",        ["ownerName", "owner_name", "creator_name"]),
            ("ownerRef",         ["ownerRef",  "owner_ref",  "creator_url"]),
            ("usabilityRating",  ["usabilityRating", "usability_rating"]),
            ("tags",             ["tags"]),
            ("licenseName",      ["licenseName", "license_name"]),
            ("kernelCount",      ["kernelCount", "kernel_count"]),
            ("viewCount",        ["viewCount", "view_count"]),
        ]

        for our_key, attrs in field_candidates:
            for attr in attrs:
                value = getattr(obj, attr, None)
                if value is not None:
                    result[our_key] = value
                    break

        # 2.x ApiDataset: tags is a list of ApiDatasetTag objects with .name
        raw_tags = result.get("tags", [])
        if raw_tags and not isinstance(raw_tags[0], str):
            try:
                result["tags"] = [
                    t.name if hasattr(t, "name") else str(t)
                    for t in raw_tags
                ]
            except Exception:
                result["tags"] = []

        return result

    def _parse_single(self, raw: dict[str, Any]) -> Optional[RawDataset]:
        ref   = self._extract_str(raw, "ref", "id", default="").strip("/")
        title = self._extract_str(raw, "title", "subtitle", default="")
        if not ref:
            return None

        # ── VALIDATION: ref must be "owner/slug" ─────────────────────────────
        # Prevents wrong owner/slug pairs from reaching the download pipeline.
        parts = ref.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.warning(
                "[KAGGLE] invalid_ref_rejected",
                extra={"ref": ref, "reason": "must be owner/slug"},
            )
            return None

        owner = parts[0]
        slug  = parts[1]
        source_url  = f"https://www.kaggle.com/datasets/{owner}/{slug}"
        description = self._extract_str(raw, "description", "subtitle", default="")
        author      = self._extract_str(raw, "ownerName", "ownerRef", default=None)

        download_count = self._extract_int(raw, "downloadCount", default=None)
        upvote_count   = self._extract_int(raw, "voteCount",     default=None)
        file_size      = self._extract_int(raw, "totalBytes",    default=None)
        last_updated   = self._parse_datetime(raw.get("lastUpdated"))

        raw_license  = self._extract_str(raw, "licenseName", default="unknown")
        license_type = self._LICENSE_MAP.get(raw_license, normalize_license(raw_license))

        tags = self._parse_tags(raw.get("tags", []))

        data_domain = None
        for tag in tags:
            d = normalize_domain(tag)
            if d != DataDomain.OTHER:
                data_domain = d
                break

        task_types: list[TaskType] = []
        seen_tt: set[TaskType] = set()
        for tag in tags:
            tt = normalize_task_type(tag)
            if tt != TaskType.OTHER and tt not in seen_tt:
                task_types.append(tt)
                seen_tt.add(tt)

        modalities: list[Modality] = []
        seen_m: set[Modality] = set()
        for tag in tags:
            m = normalize_modality(tag)
            if m != Modality.OTHER and m not in seen_m:
                modalities.append(m)
                seen_m.add(m)

        # MULTI-SIGNAL modality detection (Phase 1 fix).
        # The old tag-only approach returned OTHER for PlantVillage, rice leaf
        # disease, and most image datasets tagged with "classification".
        # Now we run all four signals: tags, title, description, metadata.
        if not modalities or (len(modalities) == 1 and Modality.TABULAR in seen_m):
            result = detect_modality(
                tags=tags,
                title=title,
                description=description,
                extra={"ref": ref},
                source="kaggle",
            )
            detected = result.modality
            if detected == "image" and Modality.IMAGE not in seen_m:
                modalities.append(Modality.IMAGE)
                seen_m.add(Modality.IMAGE)
                logger.debug(
                    "[KAGGLE] modality_upgraded",
                    extra={"ref": ref, "modality": "image", "confidence": result.confidence,
                           "signals": result.signals_used},
                )
            elif detected == "text" and Modality.TEXT not in seen_m:
                modalities.append(Modality.TEXT)
                seen_m.add(Modality.TEXT)
            elif detected == "audio" and Modality.AUDIO not in seen_m:
                modalities.append(Modality.AUDIO)
                seen_m.add(Modality.AUDIO)
            elif detected == "tabular" and Modality.TABULAR not in seen_m and not modalities:
                modalities.append(Modality.TABULAR)
                seen_m.add(Modality.TABULAR)

        # If still no modality from multi-signal, apply the old heuristic fallback
        if Modality.IMAGE not in seen_m:
            _combined = " ".join([
                title or "",
                description or "",
                " ".join(tags),
            ]).lower()
            _IMAGE_SIGNALS = {
                "image", "images", "photo", "photos", "photograph", "visual",
                "picture", "pictures", "computer-vision", "computer vision",
                "cv", "vision", "object detection", "segmentation", "cnn",
                "convolutional", "jpg", "jpeg", "png", "tiff", "bmp",
                "leaf", "leaves", "plant", "flower", "satellite", "aerial",
                "medical image", "chest xray", "mri", "retinal", "fundus",
                "microscopy", "histology", "pathology slide", "dermoscopy",
                "face", "facial", "emotion recognition", "gesture",
            }
            if any(sig in _combined for sig in _IMAGE_SIGNALS):
                modalities.append(Modality.IMAGE)
                seen_m.add(Modality.IMAGE)

        if Modality.TABULAR not in seen_m and not modalities:
            # No modality detected at all — check if it looks tabular
            _combined_tb = " ".join([title or "", description or "", " ".join(tags)]).lower()
            _TABULAR_SIGNALS = {
                "csv", "tabular", "spreadsheet", "excel", "survey", "census",
                "sales", "financial", "stock", "price", "revenue", "transaction",
            }
            if any(sig in _combined_tb for sig in _TABULAR_SIGNALS):
                modalities.append(Modality.TABULAR)
                seen_m.add(Modality.TABULAR)

        data_format: Optional[DataFormat] = None
        for tag in tags:
            fmt = normalize_format(tag)
            if fmt != DataFormat.OTHER:
                data_format = fmt
                break

        # FIX: Build extra with Kaggle API download URL for tabular datasets.
        # analysis_engine tabular pipeline checks extra["download_url"] when
        # source_url is a landing page (which it always is for Kaggle).
        # Without this, ALL Kaggle tabular datasets get metadata-only analysis.
        _is_tabular = Modality.TABULAR in modalities or (
            not modalities or (len(modalities) == 1 and Modality.IMAGE not in modalities)
        )
        _kaggle_download_url: Optional[str] = None
        if _is_tabular and Modality.IMAGE not in modalities:
            # Kaggle dataset ZIP download via API — requires kaggle auth (already set)
            # Only set for datasets that are likely CSV/tabular to avoid downloading huge image zips
            _kaggle_download_url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"

        _extra: dict = {
            "usability_rating": raw.get("usabilityRating"),
            "kernel_count":     raw.get("kernelCount"),
            "view_count":       raw.get("viewCount"),
        }
        if _kaggle_download_url:
            _extra["download_url"] = _kaggle_download_url

        return self._build_raw_dataset(
            raw=raw,
            source="kaggle",
            source_id=ref,
            source_url=source_url,
            title=title,
            description=description,
            tags=tags,
            file_size_bytes=file_size,
            data_format=data_format,
            license_type=license_type,
            data_domain=data_domain,
            task_types=task_types,
            modalities=modalities,
            last_updated=last_updated,
            download_count=download_count,
            upvote_count=upvote_count,
            author=author,
            extra=_extra,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> AdapterHealth:
        if not self._ensure_initialized():
            return AdapterHealth(adapter=self.ADAPTER_NAME, healthy=False, error=self._init_error)

        start = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._call_dataset_list("test", 1),
                ),
                timeout=5.0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=True,
                latency_ms=latency_ms,
                message=f"Kaggle API reachable in {latency_ms}ms",
            )
        except asyncio.TimeoutError:
            return AdapterHealth(
                adapter=self.ADAPTER_NAME, healthy=False,
                latency_ms=5000, error="health check timed out after 5s",
            )
        except Exception as e:
            return AdapterHealth(adapter=self.ADAPTER_NAME, healthy=False, error=str(e))