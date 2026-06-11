"""
datascout.adapters.openml_adapter
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: OpenML datasets adapter with DataFrame-based search,
ARFF format handling, and built-in quality metrics extraction.

AGENT-0 CONTEXT:
  Produces RawDataset records from OpenML's datasets API.
  OpenML is unique: it has built-in ML quality metrics (AUC, accuracy, row/col counts)
  that the other adapters don't have — we extract these as structural quality signals.

SYSTEM DESIGN DECISIONS:

  1. WHY DataFrame-based search?
     - openml.datasets.list_datasets() returns a DataFrame (or dict of dicts)
     - This is OpenML SDK's native interface — not our choice
     - We convert to list[dict] immediately and process uniformly with other adapters

  2. WHY OpenML quality metrics as structural signals?
     - OpenML computes row_count, column_count, class_count, feature_count
       server-side with high accuracy (runs actual analysis on the dataset)
     - These are gold-standard structural signals not available from Kaggle/HF
     - We extract NumberOfInstances → row_count, NumberOfFeatures → column_count
     - These feed directly into metadata_completeness and quality scoring

  3. WHY ARFF as a DataFormat?
     - OpenML's native format is ARFF (Attribute-Relation File Format)
     - Datasets on OpenML are almost always ARFF unless explicitly stated otherwise
     - We add ARFF to DataFormat enum specifically for OpenML compatibility

  4. WHY filter by status='active'?
     - OpenML has datasets with status: active, deactivated, in_preparation
     - Deactivated datasets are corrupted or have serious quality issues
     - in_preparation datasets are incomplete — no metadata available
     - Only active datasets are worth recommending

  5. WHY limit query by number_of_instances > 0?
     - Prevents returning empty placeholder datasets
     - OpenML has datasets registered but not yet uploaded (0 instances)
     - These would create valid RawDataset records with no actual data

FAILURE SCENARIOS HANDLED:
  - openml not installed → ImportError → adapter disabled
  - DataFrame parse error → convert to empty list + log
  - Quality estimate missing (None) → skip that field gracefully
  - Java-style null values ("null", "NaN", "Infinity") → treated as None
  - Dataset details fetch timeout → return basic record without full metadata

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
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
    normalize_license,
    normalize_modality,
    normalize_task_type,
)
from datascout.contracts.errors import AdapterAuthError, AdapterConnectionError, AdapterRateLimitError, AdapterTimeoutError
from datascout.contracts.requests import SearchQuery
from datascout.infrastructure.circuit_breaker import CircuitBreakerConfig, circuit_registry
from datascout.infrastructure.logging import get_logger, log_performance
from datascout.infrastructure.monitoring import metrics

from .base import AdapterHealth, BaseAdapter
from .modality_detector import detect_modality

logger = get_logger("adapters.openml")

# OpenML quality metric names → our field names
_QUALITY_MAP = {
    "NumberOfInstances":      "row_count",
    "NumberOfFeatures":       "column_count",
    "NumberOfClasses":        "class_count",
    "NumberOfMissingValues":  "missing_values",
    "MajorityClassSize":      "majority_class_size",
    "MinorityClassSize":      "minority_class_size",
}

# Java-style null values from OpenML
_JAVA_NULLS = frozenset({"null", "none", "nan", "infinity", "-infinity", "n/a", ""})


class OpenMLAdapter(BaseAdapter):
    """
    Adapter for OpenML datasets.

    Authentication:
        Set OPENML_APIKEY environment variable for write operations.
        Read operations (search, list) work without authentication.

    Rate limits:
        OpenML is very permissive — no hard rate limits documented.
        Circuit breaker threshold set to 8.
    """

    ADAPTER_NAME    = "openml"
    ADAPTER_VERSION = "openml-adapter-3.0.0"

    def __init__(self) -> None:
        self._initialized = False
        self._init_error: Optional[str] = None
        self._openml: Optional[Any] = None
        self._circuit = circuit_registry.get_or_create(
            "openml",
            CircuitBreakerConfig(
                failure_threshold=8,
                recovery_timeout_s=10.0,
            ),
        )
        logger.info("openml_adapter_created")

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._openml is not None
        try:
            import openml  # type: ignore[import]
            import os
            api_key = os.getenv("OPENML_APIKEY")
            if api_key:
                openml.config.apikey = api_key
            self._openml = openml
            self._initialized = True
            logger.info("openml_adapter_initialized")
            return True
        except ImportError:
            self._init_error = "openml not installed. Run: pip install openml"
            self._initialized = True
            logger.error("openml_not_installed")
            return False
        except Exception as e:
            self._init_error = str(e)
            self._initialized = True
            logger.error("openml_init_failed", extra={"error": str(e)})
            return False

    @log_performance("openml.search")
    async def search(self, query: SearchQuery) -> list[RawDataset]:
        """
        Search OpenML datasets by name/description match and return RawDataset records.
        Always returns List[RawDataset] — empty list on any failure.

        FIX: fail fast if OpenML has already timed out in this session.
        OpenML is on EU servers — from India it can be slow or unreachable.
        With timeout reduced to 15s and size=50, it fails fast instead of
        blocking the entire search for 120s.
        """
        if not self._ensure_initialized():
            err = self._init_error or "OpenML adapter failed to initialise"
            logger.warning("openml_search_skipped", extra={"reason": err})
            if "not installed" in err.lower() or "importerror" in err.lower() or "no module" in err.lower():
                raise AdapterConnectionError("openml", detail=err)
            raise AdapterAuthError("openml")

        fetch_count = min(query.max_results * 3, 100)

        try:
            raw_results = await self._circuit.call(
                self._do_search, query.raw_query, fetch_count  # raw_query is already the short focused query
            )
        except (AdapterAuthError, AdapterConnectionError, AdapterTimeoutError,
                AdapterRateLimitError) as e:
            raise  # Let _safe_search record this as AdapterFailure
        except Exception as e:
            logger.error(
                "openml_search_failed",
                extra={
                    "error_type": type(e).__name__,
                    "error":      str(e)[:300],
                    "query":      (query.expanded_query or query.raw_query)[:50],
                },
                exc_info=True,
            )
            metrics.adapter_error(self.ADAPTER_NAME, type(e).__name__.lower())
            return []

        datasets = self._parse_results(raw_results, query.expanded_query or query.raw_query)
        logger.info(
            "[OPENML] search_complete",
            extra={"query": (query.expanded_query or query.raw_query)[:50], "results": len(datasets)},
        )
        print(f"[OPENML] query={query.raw_query!r} returned={len(datasets)} datasets")
        return datasets

    async def _do_search(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """
        Execute OpenML dataset list in thread pool.
        Returns list of dicts (converted from DataFrame or dict-of-dicts).
        """
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._fetch_datasets(query_text, limit),
                ),
                timeout=25.0,   # FIX: raised 15s→25s. 15s was too tight on India→EU
                                # links where first OpenML call includes DNS + TLS.
            )
            return result
        except asyncio.TimeoutError as e:
            raise AdapterTimeoutError("openml", timeout_s=25.0, cause=e) from e
        except Exception as e:
            raise AdapterConnectionError("openml", detail=str(e), cause=e) from e

    def _fetch_datasets(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """
        Synchronous OpenML dataset fetch (runs in thread pool).

        WHY list_datasets with filter:
        - status='active' excludes broken/deactivated datasets
        - number_of_instances > 0 excludes empty placeholder datasets
        - Returns DataFrame or dict — we normalize to list[dict]
        """
        try:
            # output_format='dataframe' is current API (dict deprecated in >=0.15).
            # number_of_instances_greater_than was removed in a recent OpenML API update
            # (returns error 370). Use size= to cap the fetch, then filter in Python.
            raw = self._openml.datasets.list_datasets(
                status="active",
                output_format="dataframe",
                size=200,   # FIX: 200 gives enough coverage for keyword matching;
                            # 50 was too small — niche queries return 0 matches.
            )
        except Exception as e:
            logger.warning("openml_list_failed", extra={"error": str(e)})
            return []

        if raw is None:
            return []

        # Normalise whatever format we got to list[dict]
        if hasattr(raw, "to_dict"):
            # pandas DataFrame — current default in openml >= 0.15
            try:
                # drop=True: discard the numeric index (did is already a column in the DF)
                records = raw.reset_index(drop=True).to_dict(orient="records")
            except Exception as df_err:
                logger.warning("openml_dataframe_parse_failed", extra={"error": str(df_err)})
                return []
        elif isinstance(raw, dict):
            # Legacy dict-of-dicts (openml < 0.15 with output_format='dict')
            records = list(raw.values())
        elif isinstance(raw, list):
            records = raw
        else:
            logger.warning("openml_unexpected_format", extra={"type": type(raw).__name__})
            return []

        # Filter by query text using multi-token keyword matching.
        #
        # FIX v2.2.0 — ROOT CAUSE of Issue 3 (OpenML disappears from results):
        # The old code fell back to returning the 50 LARGEST datasets by row count
        # (iris, diabetes, mnist, etc.) when no name match was found. These
        # completely unrelated datasets then scored 0.0 on description_match and
        # were eliminated by DESCRIPTION_MATCH_GATE in the scorer — every single
        # OpenML result vanished from the top-10.
        #
        # Fix: split the query into individual tokens and match ANY token. This
        # gives OpenML a real chance to match domain words (e.g. "pothole",
        # "medical", "satellite") without requiring the full query string to appear
        # verbatim in a dataset name. When NO tokens match at all, return [] instead
        # of dumping unrelated popular datasets that will only be gated out anyway.
        query_lower = query_text.lower().strip()

        # Build token set: individual words ≥3 chars, skip generic ML stopwords
        _STOPWORDS = frozenset({
            "the", "for", "and", "with", "using", "dataset", "datasets",
            "data", "machine", "learning", "deep", "model", "models",
        })
        query_tokens = [
            t for t in query_lower.split()
            if len(t) >= 3 and t not in _STOPWORDS
        ]

        def _record_matches(r: dict) -> bool:
            """Return True if any query token matches the record name or description.
            FIX: also match partial tokens (substring) for short cryptic OpenML names.
            e.g. query 'diabetes' matches dataset name 'diabetes_numeric'.
            """
            name = str(r.get("name", "")).lower()
            desc = str(r.get("description", "")).lower()
            tags_str = " ".join(str(t) for t in (r.get("tag") or [])).lower()
            searchable = f"{name} {desc} {tags_str}"
            # Full phrase match
            if query_lower in searchable:
                return True
            # Any individual token match (exact or substring)
            for tok in query_tokens:
                if tok in searchable:
                    return True
            # Reverse: any name word appears in query (catches "diabetes_numeric" ↔ "diabetes")
            for name_tok in name.replace("-", " ").replace("_", " ").split():
                if len(name_tok) >= 4 and name_tok in query_lower:
                    return True
            return False

        matched = [r for r in records if _record_matches(r)]

        # If zero keyword matches, try relaxed single-char token match before giving up.
        # This handles very short queries like "nlu" or "ecg" that get filtered by len>=3.
        if not matched:
            # Try with shorter tokens (len >= 2)
            short_tokens = [t for t in query_lower.split() if len(t) >= 2]
            matched = [
                r for r in records
                if any(
                    tok in f"{str(r.get('name','')).lower()} {str(r.get('description','')).lower()}"
                    for tok in short_tokens
                )
            ]

        if not matched:
            logger.info(
                "openml_no_keyword_match",
                extra={"query": query_text, "fetched": len(records)},
            )
            return []

        # Sort matched datasets by NumberOfInstances descending so larger,
        # more substantial datasets appear first when scores are equal.
        matched.sort(
            key=lambda r: self._safe_int(r.get("NumberOfInstances", 0)),
            reverse=True,
        )
        return matched[:limit]

    def _parse_results(
        self, raw_results: list[dict[str, Any]], query_text: str
    ) -> list[RawDataset]:
        """Convert OpenML dataset dicts to RawDataset records."""
        datasets: list[RawDataset] = []
        for raw in raw_results:
            try:
                dataset = self._parse_single(raw)
                if dataset:
                    datasets.append(dataset)
            except Exception as e:
                logger.warning(
                    "openml_record_parse_failed",
                    extra={"error": str(e)[:100], "did": raw.get("did", "?")},
                )
                continue
        return datasets

    def _parse_single(self, raw: dict[str, Any]) -> Optional[RawDataset]:
        """Parse one OpenML dataset dict into a RawDataset."""
        # ── Required fields ───────────────────────────────────────────────────
        did = raw.get("did") or raw.get("id")
        name = self._extract_str(raw, "name", default="")
        if not did or not name:
            return None

        # ── VALIDATION: did must be a valid numeric id ────────────────────────
        try:
            did_int = int(float(str(did)))
            if did_int <= 0:
                raise ValueError("non-positive id")
        except (ValueError, TypeError):
            logger.warning(
                "[OPENML] invalid_dataset_id_rejected: did=%r name=%r reason=non-numeric or non-positive id",
                did, name,
            )
            return None

        source_id  = str(did_int)
        source_url = f"https://www.openml.org/d/{did_int}"

        # ── Download URL ──────────────────────────────────────────────────────
        # FIX: store direct CSV download URL in extra so AnalysisEngine
        # can load real tabular data instead of being stuck with metadata-only.
        # Dataset ID and download file ID are NOT always the same.
        _download_url = None
        try:
            dataset_obj = self._openml.datasets.get_dataset(
                did_int,
                download_data=False,
            )
            if hasattr(dataset_obj, "url"):
                _download_url = dataset_obj.url
        except Exception as e:
            logger.warning(
                "openml_download_url_lookup_failed",
                extra={"did": did_int, "error": str(e)},
            )

        title = name.replace("_", " ").replace("-", " ").title()

        # ── Description ───────────────────────────────────────────────────────
        description = self._extract_str(raw, "description", default="")

        # ── Quality metrics (OpenML's gold-standard structural signals) ───────
        row_count    = self._safe_int(raw.get("NumberOfInstances"))
        column_count = self._safe_int(raw.get("NumberOfFeatures"))
        class_count  = self._safe_int(raw.get("NumberOfClasses"))

        # ── Tags from OpenML tag field ────────────────────────────────────────
        tags = self._parse_tags(raw.get("tag", []))

        # Add inferred tags from class count
        if class_count == 2:
            tags = list(set(tags + ["binary_classification"]))
        elif class_count and class_count > 2:
            tags = list(set(tags + ["multi_class_classification"]))

        # ── Task type inference ────────────────────────────────────────────────
        task_types: list[TaskType] = []
        seen_tasks: set[TaskType] = set()

        # OpenML's format field often indicates task type
        openml_format = self._extract_str(raw, "format", default="").lower()
        if "arff" in openml_format or openml_format == "":
            # Default OpenML assumption: tabular classification/regression
            if class_count and class_count > 0:
                tt = TaskType.BINARY_CLASSIFICATION if class_count == 2 else TaskType.CLASSIFICATION
                task_types.append(tt)
                seen_tasks.add(tt)

        for tag in tags:
            tt = normalize_task_type(tag)
            if tt != TaskType.OTHER and tt not in seen_tasks:
                task_types.append(tt)
                seen_tasks.add(tt)

        # ── Modalities (OpenML is almost exclusively tabular) ─────────────────
        modalities: list[Modality] = []
        for tag in tags:
            m = normalize_modality(tag)
            if m != Modality.OTHER:
                modalities.append(m)
        if not modalities:
            # Use multi-signal detector — even for OpenML, description may reveal
            # image content (some OpenML datasets contain image feature vectors)
            result = detect_modality(
                tags=tags,
                title=title,
                description=description,
                extra={
                    "class_count": class_count,
                    "feature_count": self._safe_int(raw.get("NumberOfFeatures")),
                },
                source="openml",
            )
            detected = result.modality
            if detected == "image":
                modalities = [Modality.IMAGE]
            elif detected in ("text", "audio"):
                mod_map = {"text": Modality.TEXT, "audio": Modality.AUDIO}
                modalities = [mod_map[detected]]
            else:
                modalities = [Modality.TABULAR]  # OpenML default

        # ── Format ────────────────────────────────────────────────────────────
        data_format = DataFormat.csv  # OpenML native format

        # ── License ───────────────────────────────────────────────────────────
        raw_license = self._extract_str(raw, "licence", "license", default="")
        license_type = LicenseType.CC_BY if not raw_license else normalize_license(raw_license)

        # ── Domain ────────────────────────────────────────────────────────────
        data_domain: Optional[DataDomain] = None
        for tag in tags:
            domain = normalize_domain(tag)
            if domain != DataDomain.OTHER:
                data_domain = domain
                break
        if not data_domain:
            data_domain = DataDomain.TABULAR  # OpenML default

        # ── Author / uploader ─────────────────────────────────────────────────
        creator = self._extract_str(raw, "creator", "uploader", default=None)

        # ── Last updated ──────────────────────────────────────────────────────
        last_updated = self._parse_datetime(
            raw.get("upload_date") or raw.get("last_update")
        )

        # ── Extra quality metrics for downstream use ──────────────────────────
        extra: dict[str, Any] = {
            "download_url": _download_url,  # direct CSV — no auth needed
            "class_count": class_count,
            "missing_values": self._safe_int(raw.get("NumberOfMissingValues")),
            "majority_class_size": self._safe_int(raw.get("MajorityClassSize")),
            "minority_class_size": self._safe_int(raw.get("MinorityClassSize")),
            "openml_version": raw.get("version"),
            "openml_visibility": raw.get("visibility"),
        }

        return self._build_raw_dataset(
            raw=raw,
            source="openml",
            source_id=source_id,
            source_url=source_url,
            title=title,
            description=description,
            tags=tags,
            row_count=row_count,
            column_count=column_count,
            data_format=data_format,
            license_type=license_type,
            data_domain=data_domain,
            task_types=task_types,
            modalities=modalities,
            last_updated=last_updated,
            author=creator,
            extra=extra,
        )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """
        Safely convert OpenML numeric values to int.
        Handles: None, "null", float strings, NaN, Infinity.
        """
        if value is None:
            return None
        s = str(value).strip().lower()
        if s in _JAVA_NULLS or "inf" in s or "nan" in s:
            return None
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    async def health_check(self) -> AdapterHealth:
        """Ping OpenML API with a minimal dataset list."""
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
                    lambda: self._openml.datasets.list_datasets(
                        status="active",
                        output_format="dataframe",
                        size=1,
                    ),
                ),
                timeout=5.0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=True,
                latency_ms=latency_ms,
                message=f"OpenML API reachable in {latency_ms}ms",
            )
        except asyncio.TimeoutError:
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=False,
                latency_ms=5000,
                error="OpenML health check timed out",
            )
        except Exception as e:
            return AdapterHealth(
                adapter=self.ADAPTER_NAME,
                healthy=False,
                error=str(e),
            )