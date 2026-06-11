"""
datascout.search.elasticsearch_engine
──────────────────────────────────────
Elasticsearch hybrid search engine — replaces in-memory BM25 + FAISS.

DESIGN:
  - One persistent index per deployment (settings.elasticsearch_index)
  - Each RawDataset is indexed as one ES document
  - Hybrid search = BM25 (match query on title+description+tags)
                  + kNN (dense_vector on embedding field)
                  combined via RRF (Reciprocal Rank Fusion, ES 8.8+)
  - Graceful fallback: if ES unavailable → returns [] and logs warning
  - Never raises to caller

INDEX MAPPING:
  title:          text, analyzed
  description:    text, analyzed
  tags:           keyword (array)
  source:         keyword
  canonical_id:   keyword (unique document ID)
  task_types:     keyword (array)
  modalities:     keyword (array)
  download_count: long
  row_count:      long
  metadata_completeness: float
  last_updated:   date
  embedding:      dense_vector, dims=384 (all-MiniLM-L6-v2)

DOCUMENT ID: Use canonical_id as the ES _id — this gives natural dedup/upsert.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ElasticsearchEngine"]

# ── Index mapping ─────────────────────────────────────────────────────────────

INDEX_MAPPING: Dict[str, Any] = {
    "mappings": {
        "properties": {
            "canonical_id":          {"type": "keyword"},
            "title":                 {"type": "text", "analyzer": "english"},
            "description":           {"type": "text", "analyzer": "english"},
            "tags":                  {"type": "keyword"},
            "source":                {"type": "keyword"},
            "task_types":            {"type": "keyword"},
            "modalities":            {"type": "keyword"},
            "download_count":        {"type": "long"},
            "row_count":             {"type": "long"},
            "metadata_completeness": {"type": "float"},
            "last_updated":          {"type": "date", "ignore_malformed": True},
            "author":                {"type": "keyword"},
            "license_type":          {"type": "keyword"},
            "source_url":            {"type": "keyword", "index": False},
            "source_id":             {"type": "keyword"},
            "ingestion_timestamp":   {"type": "date"},
            "pipeline_version":      {"type": "keyword"},
            "ingestion_version":     {"type": "keyword"},
            "dataset_fingerprint":   {"type": "keyword"},
            "row_count":             {"type": "long"},
            "column_count":          {"type": "integer"},
            "file_size_bytes":       {"type": "long"},
            "upvote_count":          {"type": "long"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


class ElasticsearchEngine:
    """Async Elasticsearch hybrid search engine.

    Wraps the ``elasticsearch-py[async]`` client and provides a clean
    interface for indexing and querying ``RawDataset`` objects.  All public
    methods degrade gracefully — they return safe defaults and log warnings
    instead of raising on connection or query errors.

    Parameters
    ----------
    url:
        Elasticsearch base URL, e.g. ``"http://localhost:9200"``.
    index:
        Name of the Elasticsearch index to use.
    api_key:
        Optional API key for Elastic Cloud authentication.  Pass as a plain
        string (``"id:api_key_value"``).

    Example
    -------
    >>> engine = ElasticsearchEngine(url="http://localhost:9200", index="datascout-datasets")
    >>> connected = await engine.connect()
    >>> results = await engine.hybrid_search("image classification", query_embedding)
    """

    def __init__(
        self,
        url: str,
        index: str,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialise the engine without connecting.

        The connection is established lazily via :meth:`connect`.
        """
        self._url: str = url.rstrip("/")
        self._index: str = index
        self._api_key: Optional[str] = api_key
        self._client: Any = None  # AsyncElasticsearch instance
        self._connected: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Open the Elasticsearch connection and ensure the index exists.

        Creates the index with the correct mapping if it does not already
        exist.  Sets ``self._connected = True`` on success.

        Returns
        -------
        bool:
            ``True`` if the connection was established successfully,
            ``False`` otherwise.  Never raises.
        """
        try:
            from elasticsearch import AsyncElasticsearch  # type: ignore

            client_kwargs: Dict[str, Any] = {"hosts": [self._url]}
            if self._api_key:
                # Elastic Cloud keys are base64-encoded and contain an embedded ":"
                # inside the base64 payload — do NOT split on ":" naively.
                # Pass the key directly as a string; the elasticsearch-py client
                # accepts encoded API keys in the Authorization header as-is.
                # Only split into (id, value) tuple for legacy "id:secret" style keys
                # that are short and where the id portion has no "=" padding chars.
                key = self._api_key
                is_base64_cloud_key = "=" in key or len(key) > 50
                if not is_base64_cloud_key and ":" in key:
                    parts = key.split(":", 1)
                    client_kwargs["api_key"] = (parts[0], parts[1])
                else:
                    client_kwargs["api_key"] = key

            self._client = AsyncElasticsearch(**client_kwargs)

            # Ping to verify connectivity
            if not await self._client.ping():
                logger.warning("elasticsearch_ping_failed", extra={"url": self._url})
                await self._client.close()
                self._client = None
                return False

            # Ensure the index exists with the correct mapping
            await self._ensure_index()

            self._connected = True
            logger.info(
                "elasticsearch_connected",
                extra={"url": self._url, "index": self._index},
            )
            return True

        except ImportError:
            logger.warning(
                "elasticsearch_not_installed",
                extra={
                    "hint": "pip install 'elasticsearch[async]'",
                    "url": self._url,
                },
            )
            return False
        except Exception as exc:
            logger.warning(
                "elasticsearch_connect_error",
                extra={"url": self._url, "error": str(exc)[:200]},
            )
            if self._client:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None
            return False

    async def disconnect(self) -> None:
        """Close the Elasticsearch client gracefully.

        Safe to call even if the client was never connected.
        """
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("elasticsearch_disconnected")
            except Exception as exc:
                logger.warning(
                    "elasticsearch_disconnect_error",
                    extra={"error": str(exc)[:100]},
                )
            finally:
                self._client = None
                self._connected = False

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    async def index_datasets(
        self,
        datasets: list,
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """Bulk upsert *datasets* into the Elasticsearch index.

        Uses ``canonical_id`` as the document ``_id`` so repeated calls
        with the same dataset produce an upsert (update if exists, insert
        otherwise) rather than duplicate documents.

        Parameters
        ----------
        datasets:
            List of ``RawDataset`` objects to index.
        embeddings:
            Optional list of embedding vectors, one per dataset, in the same
            order.  If provided, each embedding is attached to its
            corresponding document.

        Returns
        -------
        int:
            Number of documents successfully indexed.  Returns ``0`` on any
            error.
        """
        if not self._connected or self._client is None:
            return 0
        if not datasets:
            return 0
        try:
            from elasticsearch.helpers import async_bulk  # type: ignore

            actions = []
            for i, ds in enumerate(datasets):
                doc = self._raw_dataset_to_document(ds)
                if embeddings and i < len(embeddings) and embeddings[i]:
                    doc["embedding"] = embeddings[i]
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": self._index,
                        "_id": ds.canonical_id,
                        "_source": doc,
                    }
                )

            success, failed = await async_bulk(
                self._client,
                actions,
                raise_on_error=False,
                raise_on_exception=False,
            )
            if failed:
                logger.warning(
                    "elasticsearch_bulk_partial_failure",
                    extra={"succeeded": success, "failed": len(failed)},
                )
            else:
                logger.debug(
                    "elasticsearch_bulk_indexed",
                    extra={"count": success, "index": self._index},
                )
            return success
        except Exception as exc:
            logger.warning(
                "elasticsearch_index_error",
                extra={"error": str(exc)[:200]},
            )
            return 0

    # ──────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 20,
    ) -> list:
        """Execute a hybrid BM25 + kNN search using ES RRF (ES 8.8+).

        Falls back to plain ``multi_match`` BM25 search if the Elasticsearch
        version does not support the ``retriever`` / RRF API.

        Parameters
        ----------
        query:
            The text query string.
        query_embedding:
            384-dimensional embedding vector for the query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RawDataset]:
            Reconstructed dataset objects from ES hits.  Returns ``[]`` on any
            error or when no documents match.
        """
        if not self._connected or self._client is None:
            return []
        try:
            query_body = self._build_rrf_query(query, query_embedding, top_k)
            response = await self._client.search(
                index=self._index,
                body=query_body,
            )
            return self._hits_to_datasets(response)
        except Exception as exc:
            error_str = str(exc)
            # RRF not supported (ES < 8.8) — fall back to BM25
            if "retriever" in error_str.lower() or "unknown" in error_str.lower():
                logger.info(
                    "elasticsearch_rrf_not_supported_fallback",
                    extra={"error": error_str[:120]},
                )
                return await self.text_search(query, top_k=top_k)
            logger.warning(
                "elasticsearch_hybrid_search_error",
                extra={"error": error_str[:200]},
            )
            return []

    async def text_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list:
        """BM25-only text search fallback.

        Used when no query embedding is available, or as fallback when RRF
        is not supported by the deployed Elasticsearch version.

        Parameters
        ----------
        query:
            The text query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RawDataset]:
            Reconstructed dataset objects from ES hits.  ``[]`` on error.
        """
        if not self._connected or self._client is None:
            return []
        try:
            query_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "description^1", "tags^2"],
                        "type": "best_fields",
                    }
                },
            }
            response = await self._client.search(
                index=self._index,
                body=query_body,
            )
            return self._hits_to_datasets(response)
        except Exception as exc:
            logger.warning(
                "elasticsearch_text_search_error",
                extra={"error": str(exc)[:200]},
            )
            return []

    async def get_search_history(self, limit: int = 10) -> List[dict]:
        """Return recent search history records.

        Search history is persisted in SQLite (via ``SearchRepository``), not
        in Elasticsearch.  This method is a no-op stub that always returns an
        empty list.  It exists so that callers can treat this engine
        polymorphically without needing to know where search history lives.

        Returns
        -------
        list[dict]:
            Always ``[]``.
        """
        return []

    async def log_search(
        self,
        query: str,
        results_count: int,
        request_id: str,
    ) -> None:
        """No-op stub — search logging is handled by ``SearchRepository``.

        Parameters
        ----------
        query:
            The search query (ignored).
        results_count:
            Number of results returned (ignored).
        request_id:
            Request identifier (ignored).
        """
        return

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _ensure_index(self) -> None:
        """Create the index with the correct mapping if it does not exist.

        Called during :meth:`connect`.  Safe to call multiple times.

        On Elastic Cloud free tier, index auto-creation via API may be
        restricted by cluster permissions.  In that case we log a clear
        human-friendly hint and continue — the index may already exist
        or can be created manually via the Kibana console.
        """
        try:
            exists = await self._client.indices.exists(index=self._index)
            if not exists:
                try:
                    await self._client.indices.create(
                        index=self._index,
                        body=INDEX_MAPPING,
                    )
                    logger.info(
                        "elasticsearch_index_created",
                        extra={"index": self._index},
                    )
                except Exception as create_exc:
                    err_str = str(create_exc).lower()
                    if any(k in err_str for k in ("security_exception", "authorization", "forbidden", "403", "unauthorized")):
                        logger.warning(
                            "elasticsearch_index_create_permission_denied",
                            extra={
                                "index": self._index,
                                "hint": (
                                    f"Elastic Cloud cluster does not allow auto index creation. "
                                    f"Create index '{self._index}' manually in Kibana: "
                                    f"Stack Management → Index Management → Create Index."
                                ),
                            },
                        )
                    else:
                        logger.warning(
                            "elasticsearch_ensure_index_error",
                            extra={"index": self._index, "error": str(create_exc)[:200]},
                        )
                    # Do NOT raise — connection is still valid, searches may still work
                    # if the index already exists from a prior run.
            else:
                logger.info(
                    "elasticsearch_index_exists",
                    extra={"index": self._index},
                )
        except Exception as exc:
            logger.warning(
                "elasticsearch_ensure_index_error",
                extra={"index": self._index, "error": str(exc)[:200]},
            )

    def _build_rrf_query(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int,
    ) -> Dict[str, Any]:
        """Build an ES RRF hybrid retriever query body (ES 8.8+).

        Parameters
        ----------
        query:
            Text query string.
        query_embedding:
            Dense vector for kNN retrieval.
        top_k:
            Desired result count.

        Returns
        -------
        dict:
            Elasticsearch query body dict.
        """
        return {
            "size": top_k,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": [
                                            "title^3",
                                            "description^1",
                                            "tags^2",
                                        ],
                                        "type": "best_fields",
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": query_embedding,
                                "num_candidates": top_k * 5,
                                "k": top_k,
                            }
                        },
                    ],
                    "rank_window_size": top_k * 2,
                    "rank_constant": 60,
                }
            },
        }

    def _hits_to_datasets(self, response: Dict[str, Any]) -> list:
        """Convert an ES search response to a list of ``RawDataset`` objects.

        Parameters
        ----------
        response:
            The raw response dict returned by ``self._client.search()``.

        Returns
        -------
        list[RawDataset]:
            Successfully reconstructed datasets.  Documents that fail to parse
            are skipped (logged at DEBUG level).
        """
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            ds = self._document_to_raw_dataset(hit)
            if ds is not None:
                results.append(ds)
        return results

    def _document_to_raw_dataset(self, hit: Dict[str, Any]) -> Optional[Any]:
        """Reconstruct a ``RawDataset`` from an Elasticsearch hit dict.

        Parameters
        ----------
        hit:
            A single hit from an ES search response, with ``_id`` and
            ``_source`` keys.

        Returns
        -------
        RawDataset or None:
            The reconstructed dataset, or ``None`` if the document is
            malformed or missing required fields.
        """
        try:
            from datascout.contracts.models import RawDataset  # noqa: PLC0415

            source: Dict[str, Any] = hit.get("_source", {})
            if not source:
                return None

            # Reconstruct datetime fields
            def _parse_dt(val: Any) -> Optional[datetime]:
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val
                try:
                    dt = datetime.fromisoformat(str(val))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return None

            # Re-hydrate enum lists — store as plain strings, Pydantic coerces
            return RawDataset(
                canonical_id=source.get("canonical_id", hit.get("_id", "")),
                title=source.get("title", ""),
                description=source.get("description", ""),
                source=source.get("source", ""),
                source_url=source.get("source_url", ""),
                source_id=source.get("source_id", source.get("canonical_id", "")),
                tags=source.get("tags") or [],
                tags_primary=source.get("tags_primary") or [],
                task_types=source.get("task_types") or [],
                modalities=source.get("modalities") or [],
                row_count=source.get("row_count"),
                column_count=source.get("column_count"),
                column_names=source.get("column_names"),
                file_size_bytes=source.get("file_size_bytes"),
                license_type=source.get("license_type"),
                last_updated=_parse_dt(source.get("last_updated")),
                download_count=source.get("download_count"),
                upvote_count=source.get("upvote_count"),
                author=source.get("author"),
                metadata_completeness=float(
                    source.get("metadata_completeness", 0.0) or 0.0
                ),
                dataset_fingerprint=source.get("dataset_fingerprint", ""),
                ingestion_timestamp=_parse_dt(source.get("ingestion_timestamp"))
                or datetime.now(tz=timezone.utc),
                pipeline_version=source.get("pipeline_version", "unknown"),
                ingestion_version=source.get("ingestion_version", "unknown"),
            )
        except Exception as exc:
            logger.debug(
                "elasticsearch_document_parse_failed",
                extra={"id": hit.get("_id", "?"), "error": str(exc)[:120]},
            )
            return None

    def _raw_dataset_to_document(self, ds: Any) -> Dict[str, Any]:
        """Convert a ``RawDataset`` to an Elasticsearch document dict.

        Enum values are serialised to their string ``.value`` so they are
        stored as plain keywords.  ``None`` values are omitted to keep
        documents lean.

        Parameters
        ----------
        ds:
            A ``RawDataset`` instance.

        Returns
        -------
        dict:
            Elasticsearch-ready document dict (no ``embedding`` key — that is
            attached separately by :meth:`index_datasets`).
        """
        def _enum_list(lst: list) -> list:
            """Serialise a list of enums/strings to plain strings."""
            if not lst:
                return []
            out = []
            for item in lst:
                if hasattr(item, "value"):
                    out.append(str(item.value))
                else:
                    out.append(str(item))
            return out

        def _dt_iso(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val)

        doc: Dict[str, Any] = {
            "canonical_id": ds.canonical_id,
            "title": ds.title or "",
            "description": ds.description or "",
            "source": ds.source or "",
            "source_url": ds.source_url or "",
            "source_id": ds.source_id or "",
            "tags": _enum_list(ds.tags or []),
            "tags_primary": _enum_list(ds.tags_primary or []),
            "task_types": _enum_list(ds.task_types or []),
            "modalities": _enum_list(ds.modalities or []),
            "metadata_completeness": float(ds.metadata_completeness or 0.0),
            "dataset_fingerprint": ds.dataset_fingerprint or "",
            "pipeline_version": ds.pipeline_version or "unknown",
            "ingestion_version": ds.ingestion_version or "unknown",
            "ingestion_timestamp": _dt_iso(ds.ingestion_timestamp),
        }

        # Optional fields — only include when present
        optional_fields = [
            "row_count",
            "column_count",
            "file_size_bytes",
            "download_count",
            "upvote_count",
            "author",
        ]
        for f in optional_fields:
            val = getattr(ds, f, None)
            if val is not None:
                doc[f] = val

        if ds.license_type is not None:
            doc["license_type"] = (
                ds.license_type.value
                if hasattr(ds.license_type, "value")
                else str(ds.license_type)
            )

        if ds.last_updated is not None:
            doc["last_updated"] = _dt_iso(ds.last_updated)

        return doc