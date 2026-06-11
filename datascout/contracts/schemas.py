"""
datascout.contracts.schemas
────────────────────────────
Compatibility shim for legacy agent imports.

The canonical contracts are in contracts.models and contracts.requests.
This module provides DatasetCandidate, SearchQuery, and SearchResult
as the Level-0 agent expects them.

DatasetCandidate is a lightweight view over RawDataset + score,
used within the agent pipeline before the full EvaluatedDataset is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Re-export SearchQuery from its canonical location
from .requests import SearchQuery  # noqa: F401


@dataclass
class DatasetCandidate:
    """
    Lightweight dataset representation used within the ReAct loop.

    Bridges the gap between RawDataset (adapter output) and
    EvaluatedDataset (full scoring output). Used as the in-flight
    contract during the search → evaluate → rank cycle.
    """
    # Identity
    canonical_id: str
    title: str
    source: str
    source_url: str

    # Scores
    final_score: float = 0.0
    relevance_score: float = 0.0
    quality_score: float = 0.0

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    row_count: Optional[int] = None
    last_updated: Optional[str] = None
    download_count: Optional[int] = None

    # Explainability
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    bias_signals: list[dict] = field(default_factory=list)
    explanation: Optional[str] = None

    # Internal reference to original RawDataset (optional)
    _raw: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def from_raw_dataset(cls, raw: Any, final_score: float = 0.0) -> "DatasetCandidate":
        """
        Build a DatasetCandidate from a RawDataset object.
        Used by the HybridEngine → ScoutAgent bridge.
        """
        return cls(
            canonical_id=raw.canonical_id,
            title=raw.title,
            source=raw.source,
            source_url=raw.source_url,
            final_score=final_score,
            description=raw.description_short or raw.description or "",
            tags=raw.tags_primary or [],
            task_types=[t.value for t in (raw.task_types or [])],
            modalities=[m.value for m in (raw.modalities or [])],
            row_count=raw.row_count,
            download_count=raw.download_count,
            _raw=raw,
        )

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "final_score": self.final_score,
            "description": self.description,
            "tags": self.tags,
            "task_types": self.task_types,
            "modalities": self.modalities,
            "row_count": self.row_count,
            "download_count": self.download_count,
            "score_breakdown": self.score_breakdown,
            "bias_signals": self.bias_signals,
            "explanation": self.explanation,
        }


@dataclass
class SearchResult:
    """
    Final output of a ScoutAgent.run() call.
    Contains ranked DatasetCandidates + metadata.
    """
    query: SearchQuery
    datasets: list[DatasetCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict() if hasattr(self.query, "to_dict") else str(self.query),
            "datasets": [d.to_dict() for d in self.datasets],
            "metadata": self.metadata,
        }

    def to_dict_safe(self) -> dict:
        """API-safe representation."""
        return {
            "query_id": getattr(self.query, "query_id", ""),
            "results": [d.to_dict() for d in self.datasets],
            "total_found": len(self.datasets),
            "confidence": self.metadata.get("confidence", "low"),
            "partial_result": self.metadata.get("partial_result", False),
            "processing_time_ms": self.metadata.get("elapsed_ms", 0),
            "error_message": self.metadata.get("agent_error", None),
        }