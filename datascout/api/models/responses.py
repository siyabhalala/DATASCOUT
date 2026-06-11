"""
datascout.api.models.responses
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: FastAPI response models — typed output contracts for
OpenAPI schema generation and client SDK generation.

WHY Pydantic response models (not just JSONResponse)?
  - FastAPI auto-generates OpenAPI schema from response_model= parameter
  - Client SDKs (TypeScript, Python) are generated from OpenAPI schema
  - Type safety: FastAPI validates response structure before sending
  - Documentation: response fields appear in /docs with descriptions

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreDimensionOut(BaseModel):
    """One scoring dimension in the result."""

    name: str
    weighted_score: float
    explanation: str


class DatasetOut(BaseModel):
    """Safe dataset representation for API responses."""

    canonical_id: str
    title: str
    description_short: str
    source: str
    source_url: str
    tags_primary: list[str] = []
    row_count: int | None = None
    column_count: int | None = None
    file_size_bytes: int | None = None
    data_format: str | None = None
    license_type: str | None = None
    data_domain: str | None = None
    task_types: list[str] = []
    metadata_completeness: float = 0.0
    quality_tier: str = "incomplete"
    is_duplicate: bool = False
    composite_score: float = 0.0
    rank: int | None = None
    score_dimensions: list[ScoreDimensionOut] = []


class ExplanationOut(BaseModel):
    """Structured LLM explanation."""

    summary: str
    key_factors: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendations: list[str] = []
    confidence: float = 0.0
    model_used: str = ""
    generated_at: str = ""


class RankedResultOut(BaseModel):
    """Single ranked result — dataset + explanation."""

    rank: int
    dataset: DatasetOut
    explanation: ExplanationOut | None = None
    download_url: str | None = None


class DatasetSearchResponse(BaseModel):
    """
    POST /api/v1/datasets/search response.

    Maps from contracts.responses.AgentResponse.
    """

    query_id: str
    results: list[RankedResultOut] = []
    total_found: int = 0
    confidence: str = "low"
    partial_result: bool = False
    adapter_errors: dict[str, str] = {}
    processing_time_ms: int = 0
    total_cost_usd: float = 0.0
    responded_at: str = ""
    error_message: str | None = None
    response_version: str = "3.0.0"


class SingleDatasetResponse(BaseModel):
    """GET /api/v1/datasets/{id} response."""

    dataset: dict[str, Any]
    request_id: str = ""


class DatasetListResponse(BaseModel):
    """GET /api/v1/datasets/ response with cursor pagination."""

    datasets: list[dict[str, Any]] = []
    next_cursor: str | None = None
    has_more: bool = False
    total_count: int = 0
    error_message: str | None = None
    request_id: str = ""


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(default="", description="Request ID for log correlation")
    details: list[Any] | None = Field(default=None, description="Validation error details")
