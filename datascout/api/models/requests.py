"""
datascout.api.models.requests
──────────────────────────────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: FastAPI request body models — the API surface layer.

WHY separate API models from contract models (contracts/requests.py)?
  - API models: HTTP-layer concerns (JSON body, OpenAPI schema, validation errors)
  - Contract models: pipeline-layer concerns (distributed trace, adapter targeting)
  - Separation: API can evolve independently of internal contracts
  - Mapping layer: DatasetSearchRequest → SearchQuery (explicit, auditable)

Author: Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class FilterOptions(BaseModel):
    """Hard constraint filters for dataset search."""

    min_rows: int | None = Field(default=None, ge=0, description="Minimum row count")
    max_rows: int | None = Field(default=None, ge=0, description="Maximum row count")
    require_description: bool = Field(
        default=False, description="Only return datasets with descriptions"
    )
    require_schema_info: bool = Field(
        default=False, description="Only return datasets with column names"
    )
    require_license_info: bool = Field(
        default=False, description="Only return datasets with known license"
    )
    min_completeness: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum metadata completeness [0, 1]"
    )


class PreferenceOptions(BaseModel):
    """Soft preference signals for ranking adjustment."""

    prefer_recent: bool = Field(default=False, description="Boost recently updated datasets")
    prefer_popular: bool = Field(default=False, description="Boost high-download datasets")
    prefer_complete: bool = Field(default=True, description="Boost high-completeness datasets")
    result_diversity: float = Field(
        default=0.3, ge=0.0, le=1.0, description="0=pure relevance, 1=max diversity"
    )
    explanation_detail: str = Field(
        default="standard", description="minimal|standard|detailed"
    )

    @field_validator("explanation_detail")
    @classmethod
    def validate_detail(cls, v: str) -> str:
        valid = {"minimal", "standard", "detailed"}
        if v not in valid:
            raise ValueError(f"explanation_detail must be one of {valid}")
        return v


class DatasetSearchRequest(BaseModel):
    """
    POST /api/v1/datasets/search request body.

    Maps to contracts.requests.SearchQuery after validation.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language dataset search query",
        examples=["image classification dataset with 100k+ samples"],
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return [1, 50]",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Adapter sources to query. Empty = all. Options: kaggle, huggingface, openml",
    )
    filters: FilterOptions | None = Field(
        default=None,
        description="Hard constraint filters (applied as boolean gates)",
    )
    preferences: PreferenceOptions | None = Field(
        default=None,
        description="Soft ranking preferences (applied as score adjustments)",
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        valid = {"kaggle", "huggingface", "openml"}
        invalid = [s for s in v if s not in valid]
        if invalid:
            raise ValueError(f"Invalid sources: {invalid}. Valid: {sorted(valid)}")
        return v

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "query": "medical imaging dataset for tumor classification",
                "max_results": 5,
                "sources": ["kaggle", "huggingface"],
                "filters": {"min_rows": 1000, "require_description": True},
                "preferences": {"prefer_recent": True, "explanation_detail": "detailed"},
            }
        ]
    }}
