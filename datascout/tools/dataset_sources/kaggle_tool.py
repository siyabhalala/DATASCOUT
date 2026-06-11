"""
datascout.tools.dataset_sources.kaggle_tool
-------------------------------------------
Kaggle dataset search adapter.

Uses the official Kaggle API client to search and retrieve dataset metadata.

Installation:
    pip install kaggle

Authentication:
    Set environment variables:
        KAGGLE_USERNAME=<your_username>
        KAGGLE_KEY=<your_api_key>
    
    OR create ~/.kaggle/kaggle.json:
        {"username": "<your_username>", "key": "<your_api_key>"}

API Docs:
    https://github.com/Kaggle/kaggle-api
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from datascout.contracts import RawDataset
from datascout.contracts.errors import (
    ToolAPIError,
    ToolAuthError,
    ToolNetworkError,
    ToolParseError,
    wrap_external_exception,
)
from datascout.infrastructure.config import get_settings
from datascout.infrastructure.logging import get_logger
from datascout.infrastructure.monitoring import MetricNames, metrics
from datascout.tools.base import BaseTool
from datascout.tools.utilities.rate_limiter import get_rate_limiter

if TYPE_CHECKING:
    from datascout.contracts import SearchQuery

logger = get_logger(__name__)


class KaggleTool(BaseTool):
    """
    Kaggle dataset search adapter.
    
    Searches Kaggle's dataset catalog via their public API.
    Rate-limited to respect platform limits.
    """
    
    def __init__(self):
        """
        Initialize Kaggle adapter.
        
        Credentials loaded from:
        1. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
        2. Settings (from .env)
        3. ~/.kaggle/kaggle.json
        """
        settings = get_settings()
        
        super().__init__(
            name="kaggle",
            timeout_seconds=settings.kaggle_timeout
        )
        
        # Set environment variables if provided in settings
        if settings.kaggle_username:
            os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        if settings.kaggle_key:
            os.environ["KAGGLE_KEY"] = settings.kaggle_key
        
        # Initialize rate limiter
        self.rate_limiter = get_rate_limiter(
            "kaggle",
            rate=settings.rate_limit_kaggle,
            per=60.0
        )
        
        # Lazy-load Kaggle API client
        self._api_client = None
    
    def _get_client(self):
        """
        Lazy-initialize Kaggle API client.
        
        Raises:
            ToolAuthError: If credentials are missing or invalid
        """
        if self._api_client is None:
            try:
                from kaggle import KaggleApi
                
                api = KaggleApi()
                api.authenticate()
                self._api_client = api
                
                logger.info("Kaggle API client authenticated successfully")
                
            except OSError as e:
                # Credentials file not found or malformed
                raise ToolAuthError(
                    "Kaggle authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables "
                    "or create ~/.kaggle/kaggle.json with your credentials.",
                    context={"error": str(e)}
                ) from e
            except Exception as e:
                raise wrap_external_exception(e, layer="tool") from e
        
        return self._api_client
    
    async def search(self, query: "SearchQuery") -> list[RawDataset]:
        """
        Search Kaggle datasets.
        
        Args:
            query: Search instruction
            
        Returns:
            List of RawDataset records
            
        Raises:
            ToolAuthError: If API credentials are invalid
            ToolNetworkError: If network request fails
            ToolAPIError: If API returns error response
            ToolParseError: If response parsing fails
        """
        logger.info(
            "Searching Kaggle datasets",
            extra={
                "query_string": query.query_string,
                "max_results": query.max_results,
            },
        )
        
        # Rate limit enforcement
        async with self.rate_limiter:
            try:
                api = self._get_client()
                
                # Kaggle API is synchronous - run in executor if needed
                # For now, direct call (fast enough for metadata)
                results = api.dataset_list(
                    search=query.query_string,
                    page_size=min(query.max_results, 100),  # API max is 100
                )
                
                datasets = []
                for item in results:
                    try:
                        dataset = self._parse_kaggle_dataset(item, query.search_id)
                        datasets.append(dataset)
                    except ToolParseError as e:
                        # Log parse error but continue with other results
                        logger.warning(
                            "Failed to parse Kaggle dataset",
                            extra={"dataset_ref": getattr(item, 'ref', 'unknown'), "error": str(e)},
                        )
                        metrics.increment(
                            MetricNames.TOOL_ERRORS,
                            labels={"adapter": "kaggle", "error_type": "parse_error"}
                        )
                        continue
                
                logger.info(
                    "Kaggle search complete",
                    extra={"result_count": len(datasets)},
                )
                
                return datasets
                
            except ToolAuthError:
                # Re-raise auth errors (already wrapped)
                raise
            
            except Exception as e:
                # Wrap unexpected errors
                logger.error(
                    "Kaggle API error",
                    extra={"error": str(e)},
                    exc_info=True,
                )
                raise wrap_external_exception(e, layer="tool") from e
    
    def _parse_kaggle_dataset(self, api_response, search_id: str) -> RawDataset:
        """
        Parse Kaggle API response into RawDataset.
        
        Args:
            api_response: Kaggle API dataset object
            search_id: Parent SearchQuery.search_id
            
        Returns:
            Validated RawDataset instance
            
        Raises:
            ToolParseError: If required fields are missing or malformed
        """
        try:
            # Kaggle API returns objects with attributes
            ref = api_response.ref  # e.g., "username/dataset-name"
            
            # Calculate freshness score if last_updated available
            freshness_score = None
            last_updated = None
            if hasattr(api_response, 'lastUpdated') and api_response.lastUpdated:
                try:
                    last_updated = datetime.fromisoformat(api_response.lastUpdated.replace('Z', '+00:00'))
                    days_old = (datetime.now(last_updated.tzinfo) - last_updated).days
                    # Exponential decay: 1.0 for today, 0.5 for 90 days, 0.1 for 1 year
                    freshness_score = max(0.0, min(1.0, 1.0 / (1.0 + days_old / 90.0)))
                except Exception:
                    pass  # Ignore freshness calculation errors
            
            # Convert quality score to 0-100 scale
            raw_quality = getattr(api_response, 'usabilityRating', None)
            quality_score = (raw_quality * 100) if raw_quality else None
            
            return RawDataset(
                source_adapter="kaggle",
                parent_search_id=search_id,
                external_id=ref,
                title=getattr(api_response, 'title', None),
                description=getattr(api_response, 'subtitle', None),
                url=f"https://www.kaggle.com/datasets/{ref}",
                download_url=getattr(api_response, 'downloadUrl', None),
                license_raw=getattr(api_response, 'licenseName', None),
                file_formats=[],  # Kaggle doesn't expose formats in search results
                row_count=None,  # Not available in search results
                column_count=None,
                file_size_bytes=getattr(api_response, 'totalBytes', None),
                tags=getattr(api_response, 'tags', []),
                download_count=getattr(api_response, 'downloadCount', None),
                last_updated=last_updated,
                quality_score=quality_score,  # Now 0-100 scale
                popularity_score=float(getattr(api_response, 'downloadCount', 0)),
                freshness_score=freshness_score,
                extra={
                    "vote_count": getattr(api_response, 'voteCount', None),
                    "view_count": getattr(api_response, 'viewCount', None),
                },
            )
            
        except AttributeError as e:
            raise ToolParseError(
                f"Failed to parse Kaggle dataset response: missing attribute {e}",
                context={"raw_response": str(api_response)},
            ) from e
        except (TypeError, ValueError) as e:
            raise ToolParseError(
                f"Failed to parse Kaggle dataset response: {e}",
                context={"raw_response": str(api_response)},
            ) from e
    
    async def health_check(self) -> bool:
        """
        Check if Kaggle API is accessible.
        
        Returns:
            True if API is reachable and credentials are valid
        """
        try:
            api = self._get_client()
            # Minimal request to verify connectivity
            api.dataset_list(page_size=1)
            return True
        except Exception as e:
            logger.warning("Kaggle health check failed", extra={"error": str(e)})
            return False