"""
datascout.tools.dataset_sources.huggingface_tool
------------------------------------------------
HuggingFace datasets hub search adapter.

Uses the HuggingFace Hub API to search and retrieve dataset metadata.

Installation:
    pip install huggingface_hub

Authentication:
    Optional - token only needed for private datasets
    Set HF_TOKEN environment variable or HUGGINGFACE_TOKEN in settings

API Docs:
    https://huggingface.co/docs/huggingface_hub
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from datascout.contracts import RawDataset
from datascout.contracts.errors import wrap_external_exception
from datascout.infrastructure.config import get_settings
from datascout.infrastructure.logging import get_logger
from datascout.tools.base import BaseTool
from datascout.tools.utilities.rate_limiter import get_rate_limiter

if TYPE_CHECKING:
    from datascout.contracts import SearchQuery

logger = get_logger(__name__)


class HuggingFaceTool(BaseTool):
    """HuggingFace datasets hub search adapter."""
    
    def __init__(self):
        settings = get_settings()
        super().__init__(name="huggingface", timeout_seconds=settings.huggingface_timeout)
        
        self.token = settings.huggingface_token
        self.rate_limiter = get_rate_limiter("huggingface", rate=settings.rate_limit_huggingface, per=60.0)
        self._api_client = None
    
    def _get_client(self):
        if self._api_client is None:
            try:
                from huggingface_hub import HfApi
                self._api_client = HfApi(token=self.token)
                logger.info("HuggingFace API client initialized")
            except Exception as e:
                raise wrap_external_exception(e, layer="tool") from e
        return self._api_client
    
    async def search(self, query: "SearchQuery") -> list[RawDataset]:
        logger.info("Searching HuggingFace datasets", extra={"query": query.query_string})
        
        async with self.rate_limiter:
            try:
                api = self._get_client()
                results = api.list_datasets(search=query.query_string, limit=query.max_results)
                
                datasets = []
                for item in results:
                    try:
                        dataset = RawDataset(
                            source_adapter="huggingface",
                            parent_search_id=query.search_id,
                            external_id=item.id,
                            title=item.id.split('/')[-1] if '/' in item.id else item.id,
                            description=getattr(item, 'description', None) or item.id,
                            url=f"https://huggingface.co/datasets/{item.id}",
                            tags=getattr(item, 'tags', []),
                            download_count=getattr(item, 'downloads', None),
                            last_updated=getattr(item, 'last_modified', None),
                            popularity_score=float(getattr(item, 'downloads', 0)) if hasattr(item, 'downloads') else None,
                        )
                        datasets.append(dataset)
                    except Exception as e:
                        logger.warning(f"Failed to parse HF dataset: {e}")
                        continue
                
                logger.info(f"HuggingFace search complete: {len(datasets)} results")
                return datasets
                
            except Exception as e:
                logger.error(f"HuggingFace API error: {e}", exc_info=True)
                raise wrap_external_exception(e, layer="tool") from e
    
    async def health_check(self) -> bool:
        try:
            api = self._get_client()
            list(api.list_datasets(limit=1))
            return True
        except Exception:
            return False