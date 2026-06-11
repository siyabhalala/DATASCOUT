"""
datascout.tools.dataset_sources.openml_tool
-------------------------------------------
OpenML dataset search adapter.

Installation:
    pip install openml

API Docs:
    https://openml.github.io/openml-python
"""

from __future__ import annotations

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


class OpenMLTool(BaseTool):
    """OpenML dataset search adapter."""
    
    def __init__(self):
        settings = get_settings()
        super().__init__(name="openml", timeout_seconds=settings.openml_timeout)
        
        self.api_key = settings.openml_api_key
        self.rate_limiter = get_rate_limiter("openml", rate=settings.rate_limit_openml, per=60.0)
    
    async def search(self, query: "SearchQuery") -> list[RawDataset]:
        logger.info("Searching OpenML datasets", extra={"query": query.query_string})
        
        async with self.rate_limiter:
            try:
                import openml
                
                if self.api_key:
                    openml.config.apikey = self.api_key
                
                # Search datasets
                datasets_df = openml.datasets.list_datasets(output_format='dataframe')
                
                # Filter by query string in name or description
                mask = datasets_df['name'].str.contains(query.query_string, case=False, na=False)
                filtered = datasets_df[mask].head(query.max_results)
                
                datasets = []
                for idx, row in filtered.iterrows():
                    dataset = RawDataset(
                        source_adapter="openml",
                        parent_search_id=query.search_id,
                        external_id=str(row['did']),
                        title=row['name'],
                        description=row.get('description', ''),
                        url=f"https://www.openml.org/d/{row['did']}",
                        row_count=int(row['NumberOfInstances']) if 'NumberOfInstances' in row else None,
                        column_count=int(row['NumberOfFeatures']) if 'NumberOfFeatures' in row else None,
                        popularity_score=float(row.get('runs', 0)),
                    )
                    datasets.append(dataset)
                
                logger.info(f"OpenML search complete: {len(datasets)} results")
                return datasets
                
            except Exception as e:
                logger.error(f"OpenML API error: {e}", exc_info=True)
                raise wrap_external_exception(e, layer="tool") from e
    
    async def health_check(self) -> bool:
        try:
            import openml
            openml.datasets.list_datasets(size=1)
            return True
        except Exception:
            return False