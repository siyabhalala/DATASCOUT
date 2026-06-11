"""
datascout.search
─────────────────────────────────────────────────────
Phase 10: Semantic Search Layer

Public API:
    from datascout.search import HybridEngine, BM25Index, EmbeddingEngine, VectorIndex
    from datascout.search import SearchResult, HybridSearchOutput
"""

from .bm25_index import BM25Index, _tokenize as tokenize_query
from .embedding_engine import EmbeddingEngine
from .vector_index import VectorIndex, VectorBackend
from .hybrid_engine import HybridEngine, SearchResult, HybridSearchOutput

__all__ = [
    "BM25Index",
    "EmbeddingEngine",
    "VectorIndex",
    "VectorBackend",
    "HybridEngine",
    "SearchResult",
    "HybridSearchOutput",
    "tokenize_query",
]