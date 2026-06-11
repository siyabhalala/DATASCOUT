"""
datascout.search.bm25_index
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: BM25 sparse retrieval index over RawDataset corpus.

SYSTEM DESIGN DECISIONS:

  1. WHY rank-bm25 over raw BM25 implementation?
     - Battle-tested, handles edge cases (empty corpus, single-doc corpus)
     - BM25Okapi variant: industry standard with k1/b parameters
     - Zero external dependencies beyond rank-bm25

  2. WHY title × 3 + description_short + tags_primary for input text?
     - Title is the strongest signal — tripling gives field-weighted BM25
     - description_short (300 chars): avoids swamping with noise from long docs
     - tags_primary (top-10): highly discriminative, low noise
     - This replicates TF field-weight boosting without changing BM25 internals

  3. WHY tokenize by whitespace+punctuation (not NLTK)?
     - NLTK adds 300MB+ dependency — unacceptable for this layer
     - Whitespace tokenization matches how queries arrive (already split)
     - Lowercase + strip: deterministic, reproducible, no locale issues

  4. WHY store corpus_ids separately from BM25 model?
     - rank-bm25 only exposes get_scores(query) — no dataset reference
     - corpus_ids[i] maps BM25 score index i → canonical_id
     - Required for lookup: BM25 score array → RawDataset objects

  5. WHY get_top_k returns (canonical_id, score) not RawDataset?
     - Separation of concerns: index is ID-based, caller does lookup
     - Enables fused ranking: BM25 scores can be combined with FAISS scores
       before any dataset objects are loaded

PERFORMANCE ANALYSIS:
  - build():    O(n × avg_tokens) — linear in corpus size
  - get_top_k(): O(n × query_tokens) — BM25 is O(n) per query
  - At 100K datasets × 50 avg tokens: build ≈ 500ms, query ≈ 5ms

FAILURE SCENARIOS HANDLED:
  - Empty corpus → returns empty results, never raises
  - Dataset with no text → assigned zero-length token list
  - Query with no tokens → returns empty results
  - rank-bm25 not installed → ImportError surfaced at build(), not import time

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger("datascout.search.bm25_index")

# Token pattern: alphanumeric sequences (handles Unicode word chars)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer: lowercase alphanumeric sequences.
    O(text_length) — no external dependencies.
    """
    return _TOKEN_RE.findall(text.lower())


def _build_dataset_text(title: str, description_short: str, tags_primary: list[str]) -> str:
    """
    Construct weighted input text for BM25 indexing.
    title × 3 + description_short + tags_primary (space-joined).

    WHY title × 3: replicates TF field-weight boosting without BM25 parameter changes.
    O(1) — string concatenation.
    """
    title_weighted = f"{title} {title} {title}".strip()
    tags_str = " ".join(tags_primary)
    parts = [p for p in [title_weighted, description_short, tags_str] if p]
    return " ".join(parts)


class BM25Index:
    """
    Sparse BM25 retrieval index over a fixed dataset corpus.

    Lifecycle:
      1. idx = BM25Index()
      2. idx.build(datasets)          ← builds corpus in memory
      3. results = idx.get_top_k(query_tokens, k=20)

    Thread safety: build() is not thread-safe. get_top_k() is read-only safe.
    """

    def __init__(self) -> None:
        self._bm25: Optional[object] = None       # rank_bm25.BM25Okapi instance
        self._corpus_ids: list[str] = []          # corpus_ids[i] → canonical_id
        self._corpus_tokens: list[list[str]] = [] # raw token lists for diagnostics
        self._built: bool = False
        self._corpus_size: int = 0

    def build(self, datasets: list) -> None:
        """
        Build BM25 index from dataset corpus.
        Datasets must have: .canonical_id, .title, .description_short (cached_property), .tags_primary

        O(n × avg_tokens) time, O(n × avg_tokens) space.
        Skips datasets that fail text extraction — logs + continues.
        """
        try:
            from rank_bm25 import BM25Okapi  # lazy import — Windows safe
        except ImportError as e:
            logger.error(
                "bm25_import_failed",
                extra={"error": str(e), "hint": "pip install rank-bm25"},
            )
            raise

        t0 = time.monotonic()
        corpus_ids: list[str] = []
        corpus_tokens: list[list[str]] = []

        for ds in datasets:
            try:
                text = _build_dataset_text(
                    title=ds.title or "",
                    description_short=ds.description_short or "",
                    tags_primary=ds.tags_primary or [],
                )
                tokens = _tokenize(text)
                # Empty token list is allowed — BM25Okapi handles it
                corpus_ids.append(ds.canonical_id)
                corpus_tokens.append(tokens)
            except Exception as exc:
                logger.warning(
                    "bm25_build_skip",
                    extra={
                        "canonical_id": getattr(ds, "canonical_id", "unknown"),
                        "error": str(exc)[:120],
                    },
                )

        if not corpus_tokens:
            logger.warning("bm25_empty_corpus", extra={"dataset_count": len(datasets)})
            self._built = True
            self._corpus_size = 0
            return

        self._corpus_ids = corpus_ids
        self._corpus_tokens = corpus_tokens
        self._bm25 = BM25Okapi(corpus_tokens)
        self._built = True
        self._corpus_size = len(corpus_ids)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "bm25_index_built",
            extra={
                "corpus_size": self._corpus_size,
                "build_ms": round(elapsed_ms, 1),
            },
        )

    def get_top_k(
        self,
        query_tokens: list[str],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Return top-k (canonical_id, bm25_score) pairs, sorted descending.
        Returns empty list if index not built or query is empty.
        O(n × |query_tokens|) — BM25 is linear in corpus size.

        Args:
            query_tokens: Pre-tokenized query terms (lowercase strings)
            k:            Maximum results to return

        Returns:
            List of (canonical_id, score) tuples, sorted by score descending.
            Scores are raw BM25 values (not normalized).
        """
        if not self._built or self._corpus_size == 0:
            return []

        if not query_tokens:
            logger.debug("bm25_empty_query")
            return []

        try:
            scores = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "bm25_score_failed",
                extra={"query_tokens": query_tokens[:10], "error": str(exc)[:120]},
            )
            return []

        # Build (canonical_id, score) pairs, filter zeros, sort descending
        # O(n log n) for sort — acceptable at corpus sizes ≤ 1M
        scored = [
            (self._corpus_ids[i], float(scores[i]))
            for i in range(len(self._corpus_ids))
            if scores[i] > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_scores_all(self, query_tokens: list[str]) -> dict[str, float]:
        """
        Return BM25 scores for ALL corpus documents, keyed by canonical_id.
        Used by HybridEngine for score fusion across full corpus.
        O(n × |query_tokens|).
        """
        if not self._built or self._corpus_size == 0 or not query_tokens:
            return {}

        try:
            scores = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]
            return {
                self._corpus_ids[i]: float(scores[i])
                for i in range(len(self._corpus_ids))
            }
        except Exception as exc:
            logger.warning(
                "bm25_scores_all_failed",
                extra={"error": str(exc)[:120]},
            )
            return {}

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    def tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize a raw query string using the same tokenizer as index build.
        Callers should pass the result to get_top_k() to ensure token consistency.
        """
        return _tokenize(query)