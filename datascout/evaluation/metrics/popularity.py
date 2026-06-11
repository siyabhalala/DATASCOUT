"""
datascout.evaluation.metrics.popularity
────────────────────────────────────────
Popularity evaluator: community trust and adoption scoring.

Evaluates:
  - Download count (log-scaled — prevents viral outlier dominance)
  - Upvote/like count (log-scaled)
  - Community engagement proxy
  - Source-specific signals (Kaggle usability_rating, HF likes)

Returns a PopularityResult with score + breakdown.
Deterministic — no LLM involvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from datascout.contracts import RawDataset

# Log normalization references:
#   100K downloads → ~1.0 (widely adopted)
#   1K downloads   → ~0.5 (moderate)
#   10 downloads   → ~0.25 (niche)
_DOWNLOAD_LOG_NORM = math.log(100_001)
_UPVOTE_LOG_NORM   = math.log(1_001)

# Weights for combining signals
_DOWNLOAD_WEIGHT = 0.70
_UPVOTE_WEIGHT   = 0.30

# Usability rating from Kaggle (0–1 scale stored in extra)
_USABILITY_BONUS_WEIGHT = 0.05


@dataclass
class PopularityResult:
    score: float                   # 0.0–1.0 composite
    download_score: float          # Log-scaled download signal
    upvote_score: float            # Log-scaled upvote signal
    usability_bonus: float         # Source-specific bonus
    download_count: Optional[int]
    upvote_count: Optional[int]
    explanation: str
    has_popularity_data: bool


def score_popularity(dataset: RawDataset) -> PopularityResult:
    """
    Score community adoption deterministically.

    Algorithm:
      - log(downloads+1) / log(100_001) → [0, 1]
      - log(upvotes+1)  / log(1_001)   → [0, 1]
      - Weighted combination: 70% downloads + 30% upvotes
      - Source-specific usability bonus (Kaggle only)

    No popularity data → 0.3 (neutral-low, not zero)
    Never raises.
    """
    downloads = dataset.download_count
    upvotes = dataset.upvote_count

    if (downloads is None or downloads == 0) and (upvotes is None or upvotes == 0):
        return PopularityResult(
            score=0.3,
            download_score=0.0,
            upvote_score=0.0,
            usability_bonus=0.0,
            download_count=None,
            upvote_count=None,
            explanation="No popularity data available — neutral-low score applied.",
            has_popularity_data=False,
        )

    download_score = 0.0
    if downloads is not None and downloads > 0:
        download_score = round(math.log(downloads + 1) / _DOWNLOAD_LOG_NORM, 4)
        download_score = min(download_score, 1.0)

    upvote_score = 0.0
    if upvotes is not None and upvotes > 0:
        upvote_score = round(math.log(upvotes + 1) / _UPVOTE_LOG_NORM, 4)
        upvote_score = min(upvote_score, 1.0)

    combined = download_score * _DOWNLOAD_WEIGHT + upvote_score * _UPVOTE_WEIGHT

    # Source-specific usability bonus (Kaggle stores usability_rating 0–1)
    usability_bonus = 0.0
    usability = dataset.extra.get("usability_rating") if dataset.extra else None
    if usability is not None:
        try:
            usability_bonus = round(float(usability) * _USABILITY_BONUS_WEIGHT, 4)
        except (ValueError, TypeError):
            pass

    score = round(min(combined + usability_bonus, 1.0), 4)

    # Explanation
    dl_str = f"{downloads:,}" if downloads is not None else "unknown"
    up_str = f"{upvotes:,}" if upvotes is not None else "unknown"

    if score >= 0.8:
        tier = "extremely popular"
    elif score >= 0.6:
        tier = "widely adopted"
    elif score >= 0.4:
        tier = "moderately popular"
    elif score >= 0.2:
        tier = "niche community"
    else:
        tier = "limited adoption"

    explanation = (
        f"Dataset is {tier} — "
        f"{dl_str} downloads, {up_str} upvotes."
    )

    return PopularityResult(
        score=score,
        download_score=download_score,
        upvote_score=upvote_score,
        usability_bonus=usability_bonus,
        download_count=downloads,
        upvote_count=upvotes,
        explanation=explanation,
        has_popularity_data=True,
    )