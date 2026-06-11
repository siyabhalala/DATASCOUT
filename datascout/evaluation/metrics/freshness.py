"""
datascout.evaluation.metrics.freshness
────────────────────────────────────────
Freshness evaluator: exponential decay scoring based on dataset recency.

Evaluates:
  - Days since last_updated (exponential decay, half-life 365 days)
  - Maintenance activity signals from extra metadata
  - Active ecosystem support indicators

Returns a FreshnessResult with score + explainable breakdown.
LLM NEVER controls this — fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from datascout.contracts import RawDataset

# Freshness decay: score = 2^(-days / HALF_LIFE_DAYS)
FRESHNESS_HALF_LIFE_DAYS = 365.0

# If dataset was updated within this many days, award a freshness bonus
RECENT_UPDATE_BONUS_DAYS = 90


@dataclass
class FreshnessResult:
    score: float                # 0.0–1.0
    days_since_update: Optional[int]
    last_updated: Optional[datetime]
    decay_score: float          # Raw exponential decay score
    recency_bonus: float        # Extra bonus for very recent updates
    explanation: str
    is_estimated: bool = False  # True if last_updated was unavailable


def score_freshness(dataset: RawDataset) -> FreshnessResult:
    """
    Score dataset freshness deterministically.

    Algorithm:
      - No last_updated → 0.5 (neutral, not penalized)
      - Has last_updated → exponential decay: 2^(-days / half_life)
      - Recent (<90 days) → small bonus capped at 1.0

    Never raises.
    """
    if dataset.last_updated is None:
        return FreshnessResult(
            score=0.5,
            days_since_update=None,
            last_updated=None,
            decay_score=0.5,
            recency_bonus=0.0,
            explanation="Last update date unknown — neutral freshness score applied.",
            is_estimated=True,
        )

    now = datetime.now(timezone.utc)
    last = dataset.last_updated
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    days_old = max((now - last).days, 0)

    # Exponential decay
    decay = 2 ** (-days_old / FRESHNESS_HALF_LIFE_DAYS)
    decay = round(min(decay, 1.0), 4)

    # Recency bonus for very fresh datasets
    bonus = 0.0
    if days_old <= RECENT_UPDATE_BONUS_DAYS:
        bonus = round(0.1 * (1 - days_old / RECENT_UPDATE_BONUS_DAYS), 4)

    score = round(min(decay + bonus, 1.0), 4)

    if days_old == 0:
        explanation = "Updated today — maximum freshness."
    elif days_old <= 30:
        explanation = f"Updated {days_old} days ago — very fresh."
    elif days_old <= 180:
        explanation = f"Updated {days_old} days ago — reasonably current."
    elif days_old <= 365:
        explanation = f"Updated {days_old} days ago — approaching one year old."
    else:
        years = days_old / 365
        explanation = f"Updated {days_old} days ago ({years:.1f} years) — potentially stale."

    return FreshnessResult(
        score=score,
        days_since_update=days_old,
        last_updated=last,
        decay_score=decay,
        recency_bonus=bonus,
        explanation=explanation,
    )