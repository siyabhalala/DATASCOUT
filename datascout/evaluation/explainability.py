"""
datascout.evaluation.explainability
─────────────────────────────────────
Human-readable ranking explanations for evaluated datasets.

``build_ranking_explanation()`` is the single public entry point.  It
accepts a ``ScoredDataset``, optional ``DatasetDiagnostics``, and an
optional ``analysis`` dict (from ``AnalysisEngine.analyze()``) and returns a
``RankingExplanation`` containing every field the API response exposes:

  - ``why_selected``  — one-paragraph justification of the rank position
  - ``strengths``     — positive signals, including actual-data quality checks
  - ``weaknesses``    — concerns, including AnalysisEngine suggested_fixes
  - ``bias_warnings`` — bias / risk flags (geographic, class imbalance, etc.)
  - ``score_labels``  — human-readable label for each score dimension
  - ``quality_tier``  — "excellent" | "good" | "fair" | "poor" | "incomplete"

SIGNAL HIERARCHY (most-specific wins):
  1. AnalysisEngine deep analysis — sampled-data signals (completeness_score,
     balance_score, uniqueness_score, suggested_fixes).  Most trustworthy
     because they are computed from actual downloaded data rows.
  2. EvaluatorPipeline diagnostics — metadata-derived, no data download.
  3. RawDataset metadata fields — fallback when nothing else is available.

All logic is deterministic (no LLM calls).  Results are stable across runs
for the same input, safe to cache, and fast (<1 ms per dataset).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Contracts ─────────────────────────────────────────────────────────────────
from datascout.evaluation.scorer import ScoredDataset
from datascout.evaluation.pipeline import DatasetDiagnostics

from datascout.contracts.states import compute_quality_tier

logger = logging.getLogger(__name__)

__all__ = [
    "RankingExplanation",
    "build_ranking_explanation",
]

# ── Score-label thresholds ────────────────────────────────────────────────────
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (0.85, "Excellent"),
    (0.70, "Good"),
    (0.50, "Fair"),
    (0.30, "Weak"),
    (0.00, "Poor"),
]

# Dimension display names
_DIM_NAMES: Dict[str, str] = {
    "task_relevance":    "Task Relevance",
    "quality":           "Data Quality",
    "popularity":        "Popularity",
    "freshness":         "Freshness",
    "description_match": "Description Match",
}


# ── Analysis-dict safe accessor ───────────────────────────────────────────────

def _aq(analysis: Optional[dict], key: str, default=None):
    """Safe read of ``analysis["quality"][key]``.

    The ``analysis`` dict passed through the pipeline is:
    ``{"quality": {...}, "summary": {...}, "target_column": {...}}``
    where the ``quality`` sub-dict holds all AnalysisEngine results.
    Returns *default* when *analysis* is None, the key is absent, or
    the value is None.
    """
    if not analysis:
        return default
    val = analysis.get("quality", {}).get(key)
    return default if val is None else val



@dataclass
class RankingExplanation:
    """The complete explanation payload for one ranked dataset.

    Attributes
    ----------
    why_selected:
        One paragraph (2–4 sentences) explaining *why* this dataset
        occupies its rank position.  Written for a non-technical ML
        practitioner.
    strengths:
        Positive signals that drove this dataset up the rankings.
    weaknesses:
        Quality concerns or gaps that may limit usefulness.
    bias_warnings:
        Dataset-level bias / fairness / provenance risks.  Empty list
        if none detected.
    score_labels:
        Mapping of dimension name → human-readable label string, e.g.
        ``{"task_relevance": "Excellent (92%)"}``.
    quality_tier:
        Single-word tier from ``compute_quality_tier()``.
    """

    why_selected: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    bias_warnings: List[str] = field(default_factory=list)
    score_labels: Dict[str, str] = field(default_factory=dict)
    quality_tier: str = "incomplete"


# ── Public API ────────────────────────────────────────────────────────────────

def build_ranking_explanation(
    scored: ScoredDataset,
    diagnostics: Optional[DatasetDiagnostics],
    rank: int,
    total_candidates: int = 0,
    analysis: Optional[dict] = None,
) -> RankingExplanation:
    """Build a human-readable ``RankingExplanation`` for *scored*.

    Parameters
    ----------
    scored:
        The evaluated dataset.  Must have ``.dataset`` (RawDataset) and
        ``.breakdown`` (ScoreBreakdown).
    diagnostics:
        Optional per-dimension diagnostics from ``EvaluatorPipeline``.
        When present, enables richer freshness / popularity / bias signals.
    rank:
        1-based rank position in the final result list.
    total_candidates:
        Total number of datasets evaluated before ranking.  Used in the
        ``why_selected`` prose.
    analysis:
        Optional analysis dict from ``AnalysisEngine.analyze()`` (as stored
        in ``analysis_map`` inside ``search_v2._run_pipeline()``).  When
        present, deep data-quality signals — ``completeness_score``,
        ``balance_score``, ``uniqueness_score``, ``deep_quality_score``, and
        ``suggested_fixes`` — are incorporated into strengths, weaknesses,
        bias warnings, and the why_selected narrative.

    Returns
    -------
    RankingExplanation:
        Fully populated explanation.  Never raises — returns a minimal
        explanation on any internal error.
    """
    try:
        return _build(scored, diagnostics, rank, total_candidates, analysis)
    except Exception as exc:
        logger.warning(
            "build_ranking_explanation_failed",
            extra={
                "canonical_id": getattr(
                    getattr(scored, "dataset", scored), "canonical_id", "?"
                ),
                "error": str(exc),
            },
        )
        score = getattr(scored, "composite_score", 0.0) or 0.0
        return RankingExplanation(
            why_selected=f"Ranked #{rank} with a composite score of {round(score * 100, 1)}%.",
            quality_tier="incomplete",
        )


# ── Private implementation ────────────────────────────────────────────────────

def _build(
    scored: ScoredDataset,
    diagnostics: Optional[DatasetDiagnostics],
    rank: int,
    total_candidates: int,
    analysis: Optional[dict] = None,
) -> RankingExplanation:
    """Core build logic, called inside try/except by the public function."""
    raw = scored.dataset
    bd = scored.breakdown
    score = scored.composite_score

    # ── Quality tier ──────────────────────────────────────────────────────────
    tier = _safe_compute_quality_tier(score)

    # ── Score labels ──────────────────────────────────────────────────────────
    score_labels = _build_score_labels(bd)

    # ── Strengths ─────────────────────────────────────────────────────────────
    strengths = _collect_strengths(raw, bd, diagnostics, analysis)

    # ── Weaknesses ────────────────────────────────────────────────────────────
    weaknesses = _collect_weaknesses(raw, bd, diagnostics, analysis)

    # ── Bias warnings ─────────────────────────────────────────────────────────
    bias_warnings = _collect_bias_warnings(raw, diagnostics, analysis)

    # ── Why selected ─────────────────────────────────────────────────────────
    why = _build_why_selected(raw, bd, score, rank, total_candidates, tier, strengths, analysis)

    return RankingExplanation(
        why_selected=why,
        strengths=strengths,
        weaknesses=weaknesses,
        bias_warnings=bias_warnings,
        score_labels=score_labels,
        quality_tier=tier,
    )


def _safe_compute_quality_tier(composite: float) -> str:
    """Call ``compute_quality_tier`` with graceful fallback."""
    try:
        result = compute_quality_tier(composite)
        # compute_quality_tier may return an enum or string
        if hasattr(result, "value"):
            return str(result.value)
        return str(result)
    except Exception:
        # Manual fallback if contracts function unavailable
        if composite >= 0.80:
            return "excellent"
        if composite >= 0.65:
            return "good"
        if composite >= 0.45:
            return "fair"
        if composite >= 0.25:
            return "poor"
        return "incomplete"


def _build_score_labels(bd) -> Dict[str, str]:
    """Convert ScoreBreakdown floats to labelled strings.

    Example output::

        {
            "task_relevance":    "Excellent (88%)",
            "quality":           "Good (71%)",
            "popularity":        "Fair (55%)",
            "freshness":         "Poor (28%)",
            "description_match": "Good (70%)",
        }
    """
    labels: Dict[str, str] = {}
    for attr, display in _DIM_NAMES.items():
        val: float = float(getattr(bd, attr, 0.0) or 0.0)
        pct = round(val * 100)
        tier_label = _score_to_label(val)
        labels[attr] = f"{tier_label} ({pct}%)"
    return labels


def _score_to_label(score: float) -> str:
    """Map a [0,1] score to a human-readable tier label."""
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Poor"


def _collect_strengths(raw, bd, diagnostics: Optional[DatasetDiagnostics], analysis: Optional[dict] = None) -> List[str]:
    """Identify positive signals worth surfacing to the user.

    Priority order:
      1. AnalysisEngine deep signals (sampled actual data — most specific)
      2. EvaluatorPipeline diagnostics (metadata-derived)
      3. RawDataset metadata fields (fallback)
    """
    strengths: List[str] = []

    # ── Deep analysis signals (AnalysisEngine) ────────────────────────────────
    deep_score  = _aq(analysis, "deep_quality_score")   # 0-100
    comp_score  = _aq(analysis, "completeness_score")   # 0-100, missing-value based
    bal_score   = _aq(analysis, "balance_score")        # 0-100, class imbalance
    uniq_score  = _aq(analysis, "uniqueness_score")     # 0-100, duplicate rows
    is_partial  = _aq(analysis, "analysis_partial", True)

    if not is_partial:
        # Only surface deep signals when data was actually downloaded
        if comp_score is not None and comp_score >= 90:
            strengths.append(
                f"Sampled data: high completeness — only "
                f"{round(100 - comp_score, 1)}% missing values across all columns."
            )
        if uniq_score is not None and uniq_score >= 98:
            strengths.append("Sampled data: no significant duplicate rows found.")
        if bal_score is not None and bal_score >= 75:
            strengths.append(
                f"Sampled data: well-balanced class distribution "
                f"(balance score {round(bal_score)}/100)."
            )
        if deep_score is not None and deep_score >= 75:
            strengths.append(
                f"Deep data quality score: {round(deep_score)}/100 "
                f"(completeness, balance, uniqueness combined)."
            )

    # ── EvaluatorPipeline signals ─────────────────────────────────────────────

    # Task relevance
    tr = float(getattr(bd, "task_relevance", 0.0) or 0.0)
    if tr >= 0.80:
        strengths.append("Highly relevant to the requested task type.")
    elif tr >= 0.65:
        strengths.append("Good task-type alignment with your query.")

    # Description quality
    dm = float(getattr(bd, "description_match", 0.0) or 0.0)
    if dm >= 0.75:
        strengths.append("Well-described with rich metadata.")

    # Popularity
    pop = float(getattr(bd, "popularity", 0.0) or 0.0)
    if pop >= 0.70:
        dl = getattr(raw, "download_count", None)
        if dl:
            strengths.append(f"Widely used — {dl:,} downloads.")
        else:
            strengths.append("Highly popular in the community.")

    # Freshness
    fresh = float(getattr(bd, "freshness", 0.0) or 0.0)
    if fresh >= 0.75:
        strengths.append("Recently updated — data is current.")

    # Quality metadata score
    qual = float(getattr(bd, "quality", 0.0) or 0.0)
    if qual >= 0.75 and deep_score is None:
        # Only surface the metadata-derived quality signal if deep analysis
        # didn't run (avoid repeating quality info twice)
        strengths.append("High data quality score.")

    # ── RawDataset metadata fallbacks ─────────────────────────────────────────

    col_names = getattr(raw, "column_names", None)
    if col_names and len(col_names) > 0:
        strengths.append(f"Full schema available ({len(col_names)} columns documented).")

    lic = getattr(raw, "license_type", None)
    if lic is not None:
        lic_str = lic.value if hasattr(lic, "value") else str(lic)
        strengths.append(f"License clearly specified ({lic_str}).")

    row_count = getattr(raw, "row_count", None)
    if row_count and row_count >= 10_000:
        strengths.append(f"Substantial size — {row_count:,} rows.")

    # Diagnostics extras
    if diagnostics:
        task_match = getattr(diagnostics, "task_match", None)
        if task_match and getattr(task_match, "match_type", "") == "exact":
            strengths.append("Exact task-type match for your query.")

    return strengths[:7]  # cap at 7 (was 6; deep analysis may add 1-2 extra)


def _collect_weaknesses(raw, bd, diagnostics: Optional[DatasetDiagnostics], analysis: Optional[dict] = None) -> List[str]:
    """Identify quality concerns worth surfacing to the user.

    When AnalysisEngine results are available, ``suggested_fixes`` from the
    engine become the primary source of weaknesses — they are computed from
    actual sampled data so they are more specific and actionable than
    metadata-derived signals.  Metadata-derived weaknesses are still included
    when deep analysis didn't run or didn't cover a particular dimension.
    """
    weaknesses: List[str] = []

    # ── Deep analysis signals (AnalysisEngine) — most specific ───────────────
    suggested_fixes: List[str] = _aq(analysis, "suggested_fixes") or []
    comp_score  = _aq(analysis, "completeness_score")  # 0-100
    bal_score   = _aq(analysis, "balance_score")       # 0-100
    uniq_score  = _aq(analysis, "uniqueness_score")    # 0-100
    is_partial  = _aq(analysis, "analysis_partial", True)
    deep_score  = _aq(analysis, "deep_quality_score")

    # Surface the top suggested fixes directly (they're already actionable sentences)
    for fix in suggested_fixes[:4]:
        weaknesses.append(fix)

    # Add specific score-based warnings for scores that didn't produce a fix
    if not is_partial:
        if comp_score is not None and comp_score < 70 and not any("missing" in f.lower() for f in suggested_fixes):
            weaknesses.append(
                f"Sampled data: high missing values — only {round(comp_score)}% of "
                f"cells are populated. Imputation or column removal may be needed."
            )
        if bal_score is not None and bal_score < 50 and not any("balance" in f.lower() or "samples" in f.lower() for f in suggested_fixes):
            weaknesses.append(
                f"Sampled data: class imbalance detected "
                f"(balance score {round(bal_score)}/100). Consider resampling strategies."
            )
        if uniq_score is not None and uniq_score < 95 and not any("duplicate" in f.lower() for f in suggested_fixes):
            dup_pct = round(100 - uniq_score, 1)
            weaknesses.append(
                f"Sampled data: ~{dup_pct}% duplicate rows found. "
                f"De-duplication recommended before training."
            )

    if is_partial and analysis is not None:
        # Analysis ran but couldn't download sample data — say so clearly
        err = _aq(analysis, "analysis_error")
        if err:
            weaknesses.append(
                f"Deep analysis was limited (could not download sample data: {err[:80]}). "
                f"Quality scores above are metadata-only estimates."
            )
        else:
            weaknesses.append(
                "Deep data analysis was limited — quality scores are metadata-only estimates."
            )

    # ── EvaluatorPipeline + metadata signals (fallbacks) ─────────────────────
    # Only add these when deep analysis didn't already cover the dimension

    # Low freshness (metadata signal — analysis engine doesn't check dates)
    fresh = float(getattr(bd, "freshness", 0.0) or 0.0)
    if fresh < 0.35:
        if diagnostics:
            days = getattr(getattr(diagnostics, "freshness", None), "days_since_update", None)
            if days and days > 365:
                weaknesses.append(f"Not updated in over {days // 365} year(s) — may be outdated.")
            else:
                weaknesses.append("Dataset has not been updated recently.")
        else:
            weaknesses.append("Dataset may be outdated.")

    # Low metadata quality score (only if deep analysis didn't run)
    qual = float(getattr(bd, "quality", 0.0) or 0.0)
    if qual < 0.40 and deep_score is None:
        if diagnostics:
            missing = getattr(getattr(diagnostics, "quality", None), "missing_fields", [])
            if missing:
                weaknesses.append(f"Missing metadata fields: {', '.join(missing[:3])}.")
            else:
                weaknesses.append("Low data quality score — metadata is incomplete.")
        else:
            weaknesses.append("Low data quality score.")

    # No license (metadata — analysis engine doesn't check this)
    lic = getattr(raw, "license_type", None)
    if lic is None and not any("license" in f.lower() for f in suggested_fixes):
        weaknesses.append("License not specified — verify usage rights before use.")

    # No description
    desc = (getattr(raw, "description", "") or "").strip()
    if len(desc) < 20 and not any("description" in f.lower() for f in suggested_fixes):
        weaknesses.append("Minimal description provided.")

    # No column schema
    col_names = getattr(raw, "column_names", None)
    if not col_names and not any("schema" in f.lower() or "column" in f.lower() for f in suggested_fixes):
        weaknesses.append("Column schema not documented.")

    # Small dataset
    row_count = getattr(raw, "row_count", None)
    if row_count is not None and row_count < 500:
        weaknesses.append(f"Small dataset ({row_count:,} rows) — may not generalise well.")

    # Low task relevance
    tr = float(getattr(bd, "task_relevance", 0.0) or 0.0)
    if tr < 0.40:
        weaknesses.append("Limited alignment with the requested task type.")

    return weaknesses[:6]  # cap at 6


def _collect_bias_warnings(raw, diagnostics: Optional[DatasetDiagnostics], analysis: Optional[dict] = None) -> List[str]:
    """Collect bias and risk signals from analysis, diagnostics, and raw metadata."""
    warnings: List[str] = []

    # ── Deep analysis bias signals ────────────────────────────────────────────
    # 1. Geographic bias — detected by AnalysisEngine from metadata text
    #    The suggested_fixes list contains the geographic warning when present.
    geo_fix = next(
        (f for f in (_aq(analysis, "suggested_fixes") or [])
         if "region" in f.lower() or "geographic" in f.lower() or "geography" in f.lower()),
        None,
    )
    if geo_fix:
        warnings.append(geo_fix)

    # 2. Class imbalance detected via actual data (not tags)
    bal_score = _aq(analysis, "balance_score")
    is_partial = _aq(analysis, "analysis_partial", True)
    if not is_partial and bal_score is not None and bal_score < 40:
        warnings.append(
            f"Class imbalance confirmed in sampled data "
            f"(balance score {round(bal_score)}/100) — "
            f"training on this data as-is risks biased predictions."
        )

    # ── EvaluatorPipeline diagnostics signals ─────────────────────────────────
    if diagnostics:
        quality_diag = getattr(diagnostics, "quality", None)
        bias_signals = getattr(quality_diag, "bias_signals", []) or []
        for signal in bias_signals:
            msg = str(signal)
            # Deduplicate: skip if we already surfaced the same concern
            if not any(msg[:40] in w for w in warnings):
                warnings.append(msg)

    # ── RawDataset tag-based heuristics ───────────────────────────────────────
    tags = [str(t).lower() for t in (getattr(raw, "tags", []) or [])]

    if "synthetic" in tags:
        warnings.append(
            "Synthetic dataset — results may not reflect real-world distributions."
        )
    if "imbalanced" in tags or "class imbalance" in tags:
        # Only add if we didn't already surface via actual-data analysis
        if bal_score is None or is_partial:
            warnings.append(
                "Dataset may have class imbalance — check target distribution before training."
            )
    if "demographic" in tags or "sensitive" in tags:
        warnings.append(
            "Contains potentially sensitive demographic features — review for fairness."
        )

    return warnings[:4]


def _build_why_selected(
    raw,
    bd,
    score: float,
    rank: int,
    total_candidates: int,
    tier: str,
    strengths: List[str],
    analysis: Optional[dict] = None,
) -> str:
    """Compose the why_selected paragraph.

    Aims for 2–4 sentences that a non-technical ML practitioner can read
    in under 10 seconds.  When AnalysisEngine results are available a
    data-analysis sentence is inserted to ground the explanation in facts
    from the actual dataset (not just metadata estimates).
    """
    title = getattr(raw, "title", "This dataset") or "This dataset"
    source = (getattr(raw, "source", "") or "").title()
    score_pct = round(score * 100, 1)

    # Opening sentence
    rank_suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
    candidate_clause = (
        f" out of {total_candidates:,} candidates" if total_candidates > 1 else ""
    )
    opening = (
        f"**{title}** ranks {rank}{rank_suffix}{candidate_clause} "
        f"with a {tier} composite score of {score_pct}%."
    )

    # Driving factors sentence
    top_dims = _top_dimensions(bd, n=2)
    if top_dims:
        factors = " and ".join(top_dims)
        driver_sentence = f"Its strongest signals are {factors}."
    else:
        driver_sentence = ""

    # ── Data-analysis sentence (new — inserted when AnalysisEngine ran) ───────
    data_sentence = ""
    deep_score = _aq(analysis, "deep_quality_score")
    comp_score  = _aq(analysis, "completeness_score")
    uniq_score  = _aq(analysis, "uniqueness_score")
    bal_score   = _aq(analysis, "balance_score")
    is_partial  = _aq(analysis, "analysis_partial", True)

    if not is_partial and any(v is not None for v in (deep_score, comp_score, uniq_score)):
        # Build a compact, factual summary of what analysis found
        facts: List[str] = []
        if comp_score is not None:
            facts.append(f"{round(comp_score)}% data completeness")
        if uniq_score is not None and uniq_score < 99:
            dup_pct = round(100 - uniq_score, 1)
            facts.append(f"{dup_pct}% duplicate rows")
        elif uniq_score is not None:
            facts.append("no duplicate rows")
        if bal_score is not None and bal_score < 60:
            facts.append(f"class imbalance detected (score {round(bal_score)}/100)")
        elif bal_score is not None:
            facts.append("balanced classes")
        if deep_score is not None:
            facts.append(f"overall quality {round(deep_score)}/100")

        if facts:
            data_sentence = (
                f"Sampled data analysis found: {', '.join(facts)}."
            )

    # Source + context sentence
    source_clause = f" (from {source})" if source else ""
    row_count = getattr(raw, "row_count", None)
    size_clause = f" with {row_count:,} rows" if row_count else ""
    context = (
        f"The dataset{source_clause}{size_clause} was selected "
        f"based on relevance, quality, and recency."
    )

    # Optionally note a top strength (skip if it repeats the data sentence)
    strength_sentence = strengths[0] if strengths else ""
    if data_sentence and strength_sentence and (
        "sampled data" in strength_sentence.lower()
        or "completeness" in strength_sentence.lower()
    ):
        strength_sentence = ""  # already covered by data_sentence

    parts = [opening, driver_sentence, data_sentence, context]
    if strength_sentence and strength_sentence not in context:
        parts.append(strength_sentence)

    return " ".join(p for p in parts if p).strip()


def _top_dimensions(bd, n: int = 2) -> List[str]:
    """Return the names of the top-n scoring dimensions."""
    dims = []
    for attr, display in _DIM_NAMES.items():
        val = float(getattr(bd, attr, 0.0) or 0.0)
        dims.append((val, display))
    dims.sort(key=lambda x: -x[0])
    return [f"{name.lower()} ({round(val*100)}%)" for val, name in dims[:n] if val > 0]