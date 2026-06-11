"""
datascout.evaluation.scorer
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Multi-dimensional deterministic scoring.
Produces a composite_score (0-1) for each dataset from five dimensions.
LLM NEVER performs scoring — all logic is deterministic and auditable.

SYSTEM DESIGN DECISIONS:

  1. WHY five scoring dimensions?
     - task_relevance:    Does the dataset match what the user needs to DO?
     - quality:           Is the dataset clean, complete, documented?
     - popularity:        Is it widely used/trusted by the community?
     - freshness:         Is the data recent enough to be useful?
     - description_match: Does the content align with the user's keywords?
     Each captures a distinct signal. Any single dimension is misleading alone.

  2. WHY task_relevance is the highest-weight dimension (default 0.35)?
     - A perfectly clean, popular, fresh dataset is worthless if it's for
       the wrong task type. Task alignment is the primary decision criterion.
     - All other dimensions are secondary quality signals.

  3. WHY description_match uses TF-IDF-style overlap (not embeddings)?
     - Embeddings require a model loaded in memory — adds 200MB+ overhead
     - TF-IDF overlap is O(keywords × tokens) — sub-millisecond per dataset
     - At Phase 9, pure keyword overlap is sufficient for Level-0 agent
     - Phase 10+ can replace this with FAISS semantic search
     - The scorer interface stays the same — only the implementation changes

  4. WHY ScoreBreakdown as a typed dataclass (not dict)?
     - Dict: no type safety, silent typos, no IDE autocomplete
     - Dataclass: typed, validated, serializable, auditable
     - DecisionTrace in responses.py references score_breakdown by field name
     - Changing a dimension name is a compile-time error, not a runtime surprise

  5. WHY normalize_weights() called on every score() call?
     - Prevents division by zero if caller sets all weights to 0
     - Allows partial weight specification — "only task and quality matter"
       by setting other weights to 0, system auto-normalizes to 1.0 total

FAILURE SCENARIOS HANDLED:
  - All weight dimensions = 0 → equal-weight fallback (not ZeroDivisionError)
  - Missing download_count → popularity_score uses fallback formula
  - last_updated = None → freshness_score = 0.5 (neutral, not 0)
  - Empty keywords → description_match = 0.5 (neutral)
  - task_types empty → task_relevance scores by modality compatibility only

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from datascout.contracts import RawDataset
from datascout.contracts.task_types import Modality, TaskType, compute_task_compatibility
from datascout.query_understanding.task_types import are_in_same_family

logger = logging.getLogger("datascout.evaluation.scorer")

# Default scoring weights — must sum to 1.0
# FIX (v3.5.0): description_match raised from 0.00 → 0.20 (it was gate-only before).
# Root cause of query mismatch: ANY image-classification dataset ranked equally because
# content relevance had zero weight. Cards/cats/weather scored same as pothole datasets.
# New weights balance: task fit (0.35) + content relevance (0.20) + quality (0.20) +
# popularity (0.15) + freshness (0.10). Query-relevant datasets now rank above generic ones.
DEFAULT_WEIGHTS = {
    "task_relevance":     0.35,   # FIX: reduced from 0.45 — task match alone shouldn't dominate
    "quality":            0.20,
    "popularity":         0.15,
    "freshness":          0.10,
    "description_match":  0.20,   # FIX: raised from 0.00 — content relevance MUST count in ranking
}                                  # Cards/cats/weather no longer outrank pothole datasets

# Semantic relevance gate — MUST pass before any scoring happens.
# Uses soft semantic matching (substring + synonym aware) not exact keyword counting.
# FIX: Raised from 0.10 → 0.15 — too low allows completely off-topic datasets.
# With description_match now having 0.20 weight, a low gate still passes noise
# that the scorer would push down anyway. 0.15 = at least 1 core concept matched.
DESCRIPTION_MATCH_GATE: float = 0.25  # FIX v3.6.0: raised from 0.15 → 0.25.
# 0.15 = 1 keyword out of 7 must match. With "image" + "classification" in the query,
# Cards/Cats/Flickr datasets all match ≥1 keyword and pass the gate.
# 0.25 = at least 2 domain-specific keywords must match — generic image datasets
# that only match "image" or "classification" but not "pothole"/"road"/"damage" get gated.

# Freshness decay: datasets older than this get score approaching 0
FRESHNESS_HALF_LIFE_DAYS = 365.0  # 1 year half-life


@dataclass
class ScoreBreakdown:
    """
    Per-dimension scores for one dataset.
    All values 0.0–1.0. None = dimension could not be computed.
    """
    task_relevance:    float
    quality:           float
    popularity:        float
    freshness:         float
    description_match: float
    composite:         float           # Weighted sum

    # Applied weights (may differ from defaults if auto-normalized)
    weights: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "task_relevance":     round(self.task_relevance,    3),
            "quality":            round(self.quality,           3),
            "popularity":         round(self.popularity,        3),
            "freshness":          round(self.freshness,         3),
            "description_match":  round(self.description_match, 3),
            "composite":          round(self.composite,         4),
            "weights":            {k: round(v, 3) for k, v in self.weights.items()},
        }


@dataclass
class ScoredDataset:
    """A RawDataset paired with its score breakdown."""
    dataset:   RawDataset
    breakdown: ScoreBreakdown

    @property
    def composite_score(self) -> float:
        return self.breakdown.composite

    def to_dict(self) -> dict:
        return {
            "canonical_id":   self.dataset.canonical_id,
            "title":          self.dataset.title,
            "source":         self.dataset.source,
            "composite_score": round(self.composite_score, 4),
            "breakdown":      self.breakdown.to_dict(),
        }


class DatasetScorer:
    """
    Scores each dataset across five deterministic dimensions.
    Returns ScoredDataset list — never raises.

    Usage:
        scorer = DatasetScorer(query_task=TaskType.REGRESSION, keywords=["price", "house"])
        scored = scorer.score_all(datasets)
        # sorted by scored[i].composite_score descending
    """

    def __init__(
        self,
        query_task:      Optional[TaskType]   = None,
        query_modality:  Optional[Modality]   = None,
        keywords:        Optional[list[str]]  = None,
        weights:         Optional[dict[str, float]] = None,
    ) -> None:
        self.query_task     = query_task
        self.query_modality = query_modality
        self.keywords       = [k.lower() for k in (keywords or [])]
        self.weights        = self._normalize_weights(weights or DEFAULT_WEIGHTS.copy())

    def score_all(self, datasets: list[RawDataset]) -> list[ScoredDataset]:
        """
        Score all datasets. Returns list sorted by composite_score descending.

        Hard gate: datasets whose description_match falls below
        DESCRIPTION_MATCH_GATE are excluded entirely (appended at the very
        bottom with composite=0.0) so they never surface in top results.
        This prevents popular-but-irrelevant datasets (e.g. hotel reviews
        returned for a speech recognition query) from poisoning the ranking.

        Skips datasets that fail scoring (logs + continues) — never raises.
        """
        scored: list[ScoredDataset] = []
        gated_out: list[ScoredDataset] = []   # below query match threshold

        for ds in datasets:
            try:
                breakdown = self._score_one(ds)
                sd = ScoredDataset(dataset=ds, breakdown=breakdown)

                # Hard gate on query match — only applies when keywords exist
                if self.keywords and breakdown.description_match < DESCRIPTION_MATCH_GATE:
                    # Force composite to 0 so it sinks to the bottom
                    gated_breakdown = ScoreBreakdown(
                        task_relevance=breakdown.task_relevance,
                        quality=breakdown.quality,
                        popularity=breakdown.popularity,
                        freshness=breakdown.freshness,
                        description_match=breakdown.description_match,
                        composite=0.0,
                        weights=breakdown.weights,
                    )
                    gated_out.append(ScoredDataset(dataset=ds, breakdown=gated_breakdown))
                    logger.debug(
                        "dataset_gated_out",
                        extra={
                            "canonical_id": ds.canonical_id,
                            "description_match": round(breakdown.description_match, 3),
                            "gate": DESCRIPTION_MATCH_GATE,
                        },
                    )
                else:
                    scored.append(sd)

            except Exception as e:
                logger.warning(
                    "score_failed",
                    extra={"canonical_id": ds.canonical_id, "error": str(e)[:80]},
                )
                gated_out.append(ScoredDataset(
                    dataset=ds,
                    breakdown=ScoreBreakdown(
                        task_relevance=0.0, quality=0.0, popularity=0.0,
                        freshness=0.5, description_match=0.0,
                        composite=0.0, weights=self.weights,
                    ),
                ))

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        # Gated-out datasets appended at the bottom — visible in API response
        # for transparency but never surfaced as top results
        return scored + gated_out

    def _score_one(self, ds: RawDataset) -> ScoreBreakdown:
        task  = self._score_task_relevance(ds)
        qual  = self._score_quality(ds)
        pop   = self._score_popularity(ds)
        fresh = self._score_freshness(ds)
        desc  = self._score_description_match(ds)

        w = self.weights
        composite = (
            task  * w["task_relevance"]    +
            qual  * w["quality"]           +
            pop   * w["popularity"]        +
            fresh * w["freshness"]         +
            desc  * w["description_match"]
        )

        # ── FIX v3.7.0: Modality mismatch penalty ────────────────────────────
        # ROOT CAUSE of Issue 4: query_modality was stored but never applied.
        # When a user queries "image datasets for detecting potholes", a CSV/GeoJSON
        # pothole dataset ranked equally to an image dataset because modality had
        # zero influence on composite score. This fix applies a strong penalty when
        # the query specifies a modality that the dataset clearly does NOT have.
        #
        # Penalty levels:
        #   - Wrong primary modality (query=image, dataset=tabular/text/audio): ×0.40
        #   - Unknown modality (dataset has no modality info): ×0.90 (mild doubt)
        #   - Correct modality or no query_modality: no penalty
        #
        # WHY 0.40 (not 0.0)?
        #   A tabular dataset CAN still be useful as ground-truth labels for images.
        #   Hard-zero would incorrectly eliminate legitimate mixed-use datasets.
        #   0.40 deprioritises them strongly without making them invisible.
        modality_factor = self._modality_penalty(ds)
        composite = composite * modality_factor

        composite = round(max(0.0, min(1.0, composite)), 4)

        return ScoreBreakdown(
            task_relevance=task,
            quality=qual,
            popularity=pop,
            freshness=fresh,
            description_match=desc,
            composite=composite,
            weights=self.weights,
        )

    def _modality_penalty(self, ds: RawDataset) -> float:
        """
        Returns a multiplicative penalty factor [0.40, 1.0] based on whether
        the dataset modality matches the query modality.

        Called only when self.query_modality is set (non-None).
        When query_modality is None (no modality specified), returns 1.0 always.
        """
        if not self.query_modality:
            return 1.0  # No modality constraint — no penalty

        ds_modalities = list(ds.modalities or [])
        if not ds_modalities:
            # Dataset has no modality info — mild uncertainty penalty
            return 0.90

        query_mod = self.query_modality

        # Direct match: dataset has the requested modality
        if query_mod in ds_modalities:
            return 1.0

        # Check if any dataset modality is "compatible" with query modality.
        # MULTIMODAL datasets (image + tabular) should not be penalised when
        # the query asks for image — they contain image data.
        try:
            if Modality.MULTIMODAL in ds_modalities:
                return 1.0
        except Exception:
            pass

        # Modality mismatch — apply strong penalty.
        # Examples of clear mismatches:
        #   query=image, dataset=[tabular]  → 0.40
        #   query=image, dataset=[text]     → 0.40
        #   query=audio, dataset=[tabular]  → 0.40
        #   query=tabular, dataset=[image]  → 0.40
        return 0.40

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSION SCORERS
    # ─────────────────────────────────────────────────────────────────────────

    def _score_task_relevance(self, ds: RawDataset) -> float:
        """
        How well does this dataset serve the queried task?

        Algorithm:
          - Exact task match in dataset.task_types → 1.0
          - Same task family match → 0.7
          - Modality-based compatibility → compatibility_score
          - No task info in query → neutral 0.5
          - Dataset has no task/modality info → 0.4 (slight penalty)
        """
        if not self.query_task or self.query_task == TaskType.OTHER:
            return 0.5  # No task specified — neutral

        # Check explicit task_types on the dataset
        if ds.task_types:
            if self.query_task in ds.task_types:
                return 1.0  # Exact match

            # Family match (e.g. CLASSIFICATION matches BINARY_CLASSIFICATION)
            for dt in ds.task_types:
                if are_in_same_family(self.query_task, dt):
                    return 0.7

        # Fallback: modality-based compatibility
        if ds.modalities:
            compat = compute_task_compatibility(list(ds.modalities), self.query_task)
            if compat.is_compatible:
                return compat.compatibility_score
            return 0.1  # Modality incompatible

        # No task or modality info at all
        return 0.4

    def _score_quality(self, ds: RawDataset) -> float:
        """
        Dataset quality score — structural completeness as primary signal.
        Falls back gracefully if Phase-9 analysis hasn't run yet.

        Components (all 0-1 range before weighting):
          - metadata_completeness (60%): documentation quality
          - has_description       (20%): has a useful description
          - has_schema_info       (10%): column/feature info available
          - has_license_info      (10%): license specified (legal clarity)
        """
        score = (
            ds.metadata_completeness * 0.60 +
            (0.20 if ds.has_description  else 0.0) +
            (0.10 if ds.has_schema_info  else 0.0) +
            (0.10 if ds.has_license_info else 0.0)
        )
        return round(min(score, 1.0), 4)

    def _score_popularity(self, ds: RawDataset) -> float:
        """
        Community engagement proxy — download count + upvotes.

        WHY log-scale (not raw count)?
        - Raw: dataset with 1M downloads scores 1000× a dataset with 1K downloads
        - log: 1M → ~1.0, 1K → ~0.5, 10 → ~0.25 — meaningful gradient
        - Prevents viral outliers from dominating the score

        Normalisation reference: 10K downloads ≈ 0.6 (widely used but not viral)
        """
        download_score = 0.0
        if ds.download_count and ds.download_count > 0:
            # log(downloads+1) / log(100_000+1) → caps at ~1.0 around 100K downloads
            download_score = math.log(ds.download_count + 1) / math.log(100_001)

        upvote_score = 0.0
        if ds.upvote_count and ds.upvote_count > 0:
            upvote_score = math.log(ds.upvote_count + 1) / math.log(1_001)

        if ds.download_count is None and ds.upvote_count is None:
            return 0.3  # No popularity data — neutral-low

        combined = download_score * 0.7 + upvote_score * 0.3
        return round(min(combined, 1.0), 4)

    def _score_freshness(self, ds: RawDataset) -> float:
        """
        Exponential decay based on days since last_updated.
        Half-life = 365 days (one year).
        last_updated=None → 0.5 (neutral — unknown is not stale).
        """
        if ds.last_updated is None:
            return 0.5

        now = datetime.now(timezone.utc)
        last = ds.last_updated
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        days_old = max((now - last).days, 0)
        # Exponential decay: score = 2^(-days / half_life)
        score = 2 ** (-days_old / FRESHNESS_HALF_LIFE_DAYS)
        return round(min(score, 1.0), 4)

    # Semantic synonym map — core concepts that mean the same thing.
    # FIX: Greatly expanded to cover agricultural, medical, and vision domains.
    # REASON: Kaggle dataset titles are domain-specific shorthand. Queries use
    # natural language. Without synonyms, "crop" never matches "paddy" or "maize",
    # "disease" never matches "blight" or "lesion", "photo" never matches "image".
    _SYNONYMS: dict[str, list[str]] = {
        # ── Agricultural domain ──────────────────────────────────────────────
        "crop":         ["plant", "agriculture", "farm", "agri", "vegetation", "botanical",
                         "paddy", "maize", "wheat", "rice", "corn", "soybean", "cotton",
                         "sugarcane", "potato", "tomato", "citrus", "mango", "coffee", "tea"],
        "plant":        ["crop", "vegetation", "leaf", "botanical", "flora", "foliage",
                         "seedling", "sapling", "shrub", "tree", "herb"],
        "leaf":         ["foliage", "plant", "crop", "vegetation", "frond", "blade",
                         "canopy", "greenery"],
        "agriculture":  ["farming", "agri", "crop", "rural", "field", "horticulture",
                         "plantation", "orchard", "greenhouse", "paddy"],
        "farms":        ["farming", "agriculture", "agri", "field", "rural", "paddy",
                         "plantation", "ranch", "orchard"],
        "pest":         ["insect", "bug", "infestation", "larvae", "aphid", "mite",
                         "caterpillar", "worm", "borer", "locust"],
        # ── Disease / medical domain ─────────────────────────────────────────
        "disease":      ["pathology", "infection", "disorder", "blight", "rot", "fungal",
                         "bacterial", "viral", "lesion", "spot", "mildew", "rust",
                         "canker", "wilt", "scab", "smut", "mosaic", "necrosis",
                         "symptoms", "diagnosis", "ailment"],
        "pathology":    ["disease", "infection", "disorder", "blight", "lesion", "diagnosis"],
        "medical":      ["clinical", "health", "patient", "hospital", "radiology",
                         "retinal", "skin", "cancer", "tumor", "mri", "xray", "ct"],
        "cancer":       ["tumor", "malignant", "carcinoma", "oncology", "melanoma",
                         "lymphoma", "biopsy"],
        "skin":         ["dermatology", "dermoscopy", "lesion", "rash", "acne",
                         "melanoma", "nevus"],
        # ── Vision / image domain ─────────────────────────────────────────────
        "image":        ["photo", "picture", "visual", "cv", "vision", "photograph",
                         "frame", "snapshot", "thumbnail", "pixel", "rgb"],
        "photo":        ["image", "picture", "photograph", "snapshot", "visual"],
        "vision":       ["image", "visual", "cv", "photo", "picture", "optical"],
        "camera":       ["image", "photo", "video", "capture", "snapshot"],
        # ── ML task domain ───────────────────────────────────────────────────
        "classification": ["classifier", "detection", "recognition", "labeling",
                           "categorization", "tagging", "sorting", "identification"],
        "detection":    ["classifier", "recognition", "identification", "detector",
                         "localization", "segmentation", "bounding"],
        "detector":     ["detection", "classifier", "classification", "recognition",
                         "identification", "model"],
        "recognition":  ["detection", "classification", "identification", "reading"],
        "segmentation": ["pixel", "mask", "semantic", "instance", "panoptic"],
        "prediction":   ["forecast", "estimation", "regression", "inference"],
        # ── NLP domain ──────────────────────────────────────────────────────
        "speech":       ["audio", "voice", "spoken", "asr", "transcription", "acoustic"],
        "sentiment":    ["opinion", "emotion", "review", "polarity", "attitude"],
        "text":         ["nlp", "language", "corpus", "document", "article", "tweet"],
        # ── Geographic/demographic ───────────────────────────────────────────
        "indian":       ["india", "indic", "desi", "subcontinent", "hindi", "gujarat", "bharat"],
        "chinese":      ["china", "mandarin", "chinese", "sino", "hong kong"],
        # ── Common query words that appear in Kaggle titles ──────────────────
        "dataset":      ["data", "benchmark", "collection", "corpus", "database"],
        "benchmark":    ["dataset", "evaluation", "standard", "challenge", "competition"],
        "quality":      ["clean", "labeled", "annotated", "curated", "verified"],
        "large":        ["big", "million", "thousands", "extensive", "comprehensive"],
        # ── Road / infrastructure domain ─────────────────────────────────────
        "pothole":      ["road damage", "pavement", "crack", "asphalt", "road defect",
                         "road hazard", "surface damage", "road crack", "road condition"],
        "road":         ["pavement", "highway", "street", "asphalt", "infrastructure",
                         "traffic", "driving", "lane", "intersection"],
        "infrastructure":["road", "bridge", "construction", "urban", "pavement", "civil"],
        "damage":       ["defect", "crack", "deterioration", "fault", "wear", "break",
                         "pothole", "hazard", "anomaly"],
        # ── Generic CV words that should NOT count as domain matches ─────────
        # These are query-structural words that appear in thousands of Kaggle datasets.
        # Giving them synonym matches would let Cards/Cats/Flickr pass the gate.
        # They are intentionally left out of the synonym table so the gate only
        # passes when a DOMAIN-SPECIFIC keyword like "pothole" or "road" matches.
        # "image", "classification", "detection", "computer", "vision" are query
        # noise for a pothole search — they are not synonyms for road damage.
    }

    def _score_description_match(self, ds: RawDataset) -> float:
        """
        Semantic recall: what fraction of the user's INTENT is covered by this dataset.

        Unlike exact keyword matching, this uses synonym expansion so:
        - "crop disease detector" matches "PlantVillage plant pathology classification"
        - "speech recognizer Hindi" matches "Indic ASR audio transcription dataset"
        - Typos in the query don't matter because enricher already expanded them

        A keyword is considered "matched" if:
          1. The exact keyword appears in the dataset text, OR
          2. Any synonym of the keyword appears in the dataset text

        Score = matched_concepts / total_query_keywords
        Gate threshold (0.25) means at least 2-3 core concepts must match.
        """
        if not self.keywords:
            return 0.5

        # Build full searchable text from dataset
        text_parts = [
            ds.title or "",
            ds.description_short or "",
            getattr(ds, "description", "") or "",
            " ".join(ds.tags_primary or []),
            " ".join(getattr(ds, "task_types", []) or []),
            " ".join(getattr(ds, "modalities", []) or []),
        ]
        full_text = " ".join(text_parts).lower()

        if not full_text.strip():
            return 0.0

        dataset_tokens = set(
            t for part in text_parts
            for t in part.lower().split()
            if len(t) >= 2
        )

        # FIX v3.6.0: filter structural/generic ML words from keywords before
        # scoring. These appear in nearly every image dataset (cards, cats, plants)
        # and dilute domain-specific signals. When a query has domain words like
        # "pothole", "road", "damage" alongside "image classification", the domain
        # words are what differentiate relevant from irrelevant datasets.
        # Keywords like "image", "classification", "computer", "vision", "dataset"
        # appear in 90%+ of all Kaggle/HF image datasets — they carry zero signal.
        _GENERIC_NOISE = frozenset({
            "image", "images", "classification", "classifier", "computer", "vision",
            "dataset", "datasets", "data", "machine", "learning", "deep", "neural",
            "network", "model", "detection", "detector", "training", "train",
            "test", "benchmark", "label", "labels", "annotated", "annotation",
        })
        # Use domain-filtered keywords for scoring; if ALL keywords are generic
        # (e.g. query="image classification") keep them all to avoid empty set.
        _filtered_kws = [k for k in self.keywords if k.lower() not in _GENERIC_NOISE]
        _score_kws = _filtered_kws if _filtered_kws else self.keywords

        matched = 0
        for kw in _score_kws:
            kw_lower = kw.lower()

            # Direct match
            if kw_lower in dataset_tokens or kw_lower in full_text:
                matched += 1
                continue

            # Synonym match — "crop" also matches if "plant" or "agriculture" is present
            synonyms = self._SYNONYMS.get(kw_lower, [])
            if any(syn in full_text for syn in synonyms):
                matched += 1
                continue

            # Partial/stem match — "detect" matches "detection", "detector"
            if any(kw_lower[:5] in token for token in dataset_tokens if len(token) >= 5):
                matched += 0.5  # partial credit

        n_keywords = len(_score_kws)
        recall = min(matched / n_keywords, 1.0) if n_keywords > 0 else 0.0
        return round(recall, 4)

    # ─────────────────────────────────────────────────────────────────────────
    # WEIGHT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """
        Ensure weights sum to exactly 1.0.
        Any missing dimensions get 0 weight.
        All-zero weights → equal weight fallback.
        """
        dims = ["task_relevance", "quality", "popularity", "freshness", "description_match"]
        full = {d: weights.get(d, 0.0) for d in dims}

        total = sum(full.values())
        if total == 0.0:
            return {d: 1.0 / len(dims) for d in dims}

        return {d: round(v / total, 6) for d, v in full.items()}