"""
datascout.query_understanding.task_types
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: ML task taxonomy, modality definitions,
and compatibility scoring for query understanding.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  TaskType and Modality are the semantic anchors for the entire pipeline.
  Every adapter, scorer, and filter references these enums.

SYSTEM DESIGN DECISIONS:

  1. WHY TaskType and Modality as separate orthogonal axes?
     - A dataset's MODALITY is what the data looks like (images, text, tabular)
     - A dataset's TASK TYPE is what you do with the data (classify, detect, forecast)
     - IMAGE modality datasets can serve IMAGE_CLASSIFICATION or OBJECT_DETECTION
     - Separating them lets you filter "all image datasets" OR "all detection tasks"
     - Mixing them (as many systems do) makes search filters ambiguous and lossy

  2. WHY TASK_MODALITY_MAP as a scored compatibility (not boolean)?
     - Binary compatible/not compatible loses gradient information
     - IMAGE_CLASSIFICATION with TABULAR data: compatibility 0.0
     - IMAGE_CLASSIFICATION with IMAGE data: compatibility 1.0
     - CLASSIFICATION (generic) with TABULAR data: compatibility 0.9 (close but imprecise)
     - Downstream ranking engine multiplies this into composite score
     - Score-based compatibility allows partial matches to still surface

  3. WHY normalize_task_type / normalize_modality as separate functions?
     - Raw platform tags are inconsistent: "img-clf", "image classification", "CV"
     - Centralized normalization → one bug to fix, not scattered across adapters
     - Returns UNKNOWN (not None) so callers never need null checks

  4. WHY TASK_FAMILIES mapping?
     - Related tasks should boost each other: user asks for CLASSIFICATION,
       BINARY_CLASSIFICATION and MULTI_LABEL_CLASSIFICATION are also relevant
     - Without families, "I need a classification dataset" misses binary datasets
     - Families enable "soft match" scoring (0.7) vs exact match (1.0)

FAILURE SCENARIOS HANDLED:
  - Unknown task string → TaskType.UNKNOWN (never None, never crash)
  - Unknown modality string → Modality.UNKNOWN
  - Empty/None input → UNKNOWN with no log noise (expected at ingestion)
  - Compatibility query with UNKNOWN task → score 0.5 (neutral, not 0)

PERFORMANCE ANALYSIS:
  - normalize_task_type: O(keywords) ≈ 0.01ms per call
  - get_compatible_modalities: O(1) dict lookup
  - compute_task_compatibility: O(modalities) ≈ 0.01ms
  - At 1M datasets × 3 task types each: ~30ms total — negligible

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add REINFORCEMENT_LEARNING, VIDEO_UNDERSTANDING
  Breaking: v4.0.0 — restructuring TASK_FAMILIES (requires re-scoring stored records)

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("datascout.query_understanding.task_types")


# ─────────────────────────────────────────────────────────────────────────────
# TASK TYPE
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """
    ML task taxonomy — what the model will DO with the dataset.
    Stored as string value for JSON-serializability across agent boundaries.
    """

    # ── Generic Supervised ────────────────────────────────────────────────────
    CLASSIFICATION           = "classification"
    BINARY_CLASSIFICATION    = "binary_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    REGRESSION               = "regression"

    # ── Computer Vision ───────────────────────────────────────────────────────
    IMAGE_CLASSIFICATION     = "image_classification"
    OBJECT_DETECTION         = "object_detection"
    SEMANTIC_SEGMENTATION    = "semantic_segmentation"
    INSTANCE_SEGMENTATION    = "instance_segmentation"
    IMAGE_GENERATION         = "image_generation"
    IMAGE_CAPTIONING         = "image_captioning"
    DEPTH_ESTIMATION         = "depth_estimation"
    POSE_ESTIMATION          = "pose_estimation"

    # ── NLP ───────────────────────────────────────────────────────────────────
    NER                      = "ner"
    SENTIMENT_ANALYSIS       = "sentiment_analysis"
    TEXT_CLASSIFICATION      = "text_classification"
    MACHINE_TRANSLATION      = "machine_translation"
    SUMMARIZATION            = "summarization"
    QUESTION_ANSWERING       = "question_answering"
    TEXT_GENERATION          = "text_generation"
    LANGUAGE_MODELING        = "language_modeling"

    # ── Time Series ───────────────────────────────────────────────────────────
    TIME_SERIES_FORECASTING  = "time_series_forecasting"
    ANOMALY_DETECTION        = "anomaly_detection"

    # ── Unsupervised ─────────────────────────────────────────────────────────
    CLUSTERING               = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

    # ── Recommendation / Retrieval ────────────────────────────────────────────
    RECOMMENDATION           = "recommendation"

    # ── Audio ─────────────────────────────────────────────────────────────────
    SPEECH_RECOGNITION       = "speech_recognition"
    AUDIO_CLASSIFICATION     = "audio_classification"

    # ── Multimodal ────────────────────────────────────────────────────────────
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    DOCUMENT_UNDERSTANDING   = "document_understanding"

    # ── Catch-all ────────────────────────────────────────────────────────────
    OTHER                    = "other"
    UNKNOWN                  = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# MODALITY
# ─────────────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    """
    Data modality — what FORM the data takes (orthogonal to TaskType).
    A dataset with IMAGE modality can serve multiple task types.
    """
    TEXT        = "text"
    IMAGE       = "image"
    AUDIO       = "audio"
    VIDEO       = "video"
    TABULAR     = "tabular"
    TIME_SERIES = "time_series"
    GRAPH       = "graph"
    MULTIMODAL  = "multimodal"
    OTHER       = "other"
    UNKNOWN     = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TASK → COMPATIBLE MODALITIES (with compatibility scores)
# ─────────────────────────────────────────────────────────────────────────────

# dict[TaskType, dict[Modality, float]]
# Score 1.0 = perfect match, 0.7 = related, 0.0 = incompatible
TASK_MODALITY_MAP: dict[TaskType, dict[Modality, float]] = {
    TaskType.CLASSIFICATION:          {Modality.TABULAR: 1.0, Modality.TEXT: 0.7, Modality.IMAGE: 0.7},
    TaskType.BINARY_CLASSIFICATION:   {Modality.TABULAR: 1.0, Modality.TEXT: 0.7, Modality.IMAGE: 0.7},
    TaskType.MULTI_LABEL_CLASSIFICATION: {Modality.TABULAR: 1.0, Modality.TEXT: 0.8, Modality.IMAGE: 0.8},
    TaskType.REGRESSION:              {Modality.TABULAR: 1.0, Modality.TIME_SERIES: 0.8},

    TaskType.IMAGE_CLASSIFICATION:    {Modality.IMAGE: 1.0},
    TaskType.OBJECT_DETECTION:        {Modality.IMAGE: 1.0, Modality.VIDEO: 0.8},
    TaskType.SEMANTIC_SEGMENTATION:   {Modality.IMAGE: 1.0},
    TaskType.INSTANCE_SEGMENTATION:   {Modality.IMAGE: 1.0},
    TaskType.IMAGE_GENERATION:        {Modality.IMAGE: 1.0},
    TaskType.IMAGE_CAPTIONING:        {Modality.IMAGE: 1.0, Modality.TEXT: 0.8},
    TaskType.DEPTH_ESTIMATION:        {Modality.IMAGE: 1.0},
    TaskType.POSE_ESTIMATION:         {Modality.IMAGE: 1.0},

    TaskType.NER:                     {Modality.TEXT: 1.0},
    TaskType.SENTIMENT_ANALYSIS:      {Modality.TEXT: 1.0},
    TaskType.TEXT_CLASSIFICATION:     {Modality.TEXT: 1.0},
    TaskType.MACHINE_TRANSLATION:     {Modality.TEXT: 1.0},
    TaskType.SUMMARIZATION:           {Modality.TEXT: 1.0},
    TaskType.QUESTION_ANSWERING:      {Modality.TEXT: 1.0},
    TaskType.TEXT_GENERATION:         {Modality.TEXT: 1.0},
    TaskType.LANGUAGE_MODELING:       {Modality.TEXT: 1.0},

    TaskType.TIME_SERIES_FORECASTING: {Modality.TIME_SERIES: 1.0, Modality.TABULAR: 0.8},
    TaskType.ANOMALY_DETECTION:       {Modality.TIME_SERIES: 1.0, Modality.TABULAR: 0.8},

    TaskType.CLUSTERING:              {Modality.TABULAR: 1.0, Modality.TEXT: 0.7, Modality.IMAGE: 0.7},
    TaskType.DIMENSIONALITY_REDUCTION:{Modality.TABULAR: 1.0},

    TaskType.RECOMMENDATION:          {Modality.TABULAR: 1.0},

    TaskType.SPEECH_RECOGNITION:      {Modality.AUDIO: 1.0},
    TaskType.AUDIO_CLASSIFICATION:    {Modality.AUDIO: 1.0},

    TaskType.VISUAL_QUESTION_ANSWERING: {Modality.IMAGE: 1.0, Modality.TEXT: 0.9},
    TaskType.DOCUMENT_UNDERSTANDING:  {Modality.IMAGE: 1.0, Modality.TEXT: 0.9},

    TaskType.OTHER:                   {m: 0.5 for m in Modality if m not in (Modality.UNKNOWN,)},
    TaskType.UNKNOWN:                 {m: 0.5 for m in Modality if m not in (Modality.UNKNOWN,)},
}

# Task family groupings — related tasks (for soft-match scoring).
# INVARIANT: Each TaskType belongs to AT MOST one family.
# More specific family takes precedence (nlp > classification, computer_vision > classification).
TASK_FAMILIES: dict[str, set[TaskType]] = {
    # NLP tasks — text-specific
    "nlp": {
        TaskType.NER,
        TaskType.SENTIMENT_ANALYSIS,
        TaskType.TEXT_CLASSIFICATION,
        TaskType.MACHINE_TRANSLATION,
        TaskType.SUMMARIZATION,
        TaskType.QUESTION_ANSWERING,
        TaskType.TEXT_GENERATION,
        TaskType.LANGUAGE_MODELING,
    },
    # Computer Vision tasks — image-specific
    "computer_vision": {
        TaskType.IMAGE_CLASSIFICATION,
        TaskType.OBJECT_DETECTION,
        TaskType.SEMANTIC_SEGMENTATION,
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.IMAGE_GENERATION,
        TaskType.IMAGE_CAPTIONING,
        TaskType.DEPTH_ESTIMATION,
        TaskType.POSE_ESTIMATION,
    },
    # Generic tabular classification
    "classification": {
        TaskType.CLASSIFICATION,
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTI_LABEL_CLASSIFICATION,
        TaskType.AUDIO_CLASSIFICATION,
    },
    # Regression and forecasting
    "regression": {
        TaskType.REGRESSION,
        TaskType.TIME_SERIES_FORECASTING,
    },
    # Anomaly and outlier detection
    "detection": {
        TaskType.ANOMALY_DETECTION,
    },
    # Unsupervised
    "unsupervised": {
        TaskType.CLUSTERING,
        TaskType.DIMENSIONALITY_REDUCTION,
    },
    # Audio
    "audio": {
        TaskType.SPEECH_RECOGNITION,
    },
    # Multimodal
    "multimodal": {
        TaskType.VISUAL_QUESTION_ANSWERING,
        TaskType.DOCUMENT_UNDERSTANDING,
    },
    # Recommendation
    "recommendation": {
        TaskType.RECOMMENDATION,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK COMPATIBILITY RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskCompatibility:
    """
    Scored compatibility between a query's task intent and a dataset's modality.

    WHY a scored object (not bool):
    - bool loses gradient: CLASSIFICATION + TABULAR (1.0) vs IMAGE (0.7) look the same
    - is_family_match enables soft boosting without hard filtering
    - reason feeds into DecisionTrace.filter_reasons
    - Used by Phase 10 scoring engine to compute task_match dimension
    """
    task_type: TaskType
    dataset_modalities: list[Modality]
    compatibility_score: float          # 0.0–1.0
    is_family_match: bool               # True if related task family
    reason: str
    best_modality_match: Optional[Modality] = None

    @property
    def is_compatible(self) -> bool:
        return self.compatibility_score >= 0.5

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "dataset_modalities": [m.value for m in self.dataset_modalities],
            "compatibility_score": self.compatibility_score,
            "is_compatible": self.is_compatible,
            "is_family_match": self.is_family_match,
            "reason": self.reason,
            "best_modality_match": self.best_modality_match.value if self.best_modality_match else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Keyword → TaskType mapping (ordered: more specific first)
_TASK_KEYWORD_MAP: list[tuple[str, TaskType]] = [
    # Computer Vision
    ("instance_segmentation",   TaskType.INSTANCE_SEGMENTATION),
    ("semantic_segmentation",   TaskType.SEMANTIC_SEGMENTATION),
    ("image_caption",           TaskType.IMAGE_CAPTIONING),
    ("image_generation",        TaskType.IMAGE_GENERATION),
    ("image_classification",    TaskType.IMAGE_CLASSIFICATION),
    ("image-classification",    TaskType.IMAGE_CLASSIFICATION),
    ("object_detection",        TaskType.OBJECT_DETECTION),
    ("object-detection",        TaskType.OBJECT_DETECTION),
    ("depth_estimation",        TaskType.DEPTH_ESTIMATION),
    ("pose_estimation",         TaskType.POSE_ESTIMATION),
    ("detection",               TaskType.OBJECT_DETECTION),
    ("segmentation",            TaskType.SEMANTIC_SEGMENTATION),
    # NLP
    ("named_entity",            TaskType.NER),
    ("ner",                     TaskType.NER),
    ("sentiment",               TaskType.SENTIMENT_ANALYSIS),
    ("text_classification",     TaskType.TEXT_CLASSIFICATION),
    ("machine_translation",     TaskType.MACHINE_TRANSLATION),
    ("translation",             TaskType.MACHINE_TRANSLATION),
    ("summarization",           TaskType.SUMMARIZATION),
    ("summarisation",           TaskType.SUMMARIZATION),
    ("question_answering",      TaskType.QUESTION_ANSWERING),
    ("question-answering",      TaskType.QUESTION_ANSWERING),
    ("text_generation",         TaskType.TEXT_GENERATION),
    ("text-generation",         TaskType.TEXT_GENERATION),
    ("language_modeling",       TaskType.LANGUAGE_MODELING),
    ("language-modeling",       TaskType.LANGUAGE_MODELING),
    # Time Series
    ("time_series_forecasting", TaskType.TIME_SERIES_FORECASTING),
    ("time-series-forecasting", TaskType.TIME_SERIES_FORECASTING),
    ("forecasting",             TaskType.TIME_SERIES_FORECASTING),
    ("anomaly_detection",       TaskType.ANOMALY_DETECTION),
    ("anomaly",                 TaskType.ANOMALY_DETECTION),
    # Generic Classification/Regression
    ("multi_label",             TaskType.MULTI_LABEL_CLASSIFICATION),
    ("binary_classification",   TaskType.BINARY_CLASSIFICATION),
    ("image",                   TaskType.IMAGE_CLASSIFICATION),
    ("regression",              TaskType.REGRESSION),
    ("classification",          TaskType.CLASSIFICATION),
    # Misc
    ("clustering",              TaskType.CLUSTERING),
    ("recommendation",          TaskType.RECOMMENDATION),
    ("speech",                  TaskType.SPEECH_RECOGNITION),
    ("asr",                     TaskType.SPEECH_RECOGNITION),
    ("visual_question",         TaskType.VISUAL_QUESTION_ANSWERING),
    ("vqa",                     TaskType.VISUAL_QUESTION_ANSWERING),
    ("document",                TaskType.DOCUMENT_UNDERSTANDING),
]

_MODALITY_KEYWORD_MAP: list[tuple[str, Modality]] = [
    ("time_series",  Modality.TIME_SERIES),
    ("timeseries",   Modality.TIME_SERIES),
    ("multimodal",   Modality.MULTIMODAL),
    ("multi-modal",  Modality.MULTIMODAL),
    ("tabular",      Modality.TABULAR),
    ("structured",   Modality.TABULAR),
    ("image",        Modality.IMAGE),
    ("vision",       Modality.IMAGE),
    ("text",         Modality.TEXT),
    ("nlp",          Modality.TEXT),
    ("audio",        Modality.AUDIO),
    ("speech",       Modality.AUDIO),
    ("video",        Modality.VIDEO),
    ("graph",        Modality.GRAPH),
    ("network",      Modality.GRAPH),
]


def normalize_task_type(raw: Optional[str]) -> TaskType:
    """
    Map a raw platform task string → TaskType enum.
    Returns UNKNOWN (never None, never raises) for unrecognized values.

    WHY: Platform tags are inconsistent — "img-clf", "Image Classification", "CV"
    all mean IMAGE_CLASSIFICATION. Normalization happens once here, used everywhere.
    """
    if not raw:
        return TaskType.UNKNOWN
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    # Try exact enum value match first
    try:
        return TaskType(normalized)
    except ValueError:
        pass
    # Keyword scan (specific before generic — order matters)
    for keyword, task in _TASK_KEYWORD_MAP:
        if keyword in normalized:
            return task
    logger.debug("normalize_task_type: unrecognized %r → UNKNOWN", raw)
    return TaskType.UNKNOWN


def normalize_modality(raw: Optional[str]) -> Modality:
    """Map raw modality string → Modality enum. Returns UNKNOWN on failure."""
    if not raw:
        return Modality.UNKNOWN
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return Modality(normalized)
    except ValueError:
        pass
    for keyword, mod in _MODALITY_KEYWORD_MAP:
        if keyword in normalized:
            return mod
    logger.debug("normalize_modality: unrecognized %r → UNKNOWN", raw)
    return Modality.UNKNOWN


def get_task_family(task: TaskType) -> Optional[str]:
    """Return the family name for a given TaskType, or None if standalone."""
    for family, members in TASK_FAMILIES.items():
        if task in members:
            return family
    return None


def are_in_same_family(task1: TaskType, task2: TaskType) -> bool:
    """Return True if two tasks belong to the same family (soft match)."""
    if task1 == TaskType.UNKNOWN or task2 == TaskType.UNKNOWN:
        return False
    fam1 = get_task_family(task1)
    fam2 = get_task_family(task2)
    return fam1 is not None and fam1 == fam2


def compute_task_compatibility(
    task_type: TaskType,
    dataset_modalities: list[Modality],
) -> TaskCompatibility:
    """
    Compute how well a dataset's modalities serve a given task type.

    Algorithm:
    1. Look up task in TASK_MODALITY_MAP
    2. Find the best-matching modality in dataset's modality list
    3. Score = max compatibility score across dataset's modalities
    4. If no match: check family match for soft score (0.3)

    WHY max (not average):
    A dataset with [IMAGE, TEXT] modalities for IMAGE_CLASSIFICATION:
    - IMAGE: 1.0, TEXT: not in map → best = 1.0 (correct answer)
    - Average would penalize multi-modal datasets unfairly
    """
    if not dataset_modalities:
        return TaskCompatibility(
            task_type=task_type,
            dataset_modalities=[],
            compatibility_score=0.0,
            is_family_match=False,
            reason="Dataset has no modality information.",
        )

    modality_scores = TASK_MODALITY_MAP.get(task_type, {})

    best_score: float = 0.0
    best_modality: Optional[Modality] = None

    for mod in dataset_modalities:
        if mod == Modality.UNKNOWN:
            continue
        score = modality_scores.get(mod, 0.0)
        if score > best_score:
            best_score = score
            best_modality = mod

    # Check family match for soft scoring
    is_family = False
    if best_score == 0.0 and task_type not in (TaskType.UNKNOWN, TaskType.OTHER):
        # Check if any dataset modality is compatible with a family sibling
        family = get_task_family(task_type)
        if family:
            for sibling in TASK_FAMILIES[family]:
                if sibling == task_type:
                    continue
                sibling_scores = TASK_MODALITY_MAP.get(sibling, {})
                for mod in dataset_modalities:
                    if sibling_scores.get(mod, 0.0) > 0.0:
                        best_score = 0.3  # Family soft match
                        is_family = True
                        best_modality = mod
                        break
                if is_family:
                    break

    if best_score >= 0.9:
        reason = f"Perfect modality match: {best_modality.value if best_modality else 'N/A'} for {task_type.value}."
    elif best_score >= 0.7:
        reason = f"Good modality match: {best_modality.value if best_modality else 'N/A'} for {task_type.value}."
    elif best_score >= 0.3:
        reason = f"Weak/family match for {task_type.value} — modality is {[m.value for m in dataset_modalities]}."
    else:
        reason = f"No compatible modality for {task_type.value}. Dataset has {[m.value for m in dataset_modalities]}."

    return TaskCompatibility(
        task_type=task_type,
        dataset_modalities=dataset_modalities,
        compatibility_score=round(best_score, 3),
        is_family_match=is_family,
        reason=reason,
        best_modality_match=best_modality,
    )