"""
datascout.contracts.task_types
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: ML task taxonomy and dataset-to-task compatibility scoring.

AGENT-0 CONTEXT:
  This file is part of Agent-0 — the foundation layer of a multi-agent system.
  Agent-0's output is the binding contract for Agent-1 through Agent-N.
  Changes here require version bumps and migration plans.

SYSTEM DESIGN DECISIONS:

  1. WHY TaskType and Modality as separate dimensions?
     - TaskType = what you DO with the data (classify, detect, translate)
     - Modality = what FORM the data is in (images, text, tabular)
     - A dataset can be TEXT modality but used for CLASSIFICATION or SENTIMENT or NER
     - Separating them allows independent filtering on either axis

  2. WHY TASK_MODALITY_MAP static mapping?
     - IMAGE_CLASSIFICATION requires IMAGE modality — this is domain knowledge
     - Encoding it as data (not logic) makes it testable, auditable, extensible
     - Agent-1 uses this map to compute compatibility scores without ML inference

  3. WHY compute_task_compatibility() returning a scored object vs bool?
     - bool: "is this compatible?" loses the WHY
     - float score: allows ranking — 0.9 compatible vs 0.6 compatible matters
     - Structured result: carries missing_modalities for explanation generation

FAILURE SCENARIOS HANDLED:
  - Unknown task type string → normalize_task_type() → TaskType.OTHER + log
  - Unknown modality string → normalize_modality() → Modality.OTHER + log
  - Empty dataset modalities → compatibility = 0.0 with reason

VERSIONING STRATEGY:
  Current:  v3.0.0
  Next:     v3.1.0 — add REINFORCEMENT_LEARNING, VIDEO_UNDERSTANDING task types
  Breaking: v4.0.0 — restructuring TASK_MODALITY_MAP format

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("datascout.contracts.task_types")


# ─────────────────────────────────────────────────────────────────────────────
# TASK TYPE
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """
    ML task taxonomy — what the model will DO with the dataset.
    Stored as string value. Normalized at ingestion from raw platform tags.
    """
    # ── Supervised / Classification ──────────────────────────────────────────
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
    NER                      = "ner"                   # Named Entity Recognition
    SENTIMENT_ANALYSIS       = "sentiment_analysis"
    TEXT_CLASSIFICATION      = "text_classification"
    MACHINE_TRANSLATION      = "machine_translation"
    SUMMARIZATION            = "summarization"
    QUESTION_ANSWERING       = "question_answering"
    TEXT_GENERATION          = "text_generation"
    LANGUAGE_MODELING        = "language_modeling"
    RELATION_EXTRACTION      = "relation_extraction"
    COREFERENCE_RESOLUTION   = "coreference_resolution"

    # ── Time Series ───────────────────────────────────────────────────────────
    TIME_SERIES_FORECASTING  = "time_series_forecasting"
    ANOMALY_DETECTION        = "anomaly_detection"

    # ── Unsupervised ─────────────────────────────────────────────────────────
    CLUSTERING               = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

    # ── Recommendation ───────────────────────────────────────────────────────
    RECOMMENDATION           = "recommendation"

    # ── Audio ─────────────────────────────────────────────────────────────────
    SPEECH_RECOGNITION       = "speech_recognition"
    AUDIO_CLASSIFICATION     = "audio_classification"

    # ── Multimodal ────────────────────────────────────────────────────────────
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    DOCUMENT_UNDERSTANDING   = "document_understanding"

    # ── Catch-all ────────────────────────────────────────────────────────────
    OTHER                    = "other"


# ─────────────────────────────────────────────────────────────────────────────
# MODALITY
# ─────────────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    """
    Data modality — what FORM the data takes.
    Orthogonal to TaskType: TEXT modality can serve many tasks.
    """
    TEXT       = "text"
    IMAGE      = "image"
    AUDIO      = "audio"
    VIDEO      = "video"
    TABULAR    = "tabular"
    TIME_SERIES = "time_series"
    GRAPH      = "graph"
    MULTIMODAL = "multimodal"
    OTHER      = "other"


# ─────────────────────────────────────────────────────────────────────────────
# TASK → REQUIRED MODALITIES MAP
# ─────────────────────────────────────────────────────────────────────────────

TASK_MODALITY_MAP: dict[TaskType, frozenset[Modality]] = {
    # Computer Vision — requires IMAGE
    TaskType.IMAGE_CLASSIFICATION:      frozenset({Modality.IMAGE}),
    TaskType.OBJECT_DETECTION:          frozenset({Modality.IMAGE}),
    TaskType.SEMANTIC_SEGMENTATION:     frozenset({Modality.IMAGE}),
    TaskType.INSTANCE_SEGMENTATION:     frozenset({Modality.IMAGE}),
    TaskType.IMAGE_GENERATION:          frozenset({Modality.IMAGE}),
    TaskType.IMAGE_CAPTIONING:          frozenset({Modality.IMAGE, Modality.TEXT}),
    TaskType.DEPTH_ESTIMATION:          frozenset({Modality.IMAGE}),
    TaskType.POSE_ESTIMATION:           frozenset({Modality.IMAGE}),

    # NLP — requires TEXT
    TaskType.NER:                       frozenset({Modality.TEXT}),
    TaskType.SENTIMENT_ANALYSIS:        frozenset({Modality.TEXT}),
    TaskType.TEXT_CLASSIFICATION:       frozenset({Modality.TEXT}),
    TaskType.MACHINE_TRANSLATION:       frozenset({Modality.TEXT}),
    TaskType.SUMMARIZATION:             frozenset({Modality.TEXT}),
    TaskType.QUESTION_ANSWERING:        frozenset({Modality.TEXT}),
    TaskType.TEXT_GENERATION:           frozenset({Modality.TEXT}),
    TaskType.LANGUAGE_MODELING:         frozenset({Modality.TEXT}),
    TaskType.RELATION_EXTRACTION:       frozenset({Modality.TEXT}),
    TaskType.COREFERENCE_RESOLUTION:    frozenset({Modality.TEXT}),

    # Tabular — requires TABULAR
    TaskType.CLASSIFICATION:            frozenset({Modality.TABULAR}),
    TaskType.BINARY_CLASSIFICATION:     frozenset({Modality.TABULAR}),
    TaskType.MULTI_LABEL_CLASSIFICATION: frozenset({Modality.TABULAR, Modality.TEXT}),
    TaskType.REGRESSION:                frozenset({Modality.TABULAR}),
    TaskType.CLUSTERING:                frozenset({Modality.TABULAR}),
    TaskType.DIMENSIONALITY_REDUCTION:  frozenset({Modality.TABULAR}),
    TaskType.RECOMMENDATION:            frozenset({Modality.TABULAR}),

    # Time Series — requires TIME_SERIES
    TaskType.TIME_SERIES_FORECASTING:   frozenset({Modality.TIME_SERIES}),
    TaskType.ANOMALY_DETECTION:         frozenset({Modality.TIME_SERIES, Modality.TABULAR}),

    # Audio — requires AUDIO
    TaskType.SPEECH_RECOGNITION:        frozenset({Modality.AUDIO}),
    TaskType.AUDIO_CLASSIFICATION:      frozenset({Modality.AUDIO}),

    # Multimodal — requires multiple
    TaskType.VISUAL_QUESTION_ANSWERING: frozenset({Modality.IMAGE, Modality.TEXT}),
    TaskType.DOCUMENT_UNDERSTANDING:    frozenset({Modality.IMAGE, Modality.TEXT}),

    # Catch-all — no modality requirement
    TaskType.OTHER:                     frozenset(),
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK COMPATIBILITY RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskCompatibility:
    """
    Result of computing compatibility between a dataset's modalities
    and a requested task type.

    WHY a scored object vs bool:
    - bool loses the gradient: 0.9 compatible vs 0.6 compatible matters for ranking
    - missing_modalities feeds into StructuredExplanation.weaknesses
    - reason feeds into DecisionTrace.filter_reasons
    """
    task_type: TaskType
    compatibility_score: float          # 0.0 – 1.0
    reason: str                         # Human-readable explanation
    required_modalities: frozenset[Modality] = field(default_factory=frozenset)
    matched_modalities: frozenset[Modality]  = field(default_factory=frozenset)
    missing_modalities: frozenset[Modality]  = field(default_factory=frozenset)
    is_compatible: bool = False         # score >= 0.5 threshold

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "compatibility_score": self.compatibility_score,
            "reason": self.reason,
            "required_modalities": [m.value for m in self.required_modalities],
            "matched_modalities": [m.value for m in self.matched_modalities],
            "missing_modalities": [m.value for m in self.missing_modalities],
            "is_compatible": self.is_compatible,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_TASK_KEYWORD_MAP: list[tuple[str, TaskType]] = [
    ("image_classification",       TaskType.IMAGE_CLASSIFICATION),
    ("image-classification",       TaskType.IMAGE_CLASSIFICATION),
    ("img-clf",                    TaskType.IMAGE_CLASSIFICATION),
    ("object_detection",           TaskType.OBJECT_DETECTION),
    ("object-detection",           TaskType.OBJECT_DETECTION),
    ("detection",                  TaskType.OBJECT_DETECTION),
    ("segmentation",               TaskType.SEMANTIC_SEGMENTATION),
    ("semantic_segmentation",      TaskType.SEMANTIC_SEGMENTATION),
    ("instance_segmentation",      TaskType.INSTANCE_SEGMENTATION),
    ("image_generation",           TaskType.IMAGE_GENERATION),
    ("image-generation",           TaskType.IMAGE_GENERATION),
    ("image_captioning",           TaskType.IMAGE_CAPTIONING),
    ("named_entity",               TaskType.NER),
    ("ner",                        TaskType.NER),
    ("sentiment",                  TaskType.SENTIMENT_ANALYSIS),
    ("text_classification",        TaskType.TEXT_CLASSIFICATION),
    ("text-classification",        TaskType.TEXT_CLASSIFICATION),
    ("translation",                TaskType.MACHINE_TRANSLATION),
    ("machine_translation",        TaskType.MACHINE_TRANSLATION),
    ("summarization",              TaskType.SUMMARIZATION),
    ("summarisation",              TaskType.SUMMARIZATION),
    ("question_answering",         TaskType.QUESTION_ANSWERING),
    ("question-answering",         TaskType.QUESTION_ANSWERING),
    ("qa",                         TaskType.QUESTION_ANSWERING),
    ("text_generation",            TaskType.TEXT_GENERATION),
    ("text-generation",            TaskType.TEXT_GENERATION),
    ("language_modeling",          TaskType.LANGUAGE_MODELING),
    ("language_model",             TaskType.LANGUAGE_MODELING),
    ("time_series_forecasting",    TaskType.TIME_SERIES_FORECASTING),
    ("time-series-forecasting",    TaskType.TIME_SERIES_FORECASTING),
    ("forecasting",                TaskType.TIME_SERIES_FORECASTING),
    ("anomaly_detection",          TaskType.ANOMALY_DETECTION),
    ("anomaly-detection",          TaskType.ANOMALY_DETECTION),
    ("anomaly",                    TaskType.ANOMALY_DETECTION),
    ("clustering",                 TaskType.CLUSTERING),
    ("recommendation",             TaskType.RECOMMENDATION),
    ("speech_recognition",         TaskType.SPEECH_RECOGNITION),
    ("speech-recognition",         TaskType.SPEECH_RECOGNITION),
    ("asr",                        TaskType.SPEECH_RECOGNITION),
    ("audio_classification",       TaskType.AUDIO_CLASSIFICATION),
    ("visual_question_answering",  TaskType.VISUAL_QUESTION_ANSWERING),
    ("vqa",                        TaskType.VISUAL_QUESTION_ANSWERING),
    ("document_understanding",     TaskType.DOCUMENT_UNDERSTANDING),
    ("binary_classification",      TaskType.BINARY_CLASSIFICATION),
    ("multi_label",                TaskType.MULTI_LABEL_CLASSIFICATION),
    ("regression",                 TaskType.REGRESSION),
    ("classification",             TaskType.CLASSIFICATION),
]

_MODALITY_KEYWORD_MAP: list[tuple[str, Modality]] = [
    ("text",        Modality.TEXT),
    ("nlp",         Modality.TEXT),
    ("image",       Modality.IMAGE),
    ("vision",      Modality.IMAGE),
    ("audio",       Modality.AUDIO),
    ("speech",      Modality.AUDIO),
    ("video",       Modality.VIDEO),
    ("tabular",     Modality.TABULAR),
    ("structured",  Modality.TABULAR),
    ("table",       Modality.TABULAR),
    ("time_series", Modality.TIME_SERIES),
    ("timeseries",  Modality.TIME_SERIES),
    ("graph",       Modality.GRAPH),
    ("network",     Modality.GRAPH),
    ("multimodal",  Modality.MULTIMODAL),
    ("multi-modal", Modality.MULTIMODAL),
]


def normalize_task_type(raw: Optional[str]) -> TaskType:
    """
    Map raw platform task string → TaskType enum.
    WHY: Kaggle/HF/OpenML each use different naming conventions.
    """
    if not raw:
        return TaskType.OTHER
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return TaskType(normalized)
    except ValueError:
        pass
    for keyword, task in _TASK_KEYWORD_MAP:
        if keyword in normalized:
            return task
    logger.debug("normalize_task_type: unknown task %r → OTHER", raw)
    return TaskType.OTHER


def normalize_modality(raw: Optional[str]) -> Modality:
    """Map raw modality string → Modality enum."""
    if not raw:
        return Modality.OTHER
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return Modality(normalized)
    except ValueError:
        pass
    for keyword, mod in _MODALITY_KEYWORD_MAP:
        if keyword in normalized:
            return mod
    logger.debug("normalize_modality: unknown modality %r → OTHER", raw)
    return Modality.OTHER


# ─────────────────────────────────────────────────────────────────────────────
# COMPATIBILITY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_task_compatibility(
    dataset_modalities: list[Modality],
    query_task_type: TaskType,
) -> TaskCompatibility:
    """
    Compute compatibility between a dataset's modalities and a requested task.

    Algorithm:
    1. Look up required modalities for the task from TASK_MODALITY_MAP
    2. If no requirements (OTHER task) → full compatibility
    3. Compute intersection and missing sets
    4. Score = matched / required (Jaccard-style partial credit)

    WHY partial credit scoring:
    - A dataset with [IMAGE, TEXT] asked for IMAGE_CAPTIONING (needs IMAGE + TEXT) → 1.0
    - A dataset with [IMAGE] asked for IMAGE_CAPTIONING → 0.5 (missing TEXT)
    - A dataset with [TABULAR] asked for IMAGE_CLASSIFICATION → 0.0 (wrong modality)
    """
    required = TASK_MODALITY_MAP.get(query_task_type, frozenset())
    dataset_set = frozenset(dataset_modalities)

    # No requirements defined → fully compatible (catch-all tasks)
    if not required:
        return TaskCompatibility(
            task_type=query_task_type,
            compatibility_score=1.0,
            reason="No specific modality requirements for this task type.",
            required_modalities=required,
            matched_modalities=dataset_set,
            missing_modalities=frozenset(),
            is_compatible=True,
        )

    matched = required & dataset_set
    missing = required - dataset_set

    if not matched:
        score = 0.0
        reason = (
            f"Dataset modalities {[m.value for m in dataset_modalities]} "
            f"do not include any required modalities "
            f"{[m.value for m in required]} for {query_task_type.value}."
        )
    else:
        score = len(matched) / len(required)
        if missing:
            reason = (
                f"Partial match: has {[m.value for m in matched]} "
                f"but missing {[m.value for m in missing]} "
                f"for {query_task_type.value}."
            )
        else:
            reason = (
                f"Full modality match for {query_task_type.value}: "
                f"{[m.value for m in matched]}."
            )

    return TaskCompatibility(
        task_type=query_task_type,
        compatibility_score=round(score, 3),
        reason=reason,
        required_modalities=required,
        matched_modalities=matched,
        missing_modalities=missing,
        is_compatible=score >= 0.5,
    )