"""
datascout.evaluation.metrics.task_match
────────────────────────────────────────
Task alignment evaluator: how well a dataset serves the queried ML task.

Evaluates:
  - Exact task type match in dataset.task_types
  - Task family soft match (e.g. CLASSIFICATION matches BINARY_CLASSIFICATION)
  - Modality-task compatibility (TASK_MODALITY_MAP)
  - Domain alignment for query intent
  - Annotation-task fit (schema info for supervised tasks)

Returns a TaskMatchResult with score + explainable breakdown.
Deterministic — no LLM involvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datascout.contracts import RawDataset
from datascout.contracts.task_types import Modality, TaskType, compute_task_compatibility
from datascout.query_understanding.task_types import are_in_same_family


# Scores by match type
EXACT_MATCH_SCORE  = 1.00
FAMILY_MATCH_SCORE = 0.70
MODALITY_COMPAT_THRESHOLD = 0.50  # Min modality compat to be considered aligned

# Bonus for having annotation info when task requires it
ANNOTATION_BONUS = 0.05

# Tasks that strongly benefit from schema info (supervised tasks)
_SUPERVISED_TASKS = frozenset({
    TaskType.CLASSIFICATION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTI_LABEL_CLASSIFICATION,
    TaskType.REGRESSION,
    TaskType.IMAGE_CLASSIFICATION,
    TaskType.OBJECT_DETECTION,
    TaskType.SEMANTIC_SEGMENTATION,
    TaskType.NER,
    TaskType.SENTIMENT_ANALYSIS,
    TaskType.TEXT_CLASSIFICATION,
    TaskType.TIME_SERIES_FORECASTING,
    TaskType.ANOMALY_DETECTION,
})


@dataclass
class TaskMatchResult:
    score: float                         # 0.0–1.0
    match_type: str                      # "exact", "family", "modality", "unknown", "none"
    matched_task: Optional[TaskType]     # Which task type caused the match
    modality_compat_score: float         # Modality compatibility score
    annotation_bonus: float              # Bonus for annotation-ready datasets
    query_task: Optional[TaskType]
    query_modality: Optional[Modality]
    explanation: str


def score_task_match(
    dataset: RawDataset,
    query_task: Optional[TaskType],
    query_modality: Optional[Modality] = None,
) -> TaskMatchResult:
    """
    Score how well a dataset serves the queried task deterministically.

    Priority order:
      1. Exact task type match → 1.0
      2. Task family match → 0.70
      3. Modality compatibility → compatibility_score
      4. No query task → 0.5 (neutral)
      5. No task or modality info → 0.4 (slight penalty)

    Never raises.
    """
    # No task specified — neutral scoring
    if not query_task or query_task == TaskType.OTHER:
        return TaskMatchResult(
            score=0.5,
            match_type="unknown",
            matched_task=None,
            modality_compat_score=0.5,
            annotation_bonus=0.0,
            query_task=query_task,
            query_modality=query_modality,
            explanation="No specific task type queried — neutral task alignment score.",
        )

    # Exact task match
    if dataset.task_types and query_task in dataset.task_types:
        bonus = _compute_annotation_bonus(dataset, query_task)
        score = round(min(EXACT_MATCH_SCORE + bonus, 1.0), 4)
        return TaskMatchResult(
            score=score,
            match_type="exact",
            matched_task=query_task,
            modality_compat_score=1.0,
            annotation_bonus=bonus,
            query_task=query_task,
            query_modality=query_modality,
            explanation=(
                f"Exact task match: dataset supports '{query_task.value}'. "
                f"Strong alignment with query intent."
            ),
        )

    # Family match (e.g. CLASSIFICATION matches BINARY_CLASSIFICATION)
    if dataset.task_types:
        for dt in dataset.task_types:
            if are_in_same_family(query_task, dt):
                bonus = _compute_annotation_bonus(dataset, query_task)
                score = round(min(FAMILY_MATCH_SCORE + bonus, 1.0), 4)
                return TaskMatchResult(
                    score=score,
                    match_type="family",
                    matched_task=dt,
                    modality_compat_score=0.7,
                    annotation_bonus=bonus,
                    query_task=query_task,
                    query_modality=query_modality,
                    explanation=(
                        f"Task family match: '{dt.value}' is in the same family as "
                        f"'{query_task.value}'. Good alignment."
                    ),
                )

    # Modality-based compatibility
    if dataset.modalities:
        compat = compute_task_compatibility(
            query_task_type=query_task,
            dataset_modalities=list(dataset.modalities),
        )
        if compat.is_compatible:
            bonus = _compute_annotation_bonus(dataset, query_task)
            score = round(min(compat.compatibility_score + bonus, 1.0), 4)
            return TaskMatchResult(
                score=score,
                match_type="modality",
                matched_task=None,
                modality_compat_score=compat.compatibility_score,
                annotation_bonus=bonus,
                query_task=query_task,
                query_modality=query_modality,
                explanation=compat.reason,
            )
        else:
            return TaskMatchResult(
                score=0.1,
                match_type="none",
                matched_task=None,
                modality_compat_score=compat.compatibility_score,
                annotation_bonus=0.0,
                query_task=query_task,
                query_modality=query_modality,
                explanation=f"Modality incompatible with task '{query_task.value}'. {compat.reason}",
            )

    # No task or modality info — slight penalty
    return TaskMatchResult(
        score=0.4,
        match_type="none",
        matched_task=None,
        modality_compat_score=0.0,
        annotation_bonus=0.0,
        query_task=query_task,
        query_modality=query_modality,
        explanation=(
            f"Dataset has no task type or modality information — "
            f"cannot verify alignment with '{query_task.value}'."
        ),
    )


def _compute_annotation_bonus(dataset: RawDataset, task: TaskType) -> float:
    """
    Small bonus for datasets that have schema info for supervised tasks.
    Schema info (column names) signals annotation readiness.
    """
    if task in _SUPERVISED_TASKS and dataset.has_schema_info:
        return ANNOTATION_BONUS
    return 0.0