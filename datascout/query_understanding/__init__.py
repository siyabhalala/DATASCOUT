"""
datascout.query_understanding
─────────────────────────────────────────────────────
Public surface for the query understanding package.
Entry point: QueryParser.parse(raw_text) → QueryParseResult
"""

from .cleaner import QueryCleaner, CleanedQuery, STOP_WORDS, SYNONYM_MAP
from .task_types import (
    TaskType,
    Modality,
    TaskCompatibility,
    TASK_MODALITY_MAP,
    TASK_FAMILIES,
    normalize_task_type,
    normalize_modality,
    compute_task_compatibility,
    get_task_family,
    are_in_same_family,
)
from .task_detector import TaskDetector, DetectionResult
from .parser import QueryParser, QueryParseResult, ConfidenceLevel

__all__ = [
    # Main entry points
    "QueryParser",
    "QueryParseResult",
    "ConfidenceLevel",
    # Task types
    "TaskType",
    "Modality",
    "TaskCompatibility",
    "TASK_MODALITY_MAP",
    "TASK_FAMILIES",
    "normalize_task_type",
    "normalize_modality",
    "compute_task_compatibility",
    "get_task_family",
    "are_in_same_family",
    # Cleaner
    "QueryCleaner",
    "CleanedQuery",
    "STOP_WORDS",
    "SYNONYM_MAP",
    # Detector
    "TaskDetector",
    "DetectionResult",
]