"""
datascout.query_understanding.task_detector
─────────────────────────────────────────────────────
Rule-based task and modality detection from normalized query tokens.

FIX (v3.1.0): Added agriculture/plant-disease/medical-imaging rules so that
"crop disease detector" scores IMAGE_CLASSIFICATION (0.95) instead of falling
through to OTHER. Without this fix, popularity wins (FineWeb 1M downloads beats
PlantVillage 5K downloads) because task_relevance gets a neutral 0.5 score.

BIGRAM NOTE: The cleaner joins bigrams with underscores.
"crop disease" → tokens ["crop","disease","crop_disease","disease_detector" ...]
Rules must use underscores to match bigrams correctly.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
from datascout.query_understanding.task_types import Modality, TaskType, normalize_task_type, normalize_modality

logger = logging.getLogger("datascout.query_understanding.task_detector")

MIN_TASK_CONFIDENCE:     float = 0.4
MIN_MODALITY_CONFIDENCE: float = 0.35

TASK_DETECTION_RULES: list[tuple[str, TaskType, float]] = [
    # ── Computer Vision ───────────────────────────────────────────────────────
    ("image_classification",          TaskType.IMAGE_CLASSIFICATION,      1.00),
    ("instance_segmentation",         TaskType.INSTANCE_SEGMENTATION,     1.00),
    ("semantic_segmentation",         TaskType.SEMANTIC_SEGMENTATION,     1.00),
    ("object_detection",              TaskType.OBJECT_DETECTION,          1.00),
    ("image_generation",              TaskType.IMAGE_GENERATION,          1.00),
    ("image_captioning",              TaskType.IMAGE_CAPTIONING,          0.95),
    ("depth_estimation",              TaskType.DEPTH_ESTIMATION,          0.95),
    ("pose_estimation",               TaskType.POSE_ESTIMATION,           0.95),
    ("yolo",                          TaskType.OBJECT_DETECTION,          0.95),
    ("coco",                          TaskType.OBJECT_DETECTION,          0.80),
    ("pascal_voc",                    TaskType.OBJECT_DETECTION,          0.80),
    ("bounding_box",                  TaskType.OBJECT_DETECTION,          0.85),
    ("bounding_boxes",                TaskType.OBJECT_DETECTION,          0.85),
    ("detect",                        TaskType.OBJECT_DETECTION,          0.70),
    ("detection",                     TaskType.OBJECT_DETECTION,          0.75),
    ("segmentation",                  TaskType.SEMANTIC_SEGMENTATION,     0.75),
    ("caption",                       TaskType.IMAGE_CAPTIONING,          0.70),
    ("captioning",                    TaskType.IMAGE_CAPTIONING,          0.80),
    ("generate_image",                TaskType.IMAGE_GENERATION,          0.90),
    ("image",                         TaskType.IMAGE_CLASSIFICATION,      0.50),

    # ── Agriculture / Plant disease (FIX: was missing, caused wrong rankings) ─
    ("crop_disease",                  TaskType.IMAGE_CLASSIFICATION,      0.95),
    ("plant_disease",                 TaskType.IMAGE_CLASSIFICATION,      0.95),
    ("leaf_disease",                  TaskType.IMAGE_CLASSIFICATION,      0.90),
    ("plant_pathology",               TaskType.IMAGE_CLASSIFICATION,      0.90),
    ("disease_detector",              TaskType.IMAGE_CLASSIFICATION,      0.90),
    ("disease_detection",             TaskType.IMAGE_CLASSIFICATION,      0.85),
    ("disease_classification",        TaskType.IMAGE_CLASSIFICATION,      0.85),
    ("pest_detection",                TaskType.IMAGE_CLASSIFICATION,      0.85),
    ("crop_detection",                TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("leaf_classification",           TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("weed_detection",                TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("plant_identification",          TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("plantvillage",                  TaskType.IMAGE_CLASSIFICATION,      0.90),
    ("plant_village",                 TaskType.IMAGE_CLASSIFICATION,      0.90),

    # ── Medical / Healthcare tabular (NOT image — these are CSV/tabular datasets) ─
    # These fire BINARY_CLASSIFICATION or REGRESSION, not IMAGE_CLASSIFICATION
    # "diabetes" → BINARY_CLASSIFICATION (predict disease yes/no)
    # "glucose" → REGRESSION (predict blood sugar level)
    ("diabetes",                      TaskType.BINARY_CLASSIFICATION,     0.95),
    ("diabetes_prediction",           TaskType.BINARY_CLASSIFICATION,     1.00),
    ("glucose",                       TaskType.BINARY_CLASSIFICATION,     0.80),
    ("clinical",                      TaskType.BINARY_CLASSIFICATION,     0.65),
    ("clinical_tabular",              TaskType.BINARY_CLASSIFICATION,     0.90),
    ("patient_records",               TaskType.BINARY_CLASSIFICATION,     0.90),
    ("patient",                       TaskType.BINARY_CLASSIFICATION,     0.60),
    ("ehr",                           TaskType.BINARY_CLASSIFICATION,     0.90),
    ("electronic_health",             TaskType.BINARY_CLASSIFICATION,     0.90),
    ("medical_records",               TaskType.BINARY_CLASSIFICATION,     0.90),
    ("heart_disease",                 TaskType.BINARY_CLASSIFICATION,     0.95),
    ("heart_failure",                 TaskType.BINARY_CLASSIFICATION,     0.95),
    ("stroke",                        TaskType.BINARY_CLASSIFICATION,     0.80),
    ("hypertension",                  TaskType.BINARY_CLASSIFICATION,     0.85),
    ("covid",                         TaskType.BINARY_CLASSIFICATION,     0.80),
    ("covid19",                       TaskType.BINARY_CLASSIFICATION,     0.85),
    ("pima",                          TaskType.BINARY_CLASSIFICATION,     0.90),  # Pima Indians diabetes
    ("mortality",                     TaskType.BINARY_CLASSIFICATION,     0.80),
    ("survival",                      TaskType.BINARY_CLASSIFICATION,     0.75),
    ("prognosis",                     TaskType.BINARY_CLASSIFICATION,     0.80),
    ("readmission",                   TaskType.BINARY_CLASSIFICATION,     0.85),
    ("sepsis",                        TaskType.BINARY_CLASSIFICATION,     0.85),
    ("alzheimer",                     TaskType.BINARY_CLASSIFICATION,     0.85),
    ("tumor",                         TaskType.BINARY_CLASSIFICATION,     0.80),
    ("breast_cancer",                 TaskType.BINARY_CLASSIFICATION,     0.95),
    ("lung_cancer",                   TaskType.BINARY_CLASSIFICATION,     0.95),
    ("house_price",                   TaskType.REGRESSION,                0.95),
    ("salary",                        TaskType.REGRESSION,                0.85),
    ("sales",                         TaskType.REGRESSION,                0.75),
    ("stock",                         TaskType.REGRESSION,                0.70),
    ("medical",                       TaskType.BINARY_CLASSIFICATION,     0.55),

    # ── Tabular modality keywords (help route to correct data type) ────────────
    ("tabular",                       TaskType.BINARY_CLASSIFICATION,     0.50),
    ("records",                       TaskType.BINARY_CLASSIFICATION,     0.45),

    # ── Medical imaging ───────────────────────────────────────────────────────
    ("medical_imaging",               TaskType.IMAGE_CLASSIFICATION,      0.95),
    ("chest_xray",                    TaskType.IMAGE_CLASSIFICATION,      0.95),
    ("xray_classification",           TaskType.IMAGE_CLASSIFICATION,      0.95),
    ("histology",                     TaskType.IMAGE_CLASSIFICATION,      0.85),
    ("skin_lesion",                   TaskType.IMAGE_CLASSIFICATION,      0.90),
    ("retinal",                       TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("fundus",                        TaskType.IMAGE_CLASSIFICATION,      0.80),
    ("dermoscopy",                    TaskType.IMAGE_CLASSIFICATION,      0.85),

    # ── NLP ───────────────────────────────────────────────────────────────────
    ("named_entity_recognition",      TaskType.NER,                       1.00),
    ("ner",                           TaskType.NER,                       1.00),
    ("sentiment_analysis",            TaskType.SENTIMENT_ANALYSIS,        1.00),
    ("text_classification",           TaskType.TEXT_CLASSIFICATION,       1.00),
    ("machine_translation",           TaskType.MACHINE_TRANSLATION,       1.00),
    ("question_answering",            TaskType.QUESTION_ANSWERING,        1.00),
    ("text_generation",               TaskType.TEXT_GENERATION,           1.00),
    ("language_modeling",             TaskType.LANGUAGE_MODELING,         1.00),
    ("summarization",                 TaskType.SUMMARIZATION,             1.00),
    ("qa",                            TaskType.QUESTION_ANSWERING,        0.95),
    ("translation",                   TaskType.MACHINE_TRANSLATION,       0.85),
    ("summarize",                     TaskType.SUMMARIZATION,             0.80),
    ("summary",                       TaskType.SUMMARIZATION,             0.70),
    ("sentiment",                     TaskType.SENTIMENT_ANALYSIS,        0.85),
    ("opinion",                       TaskType.SENTIMENT_ANALYSIS,        0.65),
    ("review",                        TaskType.SENTIMENT_ANALYSIS,        0.55),
    ("entity",                        TaskType.NER,                       0.65),
    ("entities",                      TaskType.NER,                       0.65),
    ("language_model",                TaskType.LANGUAGE_MODELING,         0.90),
    ("llm",                           TaskType.LANGUAGE_MODELING,         0.75),
    ("bert",                          TaskType.LANGUAGE_MODELING,         0.75),
    ("gpt",                           TaskType.LANGUAGE_MODELING,         0.75),
    ("transformer",                   TaskType.TEXT_CLASSIFICATION,       0.55),

    # ── Time Series ───────────────────────────────────────────────────────────
    ("time_series_forecasting",       TaskType.TIME_SERIES_FORECASTING,   1.00),
    ("anomaly_detection",             TaskType.ANOMALY_DETECTION,         1.00),
    ("forecasting",                   TaskType.TIME_SERIES_FORECASTING,   0.90),
    ("time_series",                   TaskType.TIME_SERIES_FORECASTING,   0.85),
    ("forecast",                      TaskType.TIME_SERIES_FORECASTING,   0.80),
    ("anomaly",                       TaskType.ANOMALY_DETECTION,         0.80),
    ("outlier",                       TaskType.ANOMALY_DETECTION,         0.75),
    ("intrusion",                     TaskType.ANOMALY_DETECTION,         0.65),

    # ── Classification / Regression ───────────────────────────────────────────
    ("binary_classification",         TaskType.BINARY_CLASSIFICATION,     1.00),
    ("multi_label_classification",    TaskType.MULTI_LABEL_CLASSIFICATION, 1.00),
    ("multi_label",                   TaskType.MULTI_LABEL_CLASSIFICATION, 0.90),
    ("regression",                    TaskType.REGRESSION,                1.00),
    ("classification",                TaskType.CLASSIFICATION,            0.90),
    ("classify",                      TaskType.CLASSIFICATION,            0.80),
    ("predict",                       TaskType.REGRESSION,                0.65),
    ("prediction",                    TaskType.REGRESSION,                0.65),
    ("price",                         TaskType.REGRESSION,                0.60),
    ("house_price",                   TaskType.REGRESSION,                0.85),
    ("housing",                       TaskType.REGRESSION,                0.60),
    ("churn",                         TaskType.BINARY_CLASSIFICATION,     0.75),
    ("fraud",                         TaskType.BINARY_CLASSIFICATION,     0.75),
    ("spam",                          TaskType.TEXT_CLASSIFICATION,       0.70),
    # "disease" alone → IMAGE_CLASSIFICATION (plant/medical image is dominant use case)
    # Tabular disease datasets (diabetes etc.) have stronger specific tokens
    ("disease",                       TaskType.IMAGE_CLASSIFICATION,      0.60),
    ("diagnosis",                     TaskType.BINARY_CLASSIFICATION,     0.65),
    ("cancer",                        TaskType.BINARY_CLASSIFICATION,     0.70),
    ("detector",                      TaskType.IMAGE_CLASSIFICATION,      0.55),

    # ── Unsupervised ──────────────────────────────────────────────────────────
    ("clustering",                    TaskType.CLUSTERING,                1.00),
    ("cluster",                       TaskType.CLUSTERING,                0.85),
    ("dimensionality_reduction",      TaskType.DIMENSIONALITY_REDUCTION,  1.00),
    ("pca",                           TaskType.DIMENSIONALITY_REDUCTION,  0.85),
    ("embedding",                     TaskType.DIMENSIONALITY_REDUCTION,  0.60),
    ("grouping",                      TaskType.CLUSTERING,                0.65),

    # ── Recommendation ────────────────────────────────────────────────────────
    ("recommendation",                TaskType.RECOMMENDATION,            1.00),
    ("recommender",                   TaskType.RECOMMENDATION,            0.95),
    ("collaborative_filtering",       TaskType.RECOMMENDATION,            0.90),
    ("rating",                        TaskType.RECOMMENDATION,            0.65),
    ("movie",                         TaskType.RECOMMENDATION,            0.55),

    # ── Audio ─────────────────────────────────────────────────────────────────
    ("speech_recognition",            TaskType.SPEECH_RECOGNITION,        1.00),
    ("audio_classification",          TaskType.AUDIO_CLASSIFICATION,      1.00),
    ("asr",                           TaskType.SPEECH_RECOGNITION,        0.95),
    ("speech",                        TaskType.SPEECH_RECOGNITION,        0.75),
    ("speaker",                       TaskType.SPEECH_RECOGNITION,        0.65),
    ("music",                         TaskType.AUDIO_CLASSIFICATION,      0.60),

    # ── Multimodal ────────────────────────────────────────────────────────────
    ("visual_question_answering",     TaskType.VISUAL_QUESTION_ANSWERING, 1.00),
    ("vqa",                           TaskType.VISUAL_QUESTION_ANSWERING, 0.95),
    ("document_understanding",        TaskType.DOCUMENT_UNDERSTANDING,    1.00),
    ("ocr",                           TaskType.DOCUMENT_UNDERSTANDING,    0.85),
    ("document",                      TaskType.DOCUMENT_UNDERSTANDING,    0.55),
    ("invoice",                       TaskType.DOCUMENT_UNDERSTANDING,    0.65),
    ("receipt",                       TaskType.DOCUMENT_UNDERSTANDING,    0.65),
]

MODALITY_DETECTION_RULES: list[tuple[str, Modality, float]] = [
    ("image",           Modality.IMAGE,        0.90),
    ("images",          Modality.IMAGE,        0.90),
    ("photo",           Modality.IMAGE,        0.85),
    ("photograph",      Modality.IMAGE,        0.85),
    ("picture",         Modality.IMAGE,        0.85),
    ("visual",          Modality.IMAGE,        0.75),
    ("computer_vision", Modality.IMAGE,        0.90),
    ("cv",              Modality.IMAGE,        0.70),
    # Agriculture tokens → IMAGE modality
    ("crop",            Modality.IMAGE,        0.65),
    ("leaf",            Modality.IMAGE,        0.70),
    ("plant",           Modality.IMAGE,        0.65),
    ("farm",            Modality.IMAGE,        0.55),
    ("video",           Modality.VIDEO,        0.95),
    ("videos",          Modality.VIDEO,        0.95),
    ("text",            Modality.TEXT,         0.85),
    ("nlp",             Modality.TEXT,         0.90),
    ("language",        Modality.TEXT,         0.70),
    ("corpus",          Modality.TEXT,         0.85),
    ("document",        Modality.TEXT,         0.65),
    ("review",          Modality.TEXT,         0.60),
    ("tweet",           Modality.TEXT,         0.75),
    ("sentence",        Modality.TEXT,         0.70),
    ("paragraph",       Modality.TEXT,         0.70),
    ("audio",           Modality.AUDIO,        0.95),
    ("sound",           Modality.AUDIO,        0.85),
    ("speech",          Modality.AUDIO,        0.80),
    ("music",           Modality.AUDIO,        0.75),
    ("tabular",         Modality.TABULAR,      0.95),
    ("clinical",        Modality.TABULAR,      0.80),
    ("patient",         Modality.TABULAR,      0.70),
    ("medical",         Modality.TABULAR,      0.65),
    ("diabetes",        Modality.TABULAR,      0.80),
    ("glucose",         Modality.TABULAR,      0.75),
    ("records",         Modality.TABULAR,      0.65),
    ("ehr",             Modality.TABULAR,      0.85),
    ("table",           Modality.TABULAR,      0.80),
    ("csv",             Modality.TABULAR,      0.85),
    ("structured",      Modality.TABULAR,      0.70),
    ("spreadsheet",     Modality.TABULAR,      0.80),
    ("numerical",       Modality.TABULAR,      0.65),
    ("time_series",     Modality.TIME_SERIES,  0.95),
    ("timeseries",      Modality.TIME_SERIES,  0.95),
    ("temporal",        Modality.TIME_SERIES,  0.80),
    ("sequential",      Modality.TIME_SERIES,  0.70),
    ("graph",           Modality.GRAPH,        0.90),
    ("network",         Modality.GRAPH,        0.70),
    ("node",            Modality.GRAPH,        0.65),
    ("multimodal",      Modality.MULTIMODAL,   0.95),
    ("multi_modal",     Modality.MULTIMODAL,   0.95),
]


@dataclass
class DetectionResult:
    task_type: TaskType
    task_confidence: float
    modality: Modality
    modality_confidence: float
    task_score_map: dict[str, float]
    modality_score_map: dict[str, float]
    matched_task_tokens: list[str]
    matched_modality_tokens: list[str]

    @property
    def is_task_confident(self) -> bool:
        return self.task_confidence >= MIN_TASK_CONFIDENCE

    @property
    def is_modality_confident(self) -> bool:
        return self.modality_confidence >= MIN_MODALITY_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "task_confidence": self.task_confidence,
            "modality": self.modality.value,
            "modality_confidence": self.modality_confidence,
            "is_task_confident": self.is_task_confident,
            "is_modality_confident": self.is_modality_confident,
            "matched_task_tokens": self.matched_task_tokens,
            "matched_modality_tokens": self.matched_modality_tokens,
        }


class TaskDetector:
    def detect(self, tokens: list[str]) -> DetectionResult:
        if not tokens:
            return self._empty_result()
        token_set = set(tokens)
        task_scores, task_fired    = self._score_rules(token_set, TASK_DETECTION_RULES)
        modality_scores, mod_fired = self._score_rules(token_set, MODALITY_DETECTION_RULES)
        task_type, task_conf       = self._pick_winner(task_scores, TaskType, MIN_TASK_CONFIDENCE)
        modality, modality_conf    = self._pick_winner(modality_scores, Modality, MIN_MODALITY_CONFIDENCE)
        logger.debug("task_detection_result", extra={
            "task": task_type.value, "task_confidence": task_conf,
            "modality": modality.value, "modality_confidence": modality_conf,
            "top_task_scores": dict(sorted(task_scores.items(), key=lambda x: -x[1])[:3]),
        })
        return DetectionResult(
            task_type=task_type, task_confidence=task_conf,
            modality=modality, modality_confidence=modality_conf,
            task_score_map={k.value: round(v, 3) for k, v in task_scores.items()},
            modality_score_map={k.value: round(v, 3) for k, v in modality_scores.items()},
            matched_task_tokens=task_fired, matched_modality_tokens=mod_fired,
        )

    @staticmethod
    def _score_rules(token_set: set[str], rules: list[tuple]) -> tuple[dict, list[str]]:
        scores: dict = {}
        fired: list[str] = []
        for keyword, label, weight in rules:
            if keyword in token_set:
                scores[label] = scores.get(label, 0.0) + weight
                if keyword not in fired:
                    fired.append(keyword)
        return scores, fired

    @staticmethod
    def _pick_winner(scores: dict, enum_class: type, min_confidence: float) -> tuple:
        unknown = enum_class.UNKNOWN  # type: ignore[attr-defined]
        if not scores:
            return unknown, 0.0
        winner = max(scores, key=lambda k: scores[k])
        normalized = min(scores[winner] / 1.5, 1.0)
        if normalized < min_confidence:
            return unknown, round(normalized, 3)
        return winner, round(normalized, 3)

    @staticmethod
    def _empty_result() -> DetectionResult:
        return DetectionResult(
            task_type=TaskType.UNKNOWN, task_confidence=0.0,
            modality=Modality.UNKNOWN, modality_confidence=0.0,
            task_score_map={}, modality_score_map={},
            matched_task_tokens=[], matched_modality_tokens=[],
        )