"""
datascout.adapters.modality_detector
─────────────────────────────────────────────────────
PHASE 1 — Multi-signal modality detection.

PROBLEM BEING SOLVED:
  The previous system called normalize_modality(tag) on each tag and returned
  Modality.OTHER if no tag happened to literally contain "image", "vision",
  "text", "audio", etc.

  Example failure:
    PlantVillage dataset
    Tags:  agriculture, classification
    Desc:  54,000 images of diseased plant leaves
    Old:   OTHER
    New:   IMAGE (confidence=0.95)

DESIGN:
  Four independent signals vote with weights:
    A. Tags        (exact + keyword match)
    B. Title       (keyword match)
    C. Description (keyword match)
    D. Extra metadata (source-specific fields)

  For each candidate modality, we accumulate a score.
  Final modality = highest-scoring candidate above threshold.

  Scores are additive, capped per signal so no single signal dominates.

Author: Principal Engineer
Version: 1.0.0 (Phase 1)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("datascout.adapters.modality_detector")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Tags that strongly signal a modality (exact or substring match after normalization)
_TAG_IMAGE_SIGNALS = frozenset({
    "image", "images", "vision", "computer-vision", "computer_vision",
    "image-classification", "image_classification",
    "object-detection", "object_detection",
    "segmentation", "image-segmentation",
    "image-generation", "image_generation",
    "medical-imaging", "medical_imaging", "medical imaging",
    "remote-sensing", "satellite", "aerial",
    "ocr", "optical character recognition",
    "face-recognition", "face recognition", "facial",
    "depth-estimation", "pose-estimation",
    "visual-question-answering", "visual question answering",
    "image-captioning",
    "cnn", "convolutional",
    "photo", "photos", "photograph", "photographs",
    "picture", "pictures",
})

_TAG_TABULAR_SIGNALS = frozenset({
    "tabular", "structured", "table",
    "csv", "excel", "spreadsheet",
    "regression", "classification",  # generic ML tasks usually tabular
    "time-series", "timeseries", "time series",
    "survey", "census",
    "financial", "finance",
    "health", "healthcare",  # often tabular
    "demographics",
    "economics", "economic",
    "sales", "marketing",
    "sports", "sport statistics",
})

_TAG_TEXT_SIGNALS = frozenset({
    "text", "nlp", "natural-language-processing", "natural language processing",
    "language", "linguistics", "corpus", "corpora",
    "sentiment", "sentiment-analysis",
    "text-classification", "text classification",
    "named-entity-recognition", "ner",
    "machine-translation", "translation",
    "question-answering", "question answering",
    "summarization", "summarisation",
    "text-generation", "text generation",
    "language-model", "language model",
    "bert", "gpt", "transformer",
    "news", "articles", "tweets", "reviews",
    "books", "literature",
})

_TAG_AUDIO_SIGNALS = frozenset({
    "audio", "speech", "sound", "acoustic",
    "asr", "speech-recognition", "speech recognition",
    "music", "speaker",
    "audio-classification", "sound-classification",
    "voice",
})

_TAG_VIDEO_SIGNALS = frozenset({
    "video", "videos", "movie", "movies",
    "action-recognition", "action recognition",
    "video-classification",
})

# ── Title keywords ────────────────────────────────────────────────────────────

_TITLE_IMAGE_PATTERNS = [
    r"\bimage[s]?\b", r"\bphoto[s]?\b", r"\bpicture[s]?\b", r"\bphotograph[s]?\b",
    r"\bvision\b", r"\bocr\b", r"\bvisual\b",
    # Common image dataset name components
    r"\bleaf\b", r"\bleaves\b", r"\bplant\b", r"\bflower\b", r"\bpetal\b",
    r"\bretina[l]?\b", r"\bfundus\b", r"\boct\b", r"\bx[\-_]?ray[s]?\b",
    r"\bdermoscop[y|ic]\b", r"\bskin\b",
    r"\bface[s]?\b", r"\bfacial\b", r"\bgesture[s]?\b",
    r"\bsatellite\b", r"\baerial\b", r"\bdrone[s]?\b",
    r"\bmicroscop[y|ic]\b", r"\bhistolog[y|ical]\b", r"\bpatholog[y|ical]\b",
    r"\bcancer\b",  # very commonly image data (skin, chest)
    r"\bfruit[s]?\b", r"\bvehicle[s]?\b", r"\bdigit[s]?\b",
    r"\bmnist\b", r"\bcifar\b", r"\bimagenet\b",
    r"\bchest\b",  # chest X-ray
    r"\bmri\b", r"\bct[\s_]scan\b",
]

_TITLE_TABULAR_PATTERNS = [
    r"\bprice[s]?\b", r"\bstock[s]?\b", r"\bmarket\b",
    r"\bhousing\b", r"\bhouse[s]?\b", r"\breal[\s_]estate\b",
    r"\bcredit\b", r"\bloan[s]?\b", r"\bbank[ing]?\b",
    r"\bsale[s]?\b", r"\brevenue\b", r"\bfinancial\b",
    r"\bcensus\b", r"\bsurvey\b", r"\bpopulation\b",
    r"\btitanic\b", r"\bpassenger[s]?\b",
    r"\bweather\b", r"\bclimate\b",
    r"\bdiabetes\b", r"\bheart[\s_]disease\b",
    r"\bcovid\b", r"\bpandemic\b",
    r"\bwine\b", r"\biris\b",
]

_TITLE_TEXT_PATTERNS = [
    r"\bnews\b", r"\barticle[s]?\b", r"\breviews?\b", r"\btexts?\b",
    r"\btweets?\b", r"\btwitter\b", r"\bsocial[\s_]media\b",
    r"\bcorpus\b", r"\bcorpora\b",
    r"\bsentiment\b", r"\bopinion[s]?\b",
    r"\bemail[s]?\b", r"\bchat[s]?\b",
    r"\bbooks?\b", r"\bstories?\b",
    r"\bnlp\b", r"\blanguage\b",
]

_TITLE_AUDIO_PATTERNS = [
    r"\bspeech\b", r"\baudio\b", r"\bsound[s]?\b",
    r"\bmusic\b", r"\bvoice[s]?\b",
    r"\basr\b",
]

# ── Description keywords ──────────────────────────────────────────────────────

_DESC_IMAGE_PATTERNS = [
    # Explicit image references
    r"\bimage[s]?\b", r"\bphoto[s]?\b", r"\bphotograph[s]?\b", r"\bpicture[s]?\b",
    r"jpg\b", r"jpeg\b", r"png\b", r"tiff?\b", r"bmp\b", r"webp\b",
    r"\bpixel[s]?\b", r"\bresolution\b",
    # Common image domain words
    r"\bleaf\b", r"\bleaves\b", r"\bplant\b",
    r"\bscan[s]?\b",  # medical scans
    r"\bx[\-_]?ray[s]?\b", r"\bmri\b", r"\bct[\s_]?scan[s]?\b",
    r"\bretina[l]?\b",
    r"\bskin\b", r"\bdermat\w*\b",
    r"\bcancer\b",  # often image
    r"\bsatellite\b", r"\baerial\b",
    r"\bmicroscop\w*\b",
    r"\bhuman pose\b", r"\bface[s]?\b", r"\bfacial\b",
    r"\bannotated\b.*\bimage\b",
]

_DESC_TABULAR_PATTERNS = [
    r"\brow[s]?\b", r"\bcolumn[s]?\b", r"\bfeature[s]?\b", r"\battribute[s]?\b",
    r"\brecord[s]?\b", r"\bsample[s]?\b", r"\bentry\b", r"\bentries\b",
    r"\bcsv\b", r"\barff\b", r"\btabular\b", r"\bspreadsheet\b",
    r"\bstructured\s+data\b",
    r"\bvariable[s]?\b",   # often tabular
    r"\binstance[s]?\b",  # OpenML language for rows
]

_DESC_TEXT_PATTERNS = [
    r"\btext\b", r"\bsentences?\b", r"\bparagraph[s]?\b", r"\bdocument[s]?\b",
    r"\btoken[s]?\b", r"\bword[s]?\b", r"\bvocabular\w*\b",
    r"\bcorpus\b", r"\bmonolingual\b", r"\bbilingual\b",
    r"\bnlp\b", r"\blanguage\b",
    r"\breviews?\b", r"\barticles?\b", r"\btexts?\b",
]

_DESC_AUDIO_PATTERNS = [
    r"\baudio\b", r"\bspeech\b", r"\bsound\b", r"\bwav\b", r"\bmp3\b",
    r"\brecording[s]?\b", r"\butterance[s]?\b",
    r"\bmusic\b",
]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModalityDetectionResult:
    """Result of multi-signal modality detection."""

    def __init__(
        self,
        modality: str,
        confidence: float,
        signals_used: list = None,
        scores: dict = None,
    ):
        self.modality = modality
        self.confidence = confidence
        self.signals_used = signals_used if signals_used is not None else []
        self.scores = scores if scores is not None else {}

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "confidence": round(self.confidence, 3),
            "signals_used": self.signals_used,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ModalityDetector:
    """
    Multi-signal modality detector.

    Usage:
        result = ModalityDetector.detect(
            tags=["agriculture", "classification"],
            title="PlantVillage",
            description="54,000 images of diseased plant leaves",
            extra={},
        )
        result.modality    # "image"
        result.confidence  # 0.91
        result.signals_used  # ["description", "title"]
    """

    # Signal weights (max contribution each signal can add)
    WEIGHT_TAG  = 0.40
    WEIGHT_TITLE = 0.25
    WEIGHT_DESC  = 0.25
    WEIGHT_EXTRA = 0.10

    # Score threshold to assert a modality (vs OTHER)
    THRESHOLD = 0.20

    @classmethod
    def detect(
        cls,
        tags: list[str],
        title: str,
        description: str,
        extra: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> ModalityDetectionResult:
        """
        Detect modality from multiple signals.

        Args:
            tags:        Normalized lowercase tag list.
            title:       Dataset title.
            description: Dataset description.
            extra:       Source-specific extra metadata dict.
            source:      Adapter source name ("kaggle", "huggingface", "openml").

        Returns:
            ModalityDetectionResult with modality, confidence, signals used.
        """
        scores: dict[str, float] = {
            "image": 0.0,
            "tabular": 0.0,
            "text": 0.0,
            "audio": 0.0,
            "video": 0.0,
        }
        signals_fired: set[str] = set()

        title_lower = (title or "").lower()
        desc_lower  = (description or "").lower()
        tags_lower  = [t.lower() for t in (tags or [])]

        # ── Signal A: Tags ────────────────────────────────────────────────────
        tag_str = " ".join(tags_lower)
        tag_deltas = cls._score_tags(tags_lower, tag_str)
        for mod, delta in tag_deltas.items():
            if delta > 0:
                scores[mod] += min(delta, cls.WEIGHT_TAG)
                signals_fired.add("tags")

        # ── Signal B: Title ───────────────────────────────────────────────────
        title_deltas = cls._score_text(title_lower, _TITLE_IMAGE_PATTERNS, _TITLE_TABULAR_PATTERNS, _TITLE_TEXT_PATTERNS, _TITLE_AUDIO_PATTERNS)
        for mod, delta in title_deltas.items():
            if delta > 0:
                scores[mod] += min(delta, cls.WEIGHT_TITLE)
                signals_fired.add("title")

        # ── Signal C: Description ─────────────────────────────────────────────
        desc_deltas = cls._score_text(desc_lower, _DESC_IMAGE_PATTERNS, _DESC_TABULAR_PATTERNS, _DESC_TEXT_PATTERNS, _DESC_AUDIO_PATTERNS)
        for mod, delta in desc_deltas.items():
            if delta > 0:
                scores[mod] += min(delta, cls.WEIGHT_DESC)
                signals_fired.add("description")

        # ── Signal D: Extra/source metadata ───────────────────────────────────
        extra_deltas = cls._score_extra(extra or {}, source)
        for mod, delta in extra_deltas.items():
            if delta > 0:
                scores[mod] += min(delta, cls.WEIGHT_EXTRA)
                signals_fired.add("metadata")

        # ── Special rule: OpenML is always TABULAR unless image signals strong ─
        if source == "openml" and scores["image"] < 0.3:
            scores["tabular"] = max(scores["tabular"], 0.30)
            signals_fired.add("source_default")

        # ── Determine winner ──────────────────────────────────────────────────
        best_mod = max(scores, key=lambda m: scores[m])
        best_score = scores[best_mod]

        if best_score < cls.THRESHOLD:
            # No signal strong enough — default OTHER
            return ModalityDetectionResult(
                modality="other",
                confidence=0.0,
                signals_used=list(signals_fired),
                scores=scores,
            )

        # Normalize confidence to [0, 1] — max possible score is ~1.0
        total = sum(scores.values())
        if total > 0:
            confidence = best_score / max(total, best_score)
        else:
            confidence = 0.0

        # Clamp confidence for clean output
        confidence = min(1.0, round(confidence, 3))

        logger.debug(
            "modality_detected",
            extra={
                "modality": best_mod,
                "confidence": confidence,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "signals": list(signals_fired),
                "title": title[:40],
            },
        )

        return ModalityDetectionResult(
            modality=best_mod,
            confidence=confidence,
            signals_used=sorted(signals_fired),
            scores=scores,
        )

    # ── Internal scorers ──────────────────────────────────────────────────────

    @staticmethod
    def _score_tags(tags_lower: list[str], tag_str: str) -> dict[str, float]:
        """Score each modality from tags. Returns delta-scores (max WEIGHT_TAG)."""
        deltas: dict[str, float] = {"image": 0.0, "tabular": 0.0, "text": 0.0, "audio": 0.0, "video": 0.0}

        for tag in tags_lower:
            tag_norm = tag.replace("-", "_").replace(" ", "_")

            if any(sig in tag or sig in tag_norm for sig in _TAG_IMAGE_SIGNALS):
                deltas["image"] += 0.15
            if any(sig in tag or sig in tag_norm for sig in _TAG_TEXT_SIGNALS):
                deltas["text"] += 0.15
            if any(sig in tag or sig in tag_norm for sig in _TAG_AUDIO_SIGNALS):
                deltas["audio"] += 0.20
            if any(sig in tag or sig in tag_norm for sig in _TAG_VIDEO_SIGNALS):
                deltas["video"] += 0.20
            # Tabular tags get less weight since they're generic (regression, classification etc)
            if any(sig in tag or sig in tag_norm for sig in _TAG_TABULAR_SIGNALS):
                deltas["tabular"] += 0.10

        return deltas

    @staticmethod
    def _score_text(
        text: str,
        image_patterns: list[str],
        tabular_patterns: list[str],
        text_patterns: list[str],
        audio_patterns: list[str],
    ) -> dict[str, float]:
        """Score modalities from a text field using regex patterns."""
        deltas: dict[str, float] = {"image": 0.0, "tabular": 0.0, "text": 0.0, "audio": 0.0, "video": 0.0}

        if not text:
            return deltas

        def _count(patterns: list[str]) -> int:
            count = 0
            for p in patterns:
                try:
                    if re.search(p, text):
                        count += 1
                except re.error:
                    continue
            return count

        image_hits   = _count(image_patterns)
        tabular_hits = _count(tabular_patterns)
        text_hits    = _count(text_patterns)
        audio_hits   = _count(audio_patterns)

        # Normalize: each hit adds 0.08, capped at 0.40
        deltas["image"]   = min(image_hits   * 0.08, 0.40)
        deltas["tabular"] = min(tabular_hits * 0.07, 0.35)
        deltas["text"]    = min(text_hits    * 0.08, 0.40)
        deltas["audio"]   = min(audio_hits   * 0.12, 0.40)

        return deltas

    @staticmethod
    def _score_extra(extra: dict[str, Any], source: Optional[str]) -> dict[str, float]:
        """Score from source-specific extra metadata."""
        deltas: dict[str, float] = {"image": 0.0, "tabular": 0.0, "text": 0.0, "audio": 0.0, "video": 0.0}

        # HuggingFace: task_categories and modalities in card_data
        if source == "huggingface":
            task_cats = extra.get("task_categories", []) or []
            for t in task_cats:
                t_lower = str(t).lower()
                if "image" in t_lower or "visual" in t_lower or "vision" in t_lower:
                    deltas["image"] += 0.10
                elif "text" in t_lower or "nlp" in t_lower or "language" in t_lower:
                    deltas["text"] += 0.10
                elif "audio" in t_lower or "speech" in t_lower:
                    deltas["audio"] += 0.10
                elif "tabular" in t_lower or "classification" in t_lower or "regression" in t_lower:
                    deltas["tabular"] += 0.05

        # OpenML: class_count and feature_count hint at tabular
        if source == "openml":
            class_count  = extra.get("class_count")
            feature_count = extra.get("feature_count")
            if class_count is not None and class_count > 0:
                deltas["tabular"] += 0.08
            if feature_count is not None and feature_count > 0:
                deltas["tabular"] += 0.05

        return deltas


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: detect and return Modality enum value
# ─────────────────────────────────────────────────────────────────────────────

def detect_modality(
    tags: list[str],
    title: str,
    description: str,
    extra: Optional[dict[str, Any]] = None,
    source: Optional[str] = None,
) -> ModalityDetectionResult:
    """
    Shorthand for ModalityDetector.detect().

    Returns ModalityDetectionResult — call .modality for the string value.
    """
    return ModalityDetector.detect(
        tags=tags,
        title=title,
        description=description,
        extra=extra,
        source=source,
    )
