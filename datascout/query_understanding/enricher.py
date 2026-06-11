"""
datascout.query_understanding.enricher
────────────────────────────────────────────────────────────────────────────
Query semantic enrichment — expands a cleaned query with domain synonyms,
task-specific terminology, and provider routing hints.

WHY this module exists:
  The cleaner+detector pipeline is literal: it fires on tokens present in
  the query. But "speech recognition" should also retrieve ASR, audio
  datasets, and transcription-focused corpora. "Crop disease" should pull
  plant pathology, image classification, and agriculture CV datasets.

  Without enrichment, users who write "Hindi NLP" get nothing if their
  query doesn't contain the exact tokens BM25 looks for in dataset titles.

FIX (v2.0.0):
  The original enricher broke on domain=None after first match (single `break`).
  Multi-domain queries like "speech recognition Hindi" only triggered the
  first matched domain and ignored "Hindi"'s Indic NLP signals entirely.

  Fix: remove the early break. Accumulate all matching domains. Deduplicate
  expansion terms. Merge provider weights by taking max boost / min penalty.

Author: DataScout Engineering
Version: 2.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN SYNONYM EXPANSION TABLE
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_EXPANSION: dict[str, list[str]] = {
    # Speech / ASR
    "speech recognition": ["asr", "automatic speech recognition", "spoken language", "transcription", "audio"],
    "speech to text":     ["asr", "speech recognition", "transcription", "audio"],
    "speech":             ["asr", "speech-to-text", "spoken", "transcription", "audio"],
    "asr":                ["speech recognition", "automatic speech recognition", "spoken", "audio transcription"],
    "transcription":      ["speech recognition", "asr", "spoken", "audio"],
    "voice":              ["speech", "audio", "spoken", "asr"],

    # Indian / Indic NLP
    "hindi":              ["indic", "indian language", "devanagari", "hindi nlp", "indian"],
    "gujarati":           ["indic", "indian language", "gujarati nlp", "indian"],
    "tamil":              ["indic", "dravidian", "south indian", "tamil nlp"],
    "marathi":            ["indic", "devanagari", "indian language", "marathi nlp"],
    "bengali":            ["indic", "bangla", "south asian", "bengali nlp"],
    "punjabi":            ["indic", "gurmukhi", "indian language", "punjabi nlp"],
    "telugu":             ["indic", "dravidian", "south indian", "telugu nlp"],
    "indic":              ["multilingual", "indian language", "hindi", "low-resource"],

    # Agriculture / Plant
    "crop disease":       ["plant pathology", "plant disease", "leaf disease", "agriculture", "disease classification"],
    "plant disease":      ["crop disease", "plant pathology", "leaf", "agriculture", "fungal disease"],
    "plant pathology":    ["crop disease", "plant disease", "leaf classification", "agriculture"],
    "agriculture":        ["crop", "plant", "farming", "agri", "agricultural"],
    "leaf disease":       ["plant disease", "crop disease", "plant pathology", "agriculture"],

    # Medical / Clinical
    "medical imaging":    ["radiology", "xray", "mri", "ct scan", "histology", "pathology"],
    "chest xray":         ["radiology", "pneumonia", "medical imaging", "x-ray", "thorax"],
    "ecg":                ["electrocardiogram", "cardiac", "heart", "arrhythmia"],
    "ehr":                ["electronic health record", "clinical", "patient", "medical"],
    "cancer":             ["tumor", "pathology", "medical imaging", "oncology", "histology"],
    "diabetes":           ["clinical", "tabular", "patient", "glucose", "medical"],

    # Computer Vision
    "object detection":   ["bounding box", "yolo", "detection", "localization"],
    "image segmentation": ["segmentation", "semantic segmentation", "instance segmentation", "mask"],
    "depth estimation":   ["monocular depth", "stereo", "3d reconstruction", "lidar"],
    "face recognition":   ["facial", "biometric", "identity", "face detection"],
    "pothole":            ["road damage", "infrastructure", "computer vision", "image classification"],

    # NLP / Text
    "sentiment analysis": ["sentiment", "opinion mining", "emotion", "polarity", "review"],
    "ner":                ["named entity recognition", "entity extraction", "information extraction"],
    "machine translation":["translation", "bilingual", "parallel corpus", "multilingual"],
    "question answering": ["reading comprehension", "extractive qa", "squad", "mcq"],
    "summarization":      ["abstractive", "extractive", "document summarization", "text"],
    "llm training":       ["instruction tuning", "rlhf", "finetuning", "pretraining", "language model"],

    # Tabular / Structured
    "fraud detection":    ["fraud", "anomaly", "imbalanced", "financial", "transaction"],
    "churn prediction":   ["churn", "customer retention", "binary classification", "customer"],
    "time series forecasting": ["forecasting", "temporal", "stock", "sequential", "prediction"],
    "anomaly detection":  ["outlier", "anomaly", "intrusion detection", "fault detection"],
    "electricity":        ["energy", "power", "time series", "forecasting", "demand"],
    "stock":              ["financial", "time series", "market", "trading", "forecasting"],

    # Audio
    "audio classification": ["sound event", "environmental sound", "esc", "audio"],
    "music":              ["audio", "song", "genre classification", "music information retrieval"],
    "keyword spotting":   ["wake word", "speech", "audio", "asr"],
}


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER ROUTING TABLE
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_WEIGHTS: dict[str, dict[str, float]] = {
    # Speech / Audio → HuggingFace dominant
    "speech":           {"huggingface": 1.5, "kaggle": 1.2, "openml": 0.5},
    "asr":              {"huggingface": 1.6, "kaggle": 1.1, "openml": 0.4},
    "audio":            {"huggingface": 1.5, "kaggle": 1.2, "openml": 0.4},
    "transcription":    {"huggingface": 1.6, "kaggle": 1.0, "openml": 0.3},
    "voice":            {"huggingface": 1.5, "kaggle": 1.1, "openml": 0.4},

    # NLP → HuggingFace dominates
    "text":             {"huggingface": 1.5, "kaggle": 1.1, "openml": 0.6},
    "nlp":              {"huggingface": 1.6, "kaggle": 1.1, "openml": 0.5},
    "language":         {"huggingface": 1.5, "kaggle": 1.0, "openml": 0.5},
    "translation":      {"huggingface": 1.6, "kaggle": 1.0, "openml": 0.3},
    "sentiment":        {"huggingface": 1.4, "kaggle": 1.3, "openml": 0.6},
    "llm":              {"huggingface": 1.8, "kaggle": 0.8, "openml": 0.3},
    "summarization":    {"huggingface": 1.6, "kaggle": 0.9, "openml": 0.3},

    # Indian / Indic NLP → HuggingFace + Kaggle
    "hindi":            {"huggingface": 1.6, "kaggle": 1.3, "openml": 0.4},
    "indic":            {"huggingface": 1.6, "kaggle": 1.2, "openml": 0.4},
    "gujarati":         {"huggingface": 1.5, "kaggle": 1.3, "openml": 0.3},
    "tamil":            {"huggingface": 1.5, "kaggle": 1.2, "openml": 0.3},
    "multilingual":     {"huggingface": 1.5, "kaggle": 1.1, "openml": 0.4},

    # Computer Vision → Kaggle + HuggingFace
    "image":            {"huggingface": 1.3, "kaggle": 1.5, "openml": 0.5},
    "vision":           {"huggingface": 1.3, "kaggle": 1.5, "openml": 0.4},
    "detection":        {"huggingface": 1.3, "kaggle": 1.5, "openml": 0.4},
    "segmentation":     {"huggingface": 1.4, "kaggle": 1.4, "openml": 0.3},
    "pothole":          {"huggingface": 1.1, "kaggle": 1.7, "openml": 0.3},
    "leaf":             {"huggingface": 1.2, "kaggle": 1.6, "openml": 0.4},

    # Tabular / ML Benchmarks → OpenML + Kaggle
    "tabular":          {"huggingface": 0.7, "kaggle": 1.4, "openml": 1.6},
    "regression":       {"huggingface": 0.6, "kaggle": 1.4, "openml": 1.6},
    "classification":   {"huggingface": 0.8, "kaggle": 1.3, "openml": 1.5},
    "benchmark":        {"huggingface": 1.0, "kaggle": 0.9, "openml": 1.8},
    "structured":       {"huggingface": 0.6, "kaggle": 1.3, "openml": 1.7},
    "forecasting":      {"huggingface": 0.8, "kaggle": 1.4, "openml": 1.4},

    # Agriculture → Kaggle strong
    "agriculture":      {"huggingface": 1.0, "kaggle": 1.6, "openml": 0.8},
    "plant":            {"huggingface": 1.0, "kaggle": 1.5, "openml": 0.7},
    "crop":             {"huggingface": 0.9, "kaggle": 1.6, "openml": 0.8},

    # Medical → HuggingFace + Kaggle
    "medical":          {"huggingface": 1.4, "kaggle": 1.4, "openml": 1.0},
    "clinical":         {"huggingface": 1.3, "kaggle": 1.2, "openml": 1.2},
    "radiology":        {"huggingface": 1.3, "kaggle": 1.5, "openml": 0.6},
    "diabetes":         {"huggingface": 1.0, "kaggle": 1.3, "openml": 1.5},
    "cancer":           {"huggingface": 1.3, "kaggle": 1.4, "openml": 0.9},
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnricherOutput:
    """
    Enriched query ready for multi-provider search.

    expanded_query:   Augmented text for BM25/keyword search.
    semantic_hints:   Additional context tokens for semantic retrieval.
    provider_weights: Per-provider score multipliers (default 1.0).
    domains_detected: All domains that triggered enrichment (multi-domain support).
    domain_detected:  Primary domain (compat shim — first of domains_detected).
    expansion_terms:  All added terms (for debugging/explainability).
    """
    original_query:   str
    expanded_query:   str
    semantic_hints:   list[str] = field(default_factory=list)
    provider_weights: dict[str, float] = field(default_factory=lambda: {
        "huggingface": 1.0, "kaggle": 1.0, "openml": 1.0
    })
    domains_detected: list[str] = field(default_factory=list)
    expansion_terms:  list[str] = field(default_factory=list)

    @property
    def domain_detected(self) -> Optional[str]:
        """Compat shim — callers expecting a single domain get the primary one."""
        return self.domains_detected[0] if self.domains_detected else None


# ─────────────────────────────────────────────────────────────────────────────
# QUERY ENRICHER
# ─────────────────────────────────────────────────────────────────────────────

class QueryEnricher:
    """
    Expands a cleaned query with domain synonyms and provider routing.

    Key behaviour change (v2):
      - ALL matching domains are processed (no early break).
      - Provider weights are merged across all matched domains.
      - Expansion terms are deduplicated while preserving order.
      - Multi-domain queries like "Hindi speech recognition" now correctly
        trigger BOTH the Indic NLP weights AND the ASR/audio weights.
    """

    def enrich(self, query: str) -> EnricherOutput:
        """Never raises — returns original query on any failure."""
        if not query or not query.strip():
            return EnricherOutput(original_query=query, expanded_query=query)
        try:
            return self._enrich_internal(query.strip())
        except Exception:
            return EnricherOutput(original_query=query, expanded_query=query)

    def _enrich_internal(self, query: str) -> EnricherOutput:
        q_lower = query.lower()
        expansion_terms: list[str] = []
        seen_terms: set[str] = set(q_lower.split())  # don't re-add what's already there
        provider_weights: dict[str, float] = {"huggingface": 1.0, "kaggle": 1.0, "openml": 1.0}
        domains_detected: list[str] = []

        # ── Phase 1: domain expansion — match ALL phrases (longest first) ─────
        sorted_phrases = sorted(_DOMAIN_EXPANSION.keys(), key=len, reverse=True)
        matched_phrases: set[str] = set()

        for phrase in sorted_phrases:
            if phrase in q_lower:
                # Skip if a longer phrase that contains this one already matched
                # (e.g. don't match "speech" if "speech recognition" matched)
                if any(phrase in mp and phrase != mp for mp in matched_phrases):
                    continue

                terms = _DOMAIN_EXPANSION[phrase]
                new_terms = [t for t in terms if t.lower() not in seen_terms][:4]
                for t in new_terms:
                    if t not in expansion_terms:
                        expansion_terms.append(t)
                        seen_terms.add(t.lower())

                domains_detected.append(phrase)
                matched_phrases.add(phrase)

        # ── Phase 2: provider routing from individual keywords ─────────────────
        for keyword, weights in _PROVIDER_WEIGHTS.items():
            if keyword in q_lower:
                for provider, mult in weights.items():
                    current = provider_weights.get(provider, 1.0)
                    # Accumulate: max boost, min penalty
                    if mult > 1.0:
                        provider_weights[provider] = max(current, mult)
                    else:
                        provider_weights[provider] = min(current, mult)

        # ── Phase 3: build expanded query ─────────────────────────────────────
        if expansion_terms:
            extra = " ".join(expansion_terms[:8])
            expanded_query = f"{query} {extra}"
        else:
            expanded_query = query

        # ── Phase 4: semantic hints ────────────────────────────────────────────
        semantic_hints = list(dict.fromkeys(expansion_terms[:10]))

        return EnricherOutput(
            original_query=query,
            expanded_query=expanded_query,
            semantic_hints=semantic_hints,
            provider_weights=provider_weights,
            domains_detected=domains_detected,
            expansion_terms=expansion_terms,
        )
