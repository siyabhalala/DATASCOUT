"""
datascout.evaluation.metrics.quality
────────────────────────────────────────
Dataset quality evaluator: metadata richness, documentation, and bias signals.

Evaluates:
  - Metadata completeness (structural field coverage)
  - Documentation quality (description richness)
  - Schema info (column names available)
  - License clarity
  - Annotation richness (tag count, task type coverage)
  - Bias and coverage risk signals (class imbalance, missing values)
  - Missing metadata penalties

Returns a QualityResult with score + bias signals + explainable breakdown.
Deterministic — no LLM involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from datascout.contracts import RawDataset

# Component weights (sum = 1.0)
_W_COMPLETENESS   = 0.35
_W_DESCRIPTION    = 0.20
_W_SCHEMA         = 0.15
_W_LICENSE        = 0.10
_W_ANNOTATION     = 0.10
_W_BIAS_PENALTY   = 0.10  # Reduces score when bias signals detected

# Description richness thresholds
_DESC_RICH_CHARS    = 500   # Well-documented
_DESC_MINIMAL_CHARS = 100   # Barely documented

# Class imbalance risk threshold (from OpenML extra data)
_IMBALANCE_RATIO_THRESHOLD = 5.0  # majority/minority > 5 → flagged


@dataclass
class BiasSignal:
    """A single detected bias or risk signal."""
    signal_type: str    # e.g. "class_imbalance", "missing_values", "no_license"
    severity: str       # "low", "medium", "high"
    description: str
    penalty: float      # Score deduction (0.0–0.2)

    def __str__(self) -> str:
        """Human-readable warning for UI display — never exposes internal repr."""
        icon = {"high": "⚠", "medium": "⚠", "low": "ℹ"}.get(self.severity, "⚠")
        return f"{icon} {self.description}"


@dataclass
class QualityResult:
    score: float                       # 0.0–1.0 composite
    completeness_score: float          # Structural metadata coverage
    description_score: float           # Documentation quality
    schema_score: float                # Column/schema info available
    license_score: float               # License clarity
    annotation_score: float            # Tags, task types, modality coverage
    bias_penalty: float                # Total penalty from bias signals
    bias_signals: list[BiasSignal]     # Detected risks
    missing_fields: list[str]          # Which fields are missing
    explanation: str
    has_schema_info: bool
    has_license_info: bool


def score_quality(dataset: RawDataset) -> QualityResult:
    """
    Score dataset quality and surface bias signals deterministically.

    Never raises.
    """
    # ── Completeness (structural field coverage) ──────────────────────
    completeness_score = round(dataset.metadata_completeness, 4)

    # ── Description quality ───────────────────────────────────────────
    desc_len = dataset.description_length or 0
    if desc_len >= _DESC_RICH_CHARS:
        description_score = 1.0
    elif desc_len >= _DESC_MINIMAL_CHARS:
        description_score = round(0.5 + 0.5 * (desc_len - _DESC_MINIMAL_CHARS) /
                                  (_DESC_RICH_CHARS - _DESC_MINIMAL_CHARS), 4)
    elif desc_len > 0:
        description_score = round(0.3 * desc_len / _DESC_MINIMAL_CHARS, 4)
    else:
        description_score = 0.0

    # ── Schema info ───────────────────────────────────────────────────
    schema_score = 1.0 if dataset.has_schema_info else 0.0

    # ── License clarity ───────────────────────────────────────────────
    license_score = 1.0 if dataset.has_license_info else 0.0

    # ── Annotation richness (tags + task types + modalities) ──────────
    tag_score = min(dataset.tag_count / 5.0, 1.0) if dataset.tag_count > 0 else 0.0
    task_score = 1.0 if dataset.task_types else 0.0
    mod_score  = 1.0 if dataset.modalities else 0.0
    annotation_score = round((tag_score * 0.4 + task_score * 0.4 + mod_score * 0.2), 4)

    # ── Bias & risk signal detection ──────────────────────────────────
    bias_signals: list[BiasSignal] = []
    missing_fields: list[str] = []

    # Missing license penalty
    if not dataset.has_license_info:
        bias_signals.append(BiasSignal(
            signal_type="no_license",
            severity="medium",
            description="License not specified — legal use unclear.",
            penalty=0.05,
        ))
        missing_fields.append("license")

    # Missing description penalty
    if not dataset.has_description:
        bias_signals.append(BiasSignal(
            signal_type="no_description",
            severity="high",
            description="No description — dataset purpose and content unknown.",
            penalty=0.10,
        ))
        missing_fields.append("description")

    # Missing size info
    if not dataset.has_size_info:
        bias_signals.append(BiasSignal(
            signal_type="no_size_info",
            severity="low",
            description="Dataset size (rows/file) unknown — can't assess ML suitability.",
            penalty=0.03,
        ))
        missing_fields.append("size")

    # Class imbalance (from OpenML extra data)
    if dataset.extra:
        majority = dataset.extra.get("majority_class_size")
        minority = dataset.extra.get("minority_class_size")
        if majority and minority and minority > 0:
            ratio = majority / minority
            if ratio > _IMBALANCE_RATIO_THRESHOLD:
                severity = "high" if ratio > 20 else "medium"
                bias_signals.append(BiasSignal(
                    signal_type="class_imbalance",
                    severity=severity,
                    description=(
                        f"Class imbalance detected: majority/minority ratio = {ratio:.1f}. "
                        "Consider resampling or weighted loss functions."
                    ),
                    penalty=0.05 if severity == "medium" else 0.08,
                ))

        # Missing values
        missing_vals = dataset.extra.get("missing_values")
        if missing_vals and isinstance(missing_vals, (int, float)) and missing_vals > 0:
            if dataset.row_count and dataset.row_count > 0:
                miss_pct = missing_vals / (dataset.row_count * max(dataset.column_count or 1, 1))
                if miss_pct > 0.20:
                    bias_signals.append(BiasSignal(
                        signal_type="high_missing_values",
                        severity="high",
                        description=f"High missing value rate ({miss_pct:.1%}) — imputation required.",
                        penalty=0.08,
                    ))
                elif miss_pct > 0.05:
                    bias_signals.append(BiasSignal(
                        signal_type="moderate_missing_values",
                        severity="low",
                        description=f"Moderate missing values ({miss_pct:.1%}) — review before training.",
                        penalty=0.02,
                    ))

    bias_penalty = round(min(sum(s.penalty for s in bias_signals), 0.3), 4)

    # ── Composite score ───────────────────────────────────────────────
    raw_score = (
        completeness_score   * _W_COMPLETENESS +
        description_score    * _W_DESCRIPTION  +
        schema_score         * _W_SCHEMA       +
        license_score        * _W_LICENSE      +
        annotation_score     * _W_ANNOTATION
    )
    score = round(max(0.0, min(raw_score - bias_penalty * _W_BIAS_PENALTY, 1.0)), 4)

    # ── Explanation ───────────────────────────────────────────────────
    if score >= 0.80:
        quality_label = "high quality"
    elif score >= 0.60:
        quality_label = "good quality"
    elif score >= 0.40:
        quality_label = "moderate quality"
    else:
        quality_label = "limited documentation"

    bias_note = ""
    if bias_signals:
        high_sev = [s for s in bias_signals if s.severity == "high"]
        if high_sev:
            bias_note = f" ⚠ {len(high_sev)} high-severity issue(s) detected."
        else:
            bias_note = f" {len(bias_signals)} minor issue(s) noted."

    explanation = (
        f"Dataset is {quality_label} (completeness={completeness_score:.2f}, "
        f"desc_len={desc_len} chars).{bias_note}"
    )

    return QualityResult(
        score=score,
        completeness_score=completeness_score,
        description_score=description_score,
        schema_score=schema_score,
        license_score=license_score,
        annotation_score=annotation_score,
        bias_penalty=bias_penalty,
        bias_signals=bias_signals,
        missing_fields=missing_fields,
        explanation=explanation,
        has_schema_info=dataset.has_schema_info,
        has_license_info=dataset.has_license_info,
    )