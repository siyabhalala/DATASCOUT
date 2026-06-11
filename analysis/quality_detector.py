"""
datascout.analysis.quality_detector
─────────────────────────────────────────────────────
Quality signal detectors — class balance, duplicates, blur, content consistency.
Phase 8: re-exports GeographicBiasDetector, ImageDimensionChecker, CorruptionDetector.

FIX (v3.2.0):
  1. ClassBalanceDetector.detect_from_class_counts() — added "too few classes"
     warning. A plant disease dataset with 2 folders is a red flag — PlantVillage
     has 38 diseases. The number of classes itself is a quality signal.

  2. DuplicateDetector.detect_from_image_hashes() — new method. Takes
     {class: [phash, ...]} of EVERY image (from ImageSampleResult.class_info)
     and finds exact duplicates using a hash set across ALL classes.
     Cross-class duplicates (same image under two labels) are the most dangerous
     kind — they directly inflate test accuracy metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("datascout.analysis.quality_detector")

IMBALANCE_WARNING_THRESHOLD = 30.0
BLUR_THRESHOLD              = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# CLASS BALANCE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassBalanceResult:
    score: Optional[float]
    class_distribution: dict
    dominant_class: Optional[str]
    dominant_percentage: float
    minority_percentage: float
    is_imbalanced: bool
    warning_message: Optional[str]
    num_classes: int

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2) if self.score is not None else None,
            "class_distribution": {str(k): v for k, v in self.class_distribution.items()},
            "dominant_class": str(self.dominant_class) if self.dominant_class else None,
            "dominant_percentage": round(self.dominant_percentage, 2),
            "minority_percentage": round(self.minority_percentage, 2),
            "is_imbalanced": self.is_imbalanced,
            "warning_message": self.warning_message,
            "num_classes": self.num_classes,
        }


class ClassBalanceDetector:

    def detect(self, df: Any, target_column: Optional[str] = None) -> ClassBalanceResult:
        try:
            return self._detect_internal(df, target_column)
        except Exception as e:
            logger.warning("class_balance_error", extra={"error": str(e)[:100]})
            return ClassBalanceResult(
                score=None, class_distribution={}, dominant_class=None,
                dominant_percentage=0.0, minority_percentage=0.0,
                is_imbalanced=False,
                warning_message=f"Analysis failed: {str(e)[:100]}",
                num_classes=0,
            )

    def detect_from_class_counts(
        self, images_per_class: dict[str, int]
    ) -> ClassBalanceResult:
        """
        Detect imbalance from real {class: total_image_count} dict.

        FIX: Added "too few classes" check. For a plant disease dataset,
        having only 2-3 folders means the dataset covers only 2-3 diseases
        out of potentially 38+. The class count itself is a quality signal.
        """
        try:
            if not images_per_class:
                return self._no_target_result()

            total          = sum(images_per_class.values())
            sorted_counts  = sorted(images_per_class.items(), key=lambda x: x[1], reverse=True)
            dominant_cls, majority_count = sorted_counts[0]
            minority_cls, minority_count = sorted_counts[-1]
            num_classes    = len(images_per_class)

            if num_classes == 1:
                return ClassBalanceResult(
                    score=0.0,
                    class_distribution=dict(images_per_class),
                    dominant_class=dominant_cls,
                    dominant_percentage=100.0,
                    minority_percentage=0.0,
                    is_imbalanced=True,
                    warning_message=f"Only one class found: '{dominant_cls}'. Cannot train a classifier.",
                    num_classes=1,
                )

            dominant_pct = majority_count / total * 100
            minority_pct = minority_count / total * 100
            score        = min((minority_count / majority_count) * 100, 100.0)
            is_imbalanced = score < IMBALANCE_WARNING_THRESHOLD

            warning = None
            # FIX: "too few classes" is a quality signal independent of balance
            if num_classes < 5:
                warning = (
                    f"Only {num_classes} class(es) detected. For a comprehensive dataset "
                    f"(e.g. plant disease covers 38+ diseases, animals covers 100+ species) "
                    f"this likely means limited coverage — check for a more complete version."
                )
            elif is_imbalanced:
                deficit = majority_count - minority_count
                warning = (
                    f"Class imbalance detected ({majority_count/minority_count:.1f}:1 ratio). "
                    f"Add ~{deficit} more '{minority_cls}' images to balance with '{dominant_cls}'. "
                    f"Imbalanced data causes biased models that perform well on majority class "
                    f"but poorly on minority — accuracy metrics will be misleading."
                )

            return ClassBalanceResult(
                score=round(score, 2),
                class_distribution=dict(images_per_class),
                dominant_class=dominant_cls,
                dominant_percentage=round(dominant_pct, 2),
                minority_percentage=round(minority_pct, 2),
                is_imbalanced=is_imbalanced,
                warning_message=warning,
                num_classes=num_classes,
            )
        except Exception as e:
            return ClassBalanceResult(
                score=None, class_distribution={}, dominant_class=None,
                dominant_percentage=0.0, minority_percentage=0.0,
                is_imbalanced=False,
                warning_message=f"Balance check failed: {e}",
                num_classes=0,
            )

    def _detect_internal(self, df: Any, target_column: Optional[str]) -> ClassBalanceResult:
        import pandas as pd

        if df is None or len(df) == 0:
            return self._no_data_result()

        col = target_column or self._find_target_column(df)
        if col is None or col not in df.columns:
            return self._no_target_result()

        series      = df[col].dropna()
        if len(series) == 0:
            return self._no_target_result()

        value_counts = series.value_counts()
        total        = len(series)
        dist         = {str(k): int(v) for k, v in value_counts.items()}
        num_classes  = len(dist)

        if num_classes == 0:
            return self._no_target_result()
        if num_classes == 1:
            dominant = str(value_counts.index[0])
            return ClassBalanceResult(
                score=0.0, class_distribution=dist, dominant_class=dominant,
                dominant_percentage=100.0, minority_percentage=0.0,
                is_imbalanced=True,
                warning_message=f"Only one class found: '{dominant}'. Cannot train classifier.",
                num_classes=1,
            )

        majority_count = int(value_counts.iloc[0])
        minority_count = int(value_counts.iloc[-1])
        dominant_class = str(value_counts.index[0])
        dominant_pct   = majority_count / total * 100
        minority_pct   = minority_count / total * 100
        score          = min((minority_count / majority_count) * 100, 100.0)
        is_imbalanced  = score < IMBALANCE_WARNING_THRESHOLD
        warning        = None
        if is_imbalanced:
            needed  = majority_count - minority_count
            warning = (
                f"Class imbalance: '{dominant_class}' dominates at {dominant_pct:.1f}%. "
                f"Consider adding ~{needed} more samples for minority classes."
            )
        return ClassBalanceResult(
            score=round(score, 2), class_distribution=dist,
            dominant_class=dominant_class,
            dominant_percentage=round(dominant_pct, 2),
            minority_percentage=round(minority_pct, 2),
            is_imbalanced=is_imbalanced, warning_message=warning,
            num_classes=num_classes,
        )

    @staticmethod
    def _find_target_column(df: Any) -> Optional[str]:
        target_hints = {"target","label","labels","class","y","output",
                        "survived","churn","fraud","diagnosis"}
        for col in df.columns:
            if col.lower().strip() in target_hints:
                return col
        return df.columns[-1] if len(df.columns) > 0 else None

    @staticmethod
    def _no_data_result() -> ClassBalanceResult:
        return ClassBalanceResult(score=None, class_distribution={}, dominant_class=None,
                                  dominant_percentage=0.0, minority_percentage=0.0,
                                  is_imbalanced=False, warning_message="No data available.",
                                  num_classes=0)

    @staticmethod
    def _no_target_result() -> ClassBalanceResult:
        return ClassBalanceResult(score=None, class_distribution={}, dominant_class=None,
                                  dominant_percentage=0.0, minority_percentage=0.0,
                                  is_imbalanced=False, warning_message="No target column identified.",
                                  num_classes=0)


# ══════════════════════════════════════════════════════════════════════════════
# DUPLICATE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DuplicateResult:
    duplicate_percentage: float
    duplicate_count: int
    total_rows: int
    has_duplicates: bool
    warning_message: Optional[str]

    def to_dict(self) -> dict:
        return {
            "duplicate_percentage": round(self.duplicate_percentage, 2),
            "duplicate_count": self.duplicate_count,
            "total_rows": self.total_rows,
            "has_duplicates": self.has_duplicates,
            "warning_message": self.warning_message,
        }


class DuplicateDetector:

    def detect(self, df: Any) -> DuplicateResult:
        try:
            return self._detect_internal(df)
        except Exception as e:
            logger.warning("duplicate_detection_error", extra={"error": str(e)[:100]})
            return DuplicateResult(duplicate_percentage=0.0, duplicate_count=0,
                                   total_rows=0, has_duplicates=False,
                                   warning_message=f"Detection failed: {str(e)[:100]}")

    def detect_from_image_hashes(
        self, class_hashes: dict[str, list[str]]
    ) -> DuplicateResult:
        """
        Detect duplicate images using perceptual hashes across ALL classes.

        WHY across all classes (not per-class):
        - A duplicate between class A and class B is MORE dangerous than
          within-class duplicates. The same image under two labels means
          it appears in both train and test — directly inflating accuracy.
        - Within-class duplicates cause overfitting (model memorises image).

        Args:
            class_hashes: {class_name: [phash_hex, ...]}
                          from ImageSampleResult.class_info[cls].all_hashes
                          Must be hashes of EVERY image, not a sample.
        """
        try:
            all_hashes: list[str] = []
            for hashes in class_hashes.values():
                all_hashes.extend(hashes)

            if not all_hashes:
                return DuplicateResult(duplicate_percentage=0.0, duplicate_count=0,
                                       total_rows=0, has_duplicates=False,
                                       warning_message=None)

            total     = len(all_hashes)
            seen:  set[str] = set()
            dup_count = 0
            for h in all_hashes:
                if h in seen:
                    dup_count += 1
                else:
                    seen.add(h)

            dup_pct = (dup_count / total * 100) if total > 0 else 0.0
            warning = None
            if dup_pct >= 10.0:
                warning = (
                    f"{dup_count}/{total} images ({dup_pct:.1f}%) are exact duplicates. "
                    f"This causes overfitting — the model memorises repeated images "
                    f"instead of learning generalised features. "
                    f"Remove duplicates before training."
                )
            elif dup_pct >= 3.0:
                warning = (
                    f"{dup_count}/{total} images ({dup_pct:.1f}%) are duplicates. "
                    f"Consider deduplicating to improve model generalisation."
                )

            return DuplicateResult(
                duplicate_percentage=round(dup_pct, 2),
                duplicate_count=dup_count,
                total_rows=total,
                has_duplicates=dup_count > 0,
                warning_message=warning,
            )
        except Exception as e:
            return DuplicateResult(duplicate_percentage=0.0, duplicate_count=0,
                                   total_rows=0, has_duplicates=False,
                                   warning_message=f"Image dup detection failed: {str(e)[:100]}")

    @staticmethod
    def _detect_internal(df: Any) -> DuplicateResult:
        if df is None or len(df) == 0:
            return DuplicateResult(duplicate_percentage=0.0, duplicate_count=0,
                                   total_rows=0, has_duplicates=False, warning_message=None)
        total     = len(df)
        dup_count = int(df.duplicated().sum())
        dup_pct   = (dup_count / total * 100) if total > 0 else 0.0
        warning   = None
        if dup_pct >= 5.0:
            warning = (f"{dup_count} duplicate rows ({dup_pct:.1f}%). "
                       f"Remove duplicates before training to prevent data leakage.")
        elif dup_pct > 0:
            warning = f"{dup_count} duplicate rows ({dup_pct:.1f}%)."
        return DuplicateResult(
            duplicate_percentage=round(dup_pct, 2), duplicate_count=dup_count,
            total_rows=total, has_duplicates=dup_count > 0, warning_message=warning,
        )


# ══════════════════════════════════════════════════════════════════════════════
# BLUR DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlurResult:
    score: Optional[float]
    blurry_percentage: float
    images_sampled: int
    images_failed: int
    warning_message: Optional[str]
    opencv_available: bool

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2) if self.score is not None else None,
            "blurry_percentage": round(self.blurry_percentage, 2),
            "images_sampled": self.images_sampled,
            "images_failed": self.images_failed,
            "warning_message": self.warning_message,
            "opencv_available": self.opencv_available,
        }


class BlurDetector:

    def __init__(self, blur_threshold: float = BLUR_THRESHOLD) -> None:
        self.blur_threshold = blur_threshold
        self._opencv_available: Optional[bool] = None

    def _check_opencv(self) -> bool:
        if self._opencv_available is None:
            try:
                import cv2
                self._opencv_available = True
            except ImportError:
                self._opencv_available = False
                logger.warning("opencv_not_available",
                               extra={"install": "pip install opencv-python"})
        return self._opencv_available

    def detect_from_paths(self, image_paths: list[str]) -> BlurResult:
        if not self._check_opencv():
            return BlurResult(score=None, blurry_percentage=0.0, images_sampled=0,
                              images_failed=0,
                              warning_message="OpenCV not installed — pip install opencv-python",
                              opencv_available=False)
        import cv2
        blurry = sampled = failed = 0
        for path in image_paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    failed += 1
                    continue
                if cv2.Laplacian(img, cv2.CV_64F).var() < self.blur_threshold:
                    blurry += 1
                sampled += 1
            except Exception:
                failed += 1

        if sampled == 0:
            return BlurResult(score=None, blurry_percentage=0.0, images_sampled=0,
                              images_failed=failed,
                              warning_message="No images could be loaded.",
                              opencv_available=True)
        blurry_pct = (blurry / sampled) * 100
        warning    = None
        if blurry_pct >= 20.0:
            warning = (f"{blurry_pct:.1f}% of sampled images are blurry. "
                       f"Blurry training data reduces model accuracy on real-world sharp images. "
                       f"Filter or replace low-quality images.")
        return BlurResult(score=round(100.0 - blurry_pct, 2),
                          blurry_percentage=round(blurry_pct, 2),
                          images_sampled=sampled, images_failed=failed,
                          warning_message=warning, opencv_available=True)

    def detect_from_arrays(self, images: list[Any]) -> BlurResult:
        if not self._check_opencv():
            return BlurResult(score=None, blurry_percentage=0.0, images_sampled=0,
                              images_failed=0, warning_message="OpenCV not available",
                              opencv_available=False)
        import cv2, numpy as np
        blurry = sampled = failed = 0
        for img in images:
            try:
                arr  = np.array(img)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
                if cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var() < self.blur_threshold:
                    blurry += 1
                sampled += 1
            except Exception:
                failed += 1
        if sampled == 0:
            return BlurResult(score=None, blurry_percentage=0.0, images_sampled=0,
                              images_failed=failed, warning_message="No images processed.",
                              opencv_available=True)
        blurry_pct = (blurry / sampled) * 100
        return BlurResult(score=round(100.0 - blurry_pct, 2),
                          blurry_percentage=round(blurry_pct, 2),
                          images_sampled=sampled, images_failed=failed,
                          warning_message=(f"{blurry_pct:.1f}% blurry." if blurry_pct > 0 else None),
                          opencv_available=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT CONSISTENCY DETECTOR (CLIP)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContentConsistencyResult:
    score: Optional[float]
    mismatch_percentage: float
    classes_checked: int
    images_sampled: int
    images_failed: int
    flagged_classes: list[str]
    warning_message: Optional[str]
    clip_available: bool
    details: dict

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2) if self.score is not None else None,
            "mismatch_percentage": round(self.mismatch_percentage, 2),
            "classes_checked": self.classes_checked,
            "images_sampled": self.images_sampled,
            "images_failed": self.images_failed,
            "flagged_classes": self.flagged_classes,
            "warning_message": self.warning_message,
            "clip_available": self.clip_available,
            "details": {k: round(v, 2) for k, v in self.details.items()},
        }


class ContentConsistencyDetector:
    """
    Uses CLIP to verify images contain what their labels claim.
    This catches the "I asked for tomato dataset but it has other vegetables" problem.
    Graceful fallback if CLIP/torch not installed.
    """
    IMAGES_PER_CLASS   = 10
    MISMATCH_THRESHOLD = 0.20
    FLAG_THRESHOLD     = 0.25

    def __init__(self) -> None:
        self._clip_available: Optional[bool] = None
        self._model = self._preprocess = None
        self._device = "cpu"

    def _check_clip(self) -> bool:
        if self._clip_available is None:
            try:
                import torch, clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model, self._preprocess = clip.load("ViT-B/32", device=device)
                self._device = device
                self._clip_available = True
            except ImportError:
                self._clip_available = False
                logger.warning("clip_not_available",
                               extra={"install": "pip install clip-by-openai torch"})
        return self._clip_available

    def detect_from_labeled_paths(self, class_image_paths: dict[str, list[str]]) -> ContentConsistencyResult:
        if not self._check_clip():
            return ContentConsistencyResult(
                score=None, mismatch_percentage=0.0, classes_checked=0,
                images_sampled=0, images_failed=0, flagged_classes=[],
                warning_message="CLIP not installed — pip install clip-by-openai torch",
                clip_available=False, details={})
        if not class_image_paths:
            return ContentConsistencyResult(
                score=None, mismatch_percentage=0.0, classes_checked=0,
                images_sampled=0, images_failed=0, flagged_classes=[],
                warning_message="No labeled images provided.",
                clip_available=True, details={})
        try:
            return self._run_clip_check(class_image_paths)
        except Exception as e:
            return ContentConsistencyResult(
                score=None, mismatch_percentage=0.0, classes_checked=0,
                images_sampled=0, images_failed=0, flagged_classes=[],
                warning_message=f"Check failed: {str(e)[:100]}",
                clip_available=True, details={})

    def detect_from_description(self, description: str, image_paths: list[str]) -> ContentConsistencyResult:
        if not self._check_clip():
            return ContentConsistencyResult(
                score=None, mismatch_percentage=0.0, classes_checked=0,
                images_sampled=0, images_failed=0, flagged_classes=[],
                warning_message="CLIP not installed.", clip_available=False, details={})
        subject = self._extract_subject(description)
        return self.detect_from_labeled_paths({subject: image_paths[:self.IMAGES_PER_CLASS]})

    def _run_clip_check(self, class_image_paths: dict[str, list[str]]) -> ContentConsistencyResult:
        import torch, clip, random
        from PIL import Image
        total_images = total_mismatch = total_failed = 0
        flagged_classes: list[str] = []
        details: dict[str, float] = {}
        for label, paths in class_image_paths.items():
            sample      = random.sample(paths, min(self.IMAGES_PER_CLASS, len(paths)))
            text_tokens = clip.tokenize([f"a photo of {label}"]).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            cls_mismatch = cls_sampled = cls_failed = 0
            for path in sample:
                try:
                    image = Image.open(path).convert("RGB")
                    inp   = self._preprocess(image).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        img_features = self._model.encode_image(inp)
                        img_features /= img_features.norm(dim=-1, keepdim=True)
                    if (img_features @ text_features.T).item() < self.MISMATCH_THRESHOLD:
                        cls_mismatch += 1
                    cls_sampled += 1
                except Exception:
                    cls_failed += 1
            total_images   += cls_sampled
            total_mismatch += cls_mismatch
            total_failed   += cls_failed
            if cls_sampled > 0:
                rate = cls_mismatch / cls_sampled
                details[label] = round(rate * 100, 2)
                if rate > self.FLAG_THRESHOLD:
                    flagged_classes.append(label)
        if total_images == 0:
            return ContentConsistencyResult(
                score=None, mismatch_percentage=0.0,
                classes_checked=len(class_image_paths), images_sampled=0,
                images_failed=total_failed, flagged_classes=[],
                warning_message="No images could be processed.", clip_available=True, details={})
        mismatch_pct = (total_mismatch / total_images) * 100
        warning = None
        if flagged_classes:
            warning = (f"Content mismatch in {len(flagged_classes)} class(es): "
                       f"{', '.join(flagged_classes)}. "
                       f"Images may not match their labels — verify the dataset.")
        elif mismatch_pct > 10.0:
            warning = f"{mismatch_pct:.1f}% of sampled images may not match their labels."
        return ContentConsistencyResult(
            score=round(100.0 - mismatch_pct, 2),
            mismatch_percentage=round(mismatch_pct, 2),
            classes_checked=len(class_image_paths),
            images_sampled=total_images, images_failed=total_failed,
            flagged_classes=flagged_classes, warning_message=warning,
            clip_available=True, details=details)

    @staticmethod
    def _extract_subject(description: str) -> str:
        strip_words = {"dataset","data","classification","detection","segmentation",
                       "recognition","images","image","photos","photo","collection",
                       "training","test","validation","labeled","annotated"}
        tokens = [w for w in description.lower().split()
                  if w not in strip_words and len(w) > 2]
        return " ".join(tokens[:3]) if tokens else description[:30]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 RE-EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

from .image_quality_checks import (      # noqa: E402
    GeographicBiasDetector, GeographicBiasResult,
    ImageDimensionChecker,  DimensionCheckResult,
    CorruptionDetector,     CorruptionResult,
)

__all__ = [
    "ClassBalanceDetector", "ClassBalanceResult",
    "DuplicateDetector",    "DuplicateResult",
    "BlurDetector",         "BlurResult",
    "ContentConsistencyDetector", "ContentConsistencyResult",
    "GeographicBiasDetector", "GeographicBiasResult",
    "ImageDimensionChecker",  "DimensionCheckResult",
    "CorruptionDetector",     "CorruptionResult",
]