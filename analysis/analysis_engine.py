"""
datascout.analysis.analysis_engine
─────────────────────────────────────────────────────
Top-level analysis orchestrator.

FIX (v3.3.0): Four fixes applied on top of v3.2.0:

  4. PerClassImageStats — new dataclass. Carries per-class image counts
     (total, blurry, duplicate, corrupted) for the frontend's
     "Class Distribution" deep quality panel.

     Why this matters: the existing pipeline ran BlurDetector and
     CorruptionDetector on aggregate paths and threw away the per-class
     breakdown. ClassInfo in ImageSampleResult already holds sampled_paths
     (for blur/corruption) and all_hashes (for dup detection) per class.
     This fix computes per-class stats WITHOUT any extra downloads.

  5. _compute_per_class_image_stats() — iterates ClassInfo, computes
     blur count per class (OpenCV Laplacian), dup count per class (global
     hash frequency map — cross-class duplicates correctly attributed to
     the class that has the duplicate copy), corruption count per class
     (PIL open attempt). Gracefully skips if OpenCV not installed.

  6. AnalysisReport.per_class_stats exposed in to_dict() so search_v2.py
     can pass it to the frontend without further processing.

  7. columns dict added to to_dict() so column type counts
     (total/numeric/categorical) reach the API response.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from datascout.contracts import RawDataset
from datascout.contracts.task_types import Modality, TaskType

from .column_analyzer import ColumnAnalysisResult, ColumnAnalyzer
from .image_sample_loader import ImageSampleLoader, ImageSampleResult
from .quality_detector import (
    BlurResult,
    ClassBalanceDetector, ClassBalanceResult,
    ContentConsistencyDetector, ContentConsistencyResult,
    DuplicateDetector, DuplicateResult,
    GeographicBiasDetector, GeographicBiasResult,
    ImageDimensionChecker, DimensionCheckResult,
    CorruptionDetector, CorruptionResult,
    BLUR_THRESHOLD,
)
from .sample_loader import SampleLoadResult, SampleLoader

logger = logging.getLogger("datascout.analysis.analysis_engine")

WEIGHT_COMPLETENESS = 0.30
WEIGHT_BALANCE      = 0.25
WEIGHT_UNIQUENESS   = 0.25
WEIGHT_METADATA     = 0.20

# Webpage URL prefixes — these are dataset landing pages, not downloadable files.
_WEBPAGE_URL_PREFIXES: tuple[str, ...] = (
    "https://www.kaggle.com/datasets/",
    "https://kaggle.com/datasets/",
    "https://huggingface.co/datasets/",
    "https://www.huggingface.co/datasets/",
    # FIX: Only block OpenML dataset LANDING pages (/d/<id>), not the CSV
    # download endpoint (/data/get_csv/<id>) which serves real tabular data.
    "https://openml.org/d/",
    "https://www.openml.org/d/",
    "https://openml.org/t/",
    "https://www.openml.org/t/",
)

_DATA_FILE_EXTENSIONS: tuple[str, ...] = (
    ".csv", ".tsv", ".parquet", ".pq", ".arrow",
    ".json", ".jsonl", ".ndjson",
    ".xlsx", ".xls", ".zip",
)


# ── NEW: Per-class image quality stats ────────────────────────────────────────

@dataclass
class PerClassImageStats:
    """
    Per-class image quality breakdown.

    All counts are RAW INTEGERS — not percentages.
    Raw counts are the primary signal; callers can compute percentages.

    total_images  — real count from folder/API (not sampled)
    blurry_count  — from sampled paths (OpenCV Laplacian < BLUR_THRESHOLD)
    duplicate_count — images whose perceptual hash appears elsewhere in
                      the dataset (within-class OR cross-class)
    corrupted_count — images PIL could not open (sampled paths only)
    """
    name: str
    total_images: int
    blurry_count: int
    duplicate_count: int
    corrupted_count: int

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "images":         self.total_images,    # raw count — primary
            "duplicates":     self.duplicate_count, # raw count
            "blurry":         self.blurry_count,    # raw count
            "corrupted":      self.corrupted_count, # raw count
        }


@dataclass
class AnalysisReport:
    dataset_canonical_id: str
    analyzed_at: datetime

    # Tabular pipeline
    sample_result:   Optional[SampleLoadResult]          = None
    column_result:   Optional[ColumnAnalysisResult]      = None
    balance_result:  Optional[ClassBalanceResult]        = None
    duplicate_result: Optional[DuplicateResult]          = None

    # Image pipeline
    blur_result:     Optional[BlurResult]                = None
    content_result:  Optional[ContentConsistencyResult]  = None

    # Phase 8
    geographic_result: Optional[GeographicBiasResult]   = None
    dimension_result:  Optional[DimensionCheckResult]   = None
    corruption_result: Optional[CorruptionResult]       = None

    # NEW v3.3.0: per-class image stats (image datasets only)
    per_class_stats: list[PerClassImageStats]           = field(default_factory=list)

    # NEW v3.5.1: image info extracted from description text (no download needed)
    # Available even when Kaggle credentials are not configured.
    total_images_est: Optional[int]  = None   # from "41,000 images" in description
    num_classes_est:  Optional[int]  = None   # from "5 classes" in description

    # Scores (0-100)
    completeness_score: Optional[float] = None
    balance_score:      Optional[float] = None
    uniqueness_score:   Optional[float] = None
    metadata_score:     Optional[float] = None
    quality_score:      Optional[float] = None

    suggested_fixes: list[str] = field(default_factory=list)
    analysis_version: str = "3.3.0"
    is_partial: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        col_r = self.column_result
        return {
            "dataset_canonical_id": self.dataset_canonical_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "quality_score":       round(self.quality_score, 2)       if self.quality_score       is not None else None,
            "completeness_score":  round(self.completeness_score, 2)  if self.completeness_score  is not None else None,
            "balance_score":       round(self.balance_score, 2)       if self.balance_score       is not None else None,
            "uniqueness_score":    round(self.uniqueness_score, 2)    if self.uniqueness_score    is not None else None,
            "metadata_score":      round(self.metadata_score, 2)      if self.metadata_score      is not None else None,
            "suggested_fixes": self.suggested_fixes,
            "is_partial": self.is_partial,
            "error": self.error,
            # Per-class image quality breakdown (image datasets)
            "per_class_stats": [s.to_dict() for s in self.per_class_stats] if self.per_class_stats else None,
            # Column type summary (tabular datasets)
            "columns": {
                "total":       col_r.total_columns,
                "numeric":     col_r.numeric_columns,
                "categorical": col_r.categorical_columns,
                "text":        col_r.text_columns,
                "missing_pct": round(col_r.overall_missing_percentage, 1),
            } if col_r else None,
            "sample":      self.sample_result.to_dict()   if self.sample_result   else None,
            "columns_detail": self.column_result.to_dict() if self.column_result  else None,
            "class_balance": self.balance_result.to_dict() if self.balance_result else None,
            "duplicates":  self.duplicate_result.to_dict() if self.duplicate_result else None,
            "blur":        self.blur_result.to_dict()     if self.blur_result     else None,
            "content":     self.content_result.to_dict()  if self.content_result  else None,
            "geographic":  self.geographic_result.to_dict() if self.geographic_result else None,
            "dimensions":  self.dimension_result.to_dict() if self.dimension_result else None,
            "corruption":  self.corruption_result.to_dict() if self.corruption_result else None,
            "analysis_version": self.analysis_version,
        }

    def to_storage_dict(self) -> dict:
        return {
            "class_imbalance_score":    self.balance_score,
            "duplicate_percentage":     self.duplicate_result.duplicate_percentage if self.duplicate_result else None,
            "blur_quality_score":       self.blur_result.score if self.blur_result else None,
            "geographic_bias_detected": self.geographic_result.is_single_region if self.geographic_result else None,
            "geographic_regions":       self.geographic_result.detected_regions  if self.geographic_result else None,
            "dimension_consistent":     self.dimension_result.is_consistent      if self.dimension_result  else None,
            "corruption_percentage":    self.corruption_result.corruption_percentage if self.corruption_result else None,
            "suggested_fixes":          self.suggested_fixes,
        }



# Paths that always serve downloadable data even without a file extension
_KNOWN_DOWNLOAD_PATH_PATTERNS: tuple[str, ...] = (
    "/data/get_csv/",          # OpenML CSV download
    "/data/v1/download/",      # OpenML API download
    "api.openml.org/data/",    # OpenML API base
)

class AnalysisEngine:

    def __init__(self, sample_rows: int = 500, timeout_seconds: float = 600.0,
                 target_column: Optional[str] = None,
                 download_images: bool = True,
                 deep_mode: bool = False) -> None:
        """
        Args:
            sample_rows:      Max rows to load for tabular datasets.
            timeout_seconds:  Per-dataset timeout for HTTP + image downloads.
            target_column:    Hint for target column detection (tabular).
            download_images:  If False, use Kaggle file-listing API to infer
                              class counts without downloading the dataset.
                              Set True only for the deep-analysis pass where
                              full image quality checks (blur/dup/corrupt) are
                              needed.  False is safe for all fast/meta passes.
            deep_mode:        FIX v2.1.0 (Issue 6): When True, ImageSampleLoader
                              uses MAX_IMAGES_PER_CLASS_DEEP (2000) instead of 50.
                              Must be True for top-3 deep analysis. Previously the
                              cap was always 50 regardless of fast vs deep path,
                              meaning a 5676-image dataset was analysed with ~1900
                              images instead of all 5676.
        """
        self.sample_rows      = sample_rows
        self.timeout_seconds  = timeout_seconds
        self.target_column    = target_column
        self.download_images  = download_images
        self.deep_mode        = deep_mode
        self._loader          = SampleLoader(max_rows=sample_rows, timeout_seconds=timeout_seconds)
        self._col_analyzer    = ColumnAnalyzer()
        self._balance_det     = ClassBalanceDetector()
        self._dup_det         = DuplicateDetector()
        self._image_loader    = ImageSampleLoader(deep_mode=deep_mode)
        self._geo_det         = GeographicBiasDetector()
        self._dim_checker     = ImageDimensionChecker()
        self._corruption_det  = CorruptionDetector()

    # ── URL validation ─────────────────────────────────────────────────────────

    @staticmethod
# Paths that always serve downloadable data even without a file extension

    def _is_downloadable_url(url: Optional[str]) -> bool:
        if not url:
            return False
        url_path = url.lower().split("?")[0].split("#")[0]
        if any(url.startswith(p) for p in _WEBPAGE_URL_PREFIXES):
            return False
        # Known download patterns (no extension required)
        if any(pat in url_path for pat in _KNOWN_DOWNLOAD_PATH_PATTERNS):
            return True
        return any(url_path.endswith(ext) for ext in _DATA_FILE_EXTENSIONS)

    # ── Per-class image stats ──────────────────────────────────────────────────

    @staticmethod
    def _compute_per_class_image_stats(
        image_sample: ImageSampleResult,
    ) -> list[PerClassImageStats]:
        """
        Compute per-class blur, duplicate, and corruption counts from
        ImageSampleResult.class_info — NO additional downloads needed.

        Uses:
          class_info[cls].total_images  — real count (from folder)
          class_info[cls].sampled_paths — for blur + corruption checks
          class_info[cls].all_hashes    — for duplicate detection

        Duplicate attribution: a hash is "duplicate" if it appears more than
        once anywhere in the dataset (within-class OR cross-class).
        Each class gets credit for its own duplicate images.

        Cross-class duplicates (same image under two different labels) are
        the MOST dangerous kind — they directly inflate test accuracy.
        """
        if not image_sample.class_info:
            return []

        # Build global hash frequency map across ALL classes
        # A hash with frequency > 1 anywhere = duplicate
        all_hashes_flat = [
            h
            for cls_info in image_sample.class_info.values()
            for h in cls_info.all_hashes
        ]
        hash_freq = Counter(all_hashes_flat)

        # Check if OpenCV is available
        _cv2_available = False
        try:
            import cv2 as _cv2
            _cv2_available = True
        except ImportError:
            logger.debug(
                "per_class_blur_skipped",
                extra={"reason": "OpenCV not installed", "fix": "pip install opencv-python"},
            )

        # Check if PIL is available
        _pil_available = False
        try:
            from PIL import Image as _PILImg
            _pil_available = True
        except ImportError:
            pass

        per_class: list[PerClassImageStats] = []

        for cls_name, cls_info in image_sample.class_info.items():
            blurry_count  = 0
            corrupt_count = 0

            if cls_info.sampled_paths:
                if _cv2_available:
                    import cv2
                    for path in cls_info.sampled_paths:
                        try:
                            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                corrupt_count += 1
                                continue
                            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                            if laplacian_var < BLUR_THRESHOLD:
                                blurry_count += 1
                        except Exception:
                            corrupt_count += 1
                elif _pil_available:
                    # Fallback: use PIL for corruption detection only (no blur)
                    from PIL import Image as _PILImg
                    for path in cls_info.sampled_paths:
                        try:
                            with _PILImg.open(path) as im:
                                im.verify()
                        except Exception:
                            corrupt_count += 1

            # Duplicate count: images whose hash appears >1 time globally
            dup_count = sum(1 for h in cls_info.all_hashes if hash_freq[h] > 1)

            per_class.append(PerClassImageStats(
                name=cls_name,
                total_images=cls_info.total_images,
                blurry_count=blurry_count,
                duplicate_count=dup_count,
                corrupted_count=corrupt_count,
            ))

        # Sort by total_images descending for consistent display
        per_class.sort(key=lambda s: s.total_images, reverse=True)
        return per_class

    # ── Description-based image info extraction ───────────────────────────────

    @staticmethod
    def _extract_from_description(description: str, title: str = "") -> dict:
        """
        Extract total image count and class count from description/title text.

        Works WITHOUT any API credentials or downloads.
        Used as fallback when Kaggle auth is not configured or file listing
        returns only a ZIP.

        Patterns handled:
          "over 41,000 images"         → total_images: 41000
          "Curated 19K rice-leaf"      → total_images: 19000
          "38.4k files"                → total_images: 38400
          "5 classes"                  → num_classes: 5
          "5 different class"          → num_classes: 5
          "categorized into 5 types"   → num_classes: 5
        """
        import re
        text = f"{title} {description}"
        result: dict = {}

        # ── Total image / file count ──────────────────────────────────────────
        count_patterns = [
            # "41,000 images" / "41000 photos" / "41K samples"
            r'(\d[\d,]+)\s*(?:images?|photos?|samples?|pictures?|files?)',
            # "19K images" / "38.4k files"
            r'(\d+\.?\d*)\s*[kK]\s*(?:images?|photos?|samples?|pictures?|files?)',
            # "over 41,000" — generic large number in context
            r'(?:over|around|about|~)\s*(\d[\d,]+)',
        ]
        for pat in count_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(',', '')
                try:
                    val = float(raw)
                    # Check if 'k' / 'K' appears right after the number in original match
                    if re.search(r'\d+\.?\d*\s*[kK]', m.group(0)):
                        val *= 1000
                    n = int(val)
                    if 50 <= n <= 10_000_000:     # sanity range
                        result['total_images'] = n
                        break
                except ValueError:
                    continue

        # ── Class / category count ────────────────────────────────────────────
        class_patterns = [
            r'(\d+)\s*(?:different\s+)?(?:classes?|categories?|labels?|types?|folders?|species)',
            r'(?:categorized|classified|grouped)\s+into\s+(\d+)',
        ]
        for pat in class_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if 2 <= n <= 1000:               # sanity range
                    result['num_classes'] = n
                    break

        return result

    # ── Main entry point ───────────────────────────────────────────────────────

    async def analyze(self, dataset: RawDataset, df: Optional[Any] = None) -> AnalysisReport:
        report = AnalysisReport(
            dataset_canonical_id=dataset.canonical_id,
            analyzed_at=datetime.now(timezone.utc),
        )
        try:
            is_image = Modality.IMAGE in dataset.modalities

            # ── 1. Image pipeline ─────────────────────────────────────────────
            if is_image or Modality.MULTIMODAL in dataset.modalities:
                source = getattr(dataset, "source", "") or ""

                image_sample: Optional[ImageSampleResult] = None

                _kaggle_ok = False
                _hf_ok     = False
                try:
                    import kaggle as _  # noqa: F401
                    _kaggle_ok = True
                except ImportError:
                    pass
                try:
                    import datasets as _  # noqa: F401
                    _hf_ok = True
                except ImportError:
                    pass

                if source == "kaggle" and dataset.source_url:
                    if not _kaggle_ok:
                        logger.debug(
                            "image_deep_analysis_skipped",
                            extra={
                                "reason": "kaggle SDK not installed",
                                "fix": "pip install kaggle",
                                "canonical_id": dataset.canonical_id,
                            },
                        )
                    else:
                        ref = (dataset.source_url
                               .replace("https://www.kaggle.com/datasets/", "")
                               .replace("https://kaggle.com/datasets/", "")
                               .rstrip("/"))
                        ref_parts = ref.split("/")
                        owner = ref_parts[0] if len(ref_parts) >= 1 else ""
                        name  = ref_parts[1] if len(ref_parts) >= 2 else ""

                        if not self.download_images:
                            # ── Fast path: metadata only, no download ─────────
                            # Use dataset_list_files() to parse class counts from
                            # file path structure (train/ClassName/img.jpg).
                            # < 2 seconds, no storage, works for any dataset size.
                            if owner and name:
                                meta_result = await asyncio.to_thread(
                                    self._image_loader.load_class_counts_from_kaggle_meta,
                                    owner, name
                                )
                                if meta_result.success:
                                    report.balance_result = self._balance_det.detect_from_class_counts(
                                        meta_result.images_per_class
                                    )
                                    report.balance_score = (
                                        report.balance_result.score
                                        if report.balance_result else None
                                    )
                                    report.geographic_result = self._run_geographic_check(dataset)
                                    report.metadata_score    = self._compute_metadata_score(dataset)
                                    report.quality_score     = self._compute_composite_score(report)
                                    report.suggested_fixes   = self._generate_fixes(report, dataset)
                                    report.is_partial        = True   # no blur/dup/corrupt
                                    logger.info("image_meta_analysis_complete", extra={
                                        "canonical_id": dataset.canonical_id,
                                        "classes": len(meta_result.images_per_class),
                                        "total_images": sum(meta_result.images_per_class.values()),
                                    })
                                else:
                                    logger.debug("image_meta_analysis_failed", extra={
                                        "canonical_id": dataset.canonical_id,
                                        "error": meta_result.error,
                                    })
                                    report.is_partial     = True
                                    report.error          = meta_result.error
                                    # Fallback: extract from description — no auth needed
                                    _desc_info = self._extract_from_description(
                                        dataset.description or "", dataset.title or ""
                                    )
                                    report.total_images_est = _desc_info.get("total_images")
                                    report.num_classes_est  = _desc_info.get("num_classes")
                                report.metadata_score = self._compute_metadata_score(dataset)
                                report.quality_score  = report.metadata_score
                        else:
                            # ── Deep path: full download + quality checks ──────
                            # Wrapped in asyncio.wait_for so oversized datasets
                            # don't block the server indefinitely.  On timeout
                            # we fall back to the metadata-only path so the user
                            # still gets class distribution.
                            try:
                                image_sample = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        self._image_loader.load_from_kaggle, ref
                                    ),
                                    timeout=self.timeout_seconds,
                                )
                            except asyncio.TimeoutError:
                                logger.warning("kaggle_download_timeout", extra={
                                    "canonical_id": dataset.canonical_id,
                                    "timeout_s": self.timeout_seconds,
                                })
                                image_sample = None
                                report.error = (
                                    f"Dataset too large to download within "
                                    f"{int(self.timeout_seconds)}s — "
                                    "showing class counts from file listing."
                                )
                                report.is_partial = True
                                # Fallback 1: try metadata listing
                                if owner and name:
                                    meta_result = await asyncio.to_thread(
                                        self._image_loader.load_class_counts_from_kaggle_meta,
                                        owner, name,
                                    )
                                    if meta_result.success:
                                        report.balance_result = self._balance_det.detect_from_class_counts(
                                            meta_result.images_per_class
                                        )
                                    else:
                                        # Fallback 2: description text extraction
                                        _desc_info = self._extract_from_description(
                                            dataset.description or "", dataset.title or ""
                                        )
                                        report.total_images_est = _desc_info.get("total_images")
                                        report.num_classes_est  = _desc_info.get("num_classes")

                elif source == "huggingface" and dataset.source_url:
                    # Always extract image info from description text (no auth, no download)
                    _desc_info = self._extract_from_description(
                        dataset.description or "", dataset.title or ""
                    )
                    report.total_images_est = _desc_info.get("total_images")
                    report.num_classes_est  = _desc_info.get("num_classes")

                    if not _hf_ok:
                        logger.debug(
                            "image_deep_analysis_skipped",
                            extra={
                                "reason": "datasets library not installed",
                                "fix": "pip install datasets",
                                "canonical_id": dataset.canonical_id,
                            },
                        )
                        # FIX: metadata-only path for HF when datasets lib not installed
                        # Run geographic check + metadata score so at least basic signals flow
                        report.geographic_result = self._run_geographic_check(dataset)
                        report.metadata_score    = self._compute_metadata_score(dataset)
                        report.quality_score     = report.metadata_score
                        report.suggested_fixes   = self._generate_fixes(report, dataset)
                        report.is_partial        = True

                    elif not self.download_images:
                        # FIX: HF metadata-only path (light analysis engine).
                        # Previously this branch was missing — HF image datasets got
                        # ZERO analysis when download_images=False.
                        # Now: use description-text extraction + geographic check
                        # + metadata score so the ranking has real signals.
                        hf_id = (dataset.source_url
                                 .replace("https://huggingface.co/datasets/", "")
                                 .replace("https://www.huggingface.co/datasets/", "")
                                 .rstrip("/"))
                        # Try loading class counts from HF dataset card metadata
                        # (no download — just the dataset card split info)
                        try:
                            _hf_meta = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self._image_loader.load_class_counts_from_hf_meta,
                                    hf_id,
                                ),
                                timeout=min(self.timeout_seconds, 8.0),
                            )
                            if _hf_meta and _hf_meta.success:
                                report.balance_result = self._balance_det.detect_from_class_counts(
                                    _hf_meta.images_per_class
                                )
                                report.balance_score = (
                                    report.balance_result.score if report.balance_result else None
                                )
                                # Override desc-based estimates with real counts if available
                                _real_total = sum(_hf_meta.images_per_class.values())
                                if _real_total > 0:
                                    report.total_images_est = _real_total
                                    report.num_classes_est  = len(_hf_meta.images_per_class)
                        except (asyncio.TimeoutError, Exception) as _hf_err:
                            logger.debug("hf_meta_load_failed", extra={
                                "canonical_id": dataset.canonical_id,
                                "error": str(_hf_err)[:100],
                            })
                        report.geographic_result = self._run_geographic_check(dataset)
                        report.metadata_score    = self._compute_metadata_score(dataset)
                        report.quality_score     = self._compute_composite_score(report)
                        report.suggested_fixes   = self._generate_fixes(report, dataset)
                        report.is_partial        = True

                    else:  # download_images=True (deep analysis)
                        # FIX v2.2.0 (Issue 2): prefer extra["hf_dataset_id"] set by
                        # the adapter (exact dataset_id). Only fall back to URL-parsing
                        # if hf_dataset_id is absent (handles legacy records).
                        hf_id = (
                            _extra.get("hf_dataset_id")
                            if isinstance(_extra, dict) and _extra.get("hf_dataset_id")
                            else (dataset.source_url or "")
                                 .replace("https://huggingface.co/datasets/", "")
                                 .replace("https://www.huggingface.co/datasets/", "")
                                 .rstrip("/")
                        )
                        try:
                            image_sample = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self._image_loader.load_from_huggingface, hf_id
                                ),
                                timeout=self.timeout_seconds,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("hf_download_timeout", extra={
                                "canonical_id": dataset.canonical_id,
                                "timeout_s": self.timeout_seconds,
                            })
                            image_sample = None
                            report.is_partial = True

                # Step 2: Run all detectors + per-class stats on the downloaded sample
                if image_sample and image_sample.success:
                    try:
                        class_paths = image_sample.class_image_paths   # sampled paths
                        all_paths   = [p for paths in class_paths.values() for p in paths]

                        # Balance: use REAL counts per folder — not sampled counts
                        report.balance_result = self._balance_det.detect_from_class_counts(
                            image_sample.images_per_class
                        )

                        # Duplicates: hash ALL images, not just the sample
                        class_hashes = {
                            cls: info.all_hashes
                            for cls, info in image_sample.class_info.items()
                        }
                        report.duplicate_result = self._dup_det.detect_from_image_hashes(
                            class_hashes
                        )

                        # Blur: sample is fine — checking every image is too slow
                        from .quality_detector import BlurDetector
                        report.blur_result = BlurDetector().detect_from_paths(all_paths[:100])

                        # Corruption and dimensions: on sampled paths
                        report.corruption_result = self._corruption_det.detect(class_paths)
                        report.dimension_result  = self._dim_checker.check(class_paths)

                        # Geographic bias: metadata only, always runs
                        report.geographic_result = self._run_geographic_check(dataset)

                        # NEW v3.3.0: per-class stats (blur/dup/corrupt counts per class)
                        report.per_class_stats = self._compute_per_class_image_stats(image_sample)

                        # Scores from real data
                        report.balance_score  = report.balance_result.score if report.balance_result else None
                        report.uniqueness_score = (
                            100.0 - report.duplicate_result.duplicate_percentage
                            if report.duplicate_result else None
                        )
                        report.metadata_score = self._compute_metadata_score(dataset)
                        report.quality_score  = self._compute_composite_score(report)
                        report.suggested_fixes = self._generate_fixes(report, dataset)

                        logger.info("analysis_complete", extra={
                            "canonical_id": dataset.canonical_id,
                            "quality_score": report.quality_score,
                            "source": f"{source}_deep",
                            "classes": image_sample.total_classes,
                            "images_sampled": image_sample.total_images_sampled,
                            "structure": image_sample.structure_detected,
                            "per_class_count": len(report.per_class_stats),
                        })
                    finally:
                        image_sample.cleanup()

                # Step 3: Log download failure, fall through to existing checks
                if image_sample and not image_sample.success:
                    logger.info("image_deep_analysis_failed", extra={
                        "canonical_id": dataset.canonical_id,
                        "source": source,
                        "error": image_sample.error,
                    })
                    if not report.is_partial:
                        report.is_partial = True

                # Fallback: use pre-fetched local paths from dataset.extra
                self._run_image_checks(dataset, report)
                report.geographic_result = self._run_geographic_check(dataset)
                report.balance_score  = report.balance_result.score if report.balance_result else None
                report.metadata_score = self._compute_metadata_score(dataset)
                report.quality_score  = self._compute_composite_score(report)
                report.suggested_fixes = self._generate_fixes(report, dataset)

            # ── 2. Tabular / text pipeline ────────────────────────────────────
            # Note: No 'else' or 'elif' — this runs in addition to image analysis
            # for multi-modal datasets.

            # Resolve best download URL.
            # Priority: df arg > downloadable source_url > extra.download_url
            #         > HuggingFace datasets library (new) > none
            _extra = dataset.extra or {}
            _download_url = _extra.get("download_url") if isinstance(_extra, dict) else None
            _hf_dataset_id = _extra.get("hf_dataset_id") if isinstance(_extra, dict) else None

            if df is not None:
                sample_result = self._loader.load_from_dataframe(df)
            elif self._is_downloadable_url(dataset.source_url):
                sample_result = await self._loader.load_from_url(dataset.source_url)
            elif _download_url and self._is_downloadable_url(_download_url):
                # OpenML: extra.download_url points to real CSV endpoint.
                sample_result = await self._loader.load_from_url(_download_url)
            elif _hf_dataset_id and getattr(dataset, "source", "") == "huggingface" and (
                # FIX v2.4.0: guard against audio datasets loading via the tabular
                # path. Audio datasets have audio features (binary arrays) that
                # can't be loaded as pandas DataFrames. Route them to the new
                # hf_audio_metadata_analysis path (the elif block below) instead.
                # Only use the tabular path when the dataset is NOT audio-only.
                Modality.AUDIO not in (dataset.modalities or set())
                or bool(set(dataset.modalities or set()) - {Modality.AUDIO})  # has non-audio modality too
            ):
                # FIX v3.5.0: HuggingFace tabular deep analysis path.
                # Previously ALL HuggingFace tabular datasets fell through to
                # metadata-only because source_url is a landing page (blocked by
                # _WEBPAGE_URL_PREFIXES) and no download_url was set in extra.
                # Now we use the `datasets` library directly via hf_dataset_id.
                # This gives AnalysisEngine access to real rows, enabling full
                # column analysis, class balance, duplicate detection, and
                # completeness scoring — identical to the Kaggle tabular path.
                try:
                    import datasets as _hf_datasets  # noqa: PLC0415
                    is_gated = _extra.get("gated", False)
                    if is_gated:
                        raise ValueError("Dataset is gated — requires manual access approval")

                    def _hf_load_tabular(hf_id: str, max_rows: int) -> "Any":
                        """
                        Load up to max_rows rows from a HuggingFace dataset using
                        streaming=True so the full dataset is NEVER downloaded.

                        FIX v3.8.0 (Bug 2 — only first HF dataset fully analysed):
                        The previous implementation used streaming=False + to_pandas().
                        With streaming=False the `datasets` library downloads and caches
                        the entire dataset before returning — this can be several GB and
                        takes minutes for large HF datasets.  When two deep-analysis tasks
                        run concurrently (asyncio.gather), the first dataset download fills
                        the thread pool and holds the asyncio.wait_for timeout, leaving
                        the second dataset with 0s remaining → it times out immediately
                        and returns no analysis data at all.

                        With streaming=True we pull only max_rows records over the network
                        and stop.  The network transfer is proportional to max_rows (e.g.
                        500 rows ≈ a few hundred KB), completing in seconds even on slow
                        connections.  Both concurrent tasks now finish well within the
                        per-dataset timeout, so both datasets receive full analysis.

                        This mirrors the approach already used correctly in
                        datascout/analysis/sample_loader.py::hf_tabular_load().
                        """
                        import pandas as _pd
                        ds = _hf_datasets.load_dataset(
                            hf_id,
                            split="train",
                            streaming=True,          # ← KEY FIX: never download full dataset
                            trust_remote_code=False,
                        )
                        rows: list = []
                        for item in ds:
                            rows.append(item)
                            if len(rows) >= max_rows:
                                break
                        if not rows:
                            raise ValueError(
                                "No rows returned from HuggingFace streaming load"
                            )
                        return _pd.DataFrame(rows)

                    _hf_df = await asyncio.wait_for(
                        asyncio.to_thread(_hf_load_tabular, _hf_dataset_id, self.sample_rows),
                        # FIX v3.8.0: raised from 60s → 120s per dataset.
                        # With streaming=True each dataset only fetches max_rows records
                        # (≈ a few hundred KB), so two concurrent analyses easily finish
                        # within this window without starving each other.
                        timeout=min(self.timeout_seconds, 120.0),
                    )
                    sample_result = self._loader.load_from_dataframe(_hf_df)
                    logger.info("hf_tabular_deep_analysis_loaded", extra={
                        "canonical_id": dataset.canonical_id,
                        "hf_id": _hf_dataset_id,
                        "rows": len(_hf_df),
                        "streaming": True,
                    })
                except ImportError:
                    logger.debug("hf_tabular_deep_skipped_no_datasets_lib", extra={
                        "canonical_id": dataset.canonical_id,
                        "fix": "pip install datasets",
                    })
                    sample_result = SampleLoadResult(
                        success=False,
                        error="datasets library not installed — pip install datasets",
                    )
                except Exception as _hf_err:
                    logger.debug("hf_tabular_load_failed", extra={
                        "canonical_id": dataset.canonical_id,
                        "error": str(_hf_err)[:150],
                    })
                    sample_result = SampleLoadResult(
                        success=False,
                        error=f"HuggingFace load failed: {str(_hf_err)[:100]}",
                    )
            elif _hf_dataset_id and getattr(dataset, "source", "") == "huggingface":
                # FIX v2.4.0 (Issue 2 / audio deep analysis):
                # Audio HF datasets (Modality.AUDIO) fall here because they are
                # not IMAGE and the URL is a landing page (blocked by _WEBPAGE_URL_PREFIXES).
                # The previous elif for HF tabular only fires when _hf_dataset_id is set
                # AND the dataset loaded successfully as a DataFrame. For audio datasets,
                # load_dataset returns audio features (binary arrays), not tabular rows.
                #
                # Instead of downloading audio files, use the HuggingFace Hub REST API
                # to get DatasetInfo: split sizes (num_examples), feature schema, and
                # file count. This is a lightweight metadata call (~100ms, no auth needed
                # for public datasets) that gives us:
                #   - row_count:    total num_examples across all splits
                #   - column_names: feature names from dataset_info.features
                #   - file_size:    total dataset size in bytes (if available)
                #
                # We synthesise a SampleLoadResult with a minimal 1-row DataFrame
                # whose columns match the feature names. This allows the column analyzer
                # to run schema detection and completeness scoring without any audio download.
                try:
                    from huggingface_hub import dataset_info as _hf_dataset_info_fn  # noqa: PLC0415
                    import pandas as _pd  # noqa: PLC0415

                    _ds_info = await asyncio.wait_for(
                        asyncio.to_thread(_hf_dataset_info_fn, _hf_dataset_id),
                        timeout=15.0,
                    )

                    # Sum num_examples across all splits
                    _total_examples = 0
                    _split_names = []
                    _ds_splits = getattr(_ds_info, "splits", None) or {}
                    if isinstance(_ds_splits, dict):
                        for _sname, _sinfo in _ds_splits.items():
                            _n = getattr(_sinfo, "num_examples", 0) or 0
                            _total_examples += _n
                            _split_names.append(f"{_sname}:{_n:,}")
                    elif hasattr(_ds_splits, "__iter__"):
                        for _sinfo in _ds_splits:
                            _n = getattr(_sinfo, "num_examples", 0) or 0
                            _total_examples += _n
                            _sname = getattr(_sinfo, "name", "split")
                            _split_names.append(f"{_sname}:{_n:,}")

                    # Feature names from dataset_info.features
                    _features = getattr(_ds_info, "features", None) or {}
                    _feature_names = list(_features.keys()) if isinstance(_features, dict) else []

                    # Total dataset size in bytes
                    _ds_size_bytes = getattr(_ds_info, "dataset_size", None)

                    # Patch the RawDataset with real row count if we got it
                    # (avoids the "Dataset size unknown" warning downstream)
                    if _total_examples > 0:
                        try:
                            object.__setattr__(dataset, "row_count", _total_examples)
                            object.__setattr__(dataset, "has_size_info", True)
                        except Exception:
                            pass  # RawDataset may be frozen — best effort

                    # Build a synthetic 1-row DataFrame so column_analyzer can
                    # detect schema and feature types without audio file access.
                    if _feature_names:
                        _synthetic_row = {col: None for col in _feature_names}
                        _synthetic_df = _pd.DataFrame([_synthetic_row])
                    else:
                        # Fallback: single-column DataFrame to signal "tabular" path
                        _synthetic_df = _pd.DataFrame([{"audio_samples": _total_examples}])

                    sample_result = self._loader.load_from_dataframe(_synthetic_df)
                    # Annotate with real row count so downstream stats are correct
                    if _total_examples > 0:
                        try:
                            object.__setattr__(sample_result, "rows", _total_examples)
                        except Exception:
                            pass

                    logger.info("hf_audio_metadata_analysis_loaded", extra={
                        "canonical_id": dataset.canonical_id,
                        "hf_id": _hf_dataset_id,
                        "total_examples": _total_examples,
                        "splits": ", ".join(_split_names[:5]),
                        "features": _feature_names[:8],
                        "size_bytes": _ds_size_bytes,
                    })

                except ImportError:
                    logger.debug("hf_audio_skipped_no_hub_lib", extra={
                        "canonical_id": dataset.canonical_id,
                        "fix": "pip install huggingface_hub",
                    })
                    sample_result = SampleLoadResult(
                        success=False,
                        error="huggingface_hub not installed — pip install huggingface_hub",
                    )
                except Exception as _hf_audio_err:
                    logger.debug("hf_audio_metadata_load_failed", extra={
                        "canonical_id": dataset.canonical_id,
                        "error": str(_hf_audio_err)[:150],
                    })
                    sample_result = SampleLoadResult(
                        success=False,
                        error=f"HF audio metadata load failed: {str(_hf_audio_err)[:100]}",
                    )

            else:
                sample_result = SampleLoadResult(
                    success=False,
                    error=(
                        "URL is a dataset landing page, not a downloadable file. "
                        "Using metadata-only analysis."
                        if dataset.source_url
                        else "No source URL available."
                    ),
                )

            report.sample_result = sample_result

            if not sample_result.success or sample_result.data is None:
                report.is_partial = True
                if dataset.source_url and self._is_downloadable_url(dataset.source_url):
                    logger.warning("analysis_no_sample", extra={
                        "canonical_id": dataset.canonical_id,
                        "error": sample_result.error,
                    })
                else:
                    logger.debug("analysis_metadata_only", extra={
                        "canonical_id": dataset.canonical_id,
                    })
                report.metadata_score    = self._compute_metadata_score(dataset)
                report.quality_score     = report.metadata_score
                report.geographic_result = self._run_geographic_check(dataset)
                report.suggested_fixes   = self._generate_fixes(report, dataset)
                return report

            loaded_df = sample_result.data

            # Step 2: Column analysis
            column_result        = self._col_analyzer.analyze(loaded_df)
            report.column_result = column_result
            if column_result.error:
                report.is_partial = True

            # Step 3: Class balance
            is_classification = (
                TaskType.CLASSIFICATION           in dataset.task_types or
                TaskType.BINARY_CLASSIFICATION    in dataset.task_types or
                TaskType.IMAGE_CLASSIFICATION     in dataset.task_types or
                TaskType.TEXT_CLASSIFICATION      in dataset.task_types or
                TaskType.MULTI_LABEL_CLASSIFICATION in dataset.task_types or
                Modality.TABULAR in dataset.modalities
            )
            if is_classification or not dataset.task_types:
                report.balance_result = self._balance_det.detect(loaded_df, self.target_column)

            # Step 4: Duplicate detection
            report.duplicate_result = self._dup_det.detect(loaded_df)

            # Step 5: Geographic bias
            report.geographic_result = self._run_geographic_check(dataset)

            # Step 6: Scores
            report.completeness_score = self._compute_completeness_score(column_result, dataset)
            report.balance_score      = report.balance_result.score if report.balance_result else None
            report.uniqueness_score   = (
                100.0 - report.duplicate_result.duplicate_percentage
                if report.duplicate_result else None
            )
            report.metadata_score  = self._compute_metadata_score(dataset)
            report.quality_score   = self._compute_composite_score(report)

            # Step 7: Recommendations
            report.suggested_fixes = self._generate_fixes(report, dataset)

            logger.info("analysis_complete", extra={
                "canonical_id": dataset.canonical_id,
                "quality_score": report.quality_score,
                "rows_analyzed": sample_result.rows,
                "fixes_count": len(report.suggested_fixes),
            })

        except Exception as e:
            logger.error("analysis_engine_error", extra={
                "canonical_id": dataset.canonical_id, "error": str(e)
            }, exc_info=True)
            report.is_partial = True
            report.error = str(e)[:200]

        return report

    # ── Image pipeline (fallback for pre-fetched local paths) ──────────────────

    def _run_image_checks(self, dataset: RawDataset, report: AnalysisReport) -> None:
        extra       = dataset.extra or {}
        class_paths = extra.get("class_image_paths", {})
        local_path  = extra.get("local_path")
        flat_paths  = extra.get("image_paths", [])

        if not class_paths and local_path:
            image_sample: ImageSampleResult = self._image_loader.load(local_path)
            if image_sample.success:
                class_paths = image_sample.class_image_paths
                report.balance_result = self._balance_det.detect_from_class_counts(
                    image_sample.images_per_class
                )
                # Per-class stats from local load
                report.per_class_stats = self._compute_per_class_image_stats(image_sample)
            else:
                logger.warning("image_sample_load_failed", extra={
                    "canonical_id": dataset.canonical_id, "error": image_sample.error,
                })
                report.is_partial = True
        elif class_paths:
            report.balance_result = self._balance_det.detect_from_class_counts(
                {cls: len(paths) for cls, paths in class_paths.items()}
            )

        all_paths = flat_paths or [p for paths in class_paths.values() for p in paths]

        if all_paths:
            from .quality_detector import BlurDetector
            report.blur_result = BlurDetector().detect_from_paths(all_paths[:50])

        consistency_det = ContentConsistencyDetector()
        if class_paths:
            report.content_result = consistency_det.detect_from_labeled_paths(class_paths)
        elif all_paths and (dataset.description or dataset.title):
            report.content_result = consistency_det.detect_from_description(
                dataset.description or dataset.title, all_paths[:30]
            )

        if class_paths:
            report.corruption_result = self._corruption_det.detect(class_paths)
            report.dimension_result  = self._dim_checker.check(class_paths)
        elif all_paths:
            report.corruption_result = self._corruption_det.detect({"_all": all_paths})
            report.dimension_result  = self._dim_checker.check({"_all": all_paths})

    def _run_geographic_check(self, dataset: RawDataset) -> GeographicBiasResult:
        return self._geo_det.detect(
            description=dataset.description or "",
            tags=list(dataset.tags) if dataset.tags else [],
            title=dataset.title or "",
            author=getattr(dataset, "author", "") or "",
        )

    # ── Score computation ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_completeness_score(column_result: ColumnAnalysisResult,
                                    dataset: RawDataset) -> float:
        if column_result and not column_result.error:
            return round(100.0 - column_result.overall_missing_percentage, 2)
        return round(dataset.metadata_completeness * 100, 2)

    @staticmethod
    def _compute_metadata_score(dataset: RawDataset) -> float:
        """
        Metadata quality score (0-100).

        For image datasets: includes an image-scale bonus derived from the
        description text.  A dataset with 41K images scores higher than one
        with 500 images — even before any download — because scale directly
        affects training utility.
        """
        score  = 30.0 if dataset.has_description  else 0.0
        score += 20.0 if dataset.has_schema_info  else 0.0
        score += 20.0 if dataset.has_size_info    else 0.0
        score += 15.0 if dataset.has_license_info else 0.0
        score += min(dataset.tag_count / 10.0 * 15.0, 15.0)

        # Image scale bonus (image datasets only) — replaces has_size_info signal
        # which is always False for image datasets (they have no row_count).
        modalities = getattr(dataset, "modalities", None) or []
        is_image = any(
            str(m).lower() in ("image", "modality.image", "<modality.image: 'image'>")
            for m in modalities
        )
        if is_image:
            desc  = dataset.description or ""
            title = dataset.title or ""
            extracted = AnalysisEngine._extract_from_description(desc, title)
            n = extracted.get("total_images", 0)
            if   n >= 50_000: scale_bonus = 20.0
            elif n >= 10_000: scale_bonus = 15.0
            elif n >=  1_000: scale_bonus = 10.0
            elif n >=    100: scale_bonus =  5.0
            else:             scale_bonus =  0.0
            # Remove the has_size_info component (always 0 for images) and
            # replace with the scale bonus.
            score = score - 0.0 + scale_bonus   # has_size_info already 0

        return round(score, 2)

    @staticmethod
    def _compute_composite_score(report: AnalysisReport) -> float:
        components: list[tuple[float, float]] = []
        if report.completeness_score is not None: components.append((report.completeness_score, WEIGHT_COMPLETENESS))
        if report.balance_score      is not None: components.append((report.balance_score,      WEIGHT_BALANCE))
        if report.uniqueness_score   is not None: components.append((report.uniqueness_score,   WEIGHT_UNIQUENESS))
        if report.metadata_score     is not None: components.append((report.metadata_score,     WEIGHT_METADATA))
        if not components:
            return 0.0
        total_weight = sum(w for _, w in components)
        return round(sum(s * w for s, w in components) / total_weight, 2)

    # ── Fix recommendations ────────────────────────────────────────────────────

    @staticmethod
    def _generate_fixes(report: AnalysisReport, dataset: RawDataset) -> list[str]:
        fixes: list[str] = []

        if report.column_result and not report.column_result.error:
            for col in report.column_result.columns:
                if col.missing_percentage > 20.0:
                    fixes.append(
                        f"Column '{col.name}' has {col.null_count:,} null values "
                        f"({col.missing_percentage:.1f}%). "
                        f"Consider imputation or dropping this column."
                    )

        if report.balance_result and report.balance_result.warning_message:
            fixes.append(report.balance_result.warning_message)

        if report.duplicate_result and report.duplicate_result.has_duplicates:
            if report.duplicate_result.warning_message:
                fixes.append(report.duplicate_result.warning_message)

        if report.blur_result and report.blur_result.blurry_percentage >= 10.0:
            if report.blur_result.warning_message:
                fixes.append(report.blur_result.warning_message)

        if report.content_result:
            if report.content_result.flagged_classes:
                fixes.append(
                    f"Content mismatch in classes: {', '.join(report.content_result.flagged_classes)}. "
                    f"Images may not match their labels — verify dataset labeling."
                )
            elif report.content_result.mismatch_percentage >= 10.0:
                fixes.append(
                    f"{report.content_result.mismatch_percentage:.1f}% of sampled images "
                    f"may not match their labels."
                )

        if report.geographic_result and report.geographic_result.is_single_region:
            fixes.append(
                report.geographic_result.warning_message or
                "Dataset appears collected from a single region — "
                "may not generalise globally."
            )

        if report.dimension_result and report.dimension_result.warning_message:
            fixes.append(report.dimension_result.warning_message)

        if report.corruption_result and report.corruption_result.warning_message:
            fixes.append(report.corruption_result.warning_message)

        if not dataset.has_description:
            fixes.append("Add a dataset description to improve discoverability.")
        if not dataset.has_license_info:
            fixes.append("Specify a license to clarify usage rights.")
        if dataset.tag_count < 3:
            fixes.append("Add more descriptive tags to improve search relevance.")

        return fixes[:10]


def create_analysis_engine(sample_rows: int = 500,
                            timeout_seconds: float = 600.0,
                            download_images: bool = True,
                            deep_mode: bool = False) -> AnalysisEngine:
    # FIX v2.1.0 (Issue 6): deep_mode=True passes to ImageSampleLoader so the
    # per-class image cap is raised from 50 to 2000 during deep analysis.
    return AnalysisEngine(
        sample_rows=sample_rows,
        timeout_seconds=timeout_seconds,
        download_images=download_images,
        deep_mode=deep_mode,
    )