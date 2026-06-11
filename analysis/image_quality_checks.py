"""
Image-specific quality detectors for DATASCOUT Phase 8.

Detectors:
  - GeographicBiasDetector  — single-region collection bias
  - ImageDimensionChecker   — size consistency and CNN suitability
  - CorruptionDetector      — unreadable/broken image files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Optional PIL
try:
    from PIL import Image as _PILImage, UnidentifiedImageError as _UnidentifiedImageError
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ══════════════════════════════════════════════════════════════════════════════
# Geographic Bias Detector
# ══════════════════════════════════════════════════════════════════════════════

REGION_KEYWORDS: dict[str, list[str]] = {
    "india": [
        "india", "indian", "maharashtra", "punjab", "karnataka",
        "tamil", "bengal", "gujarat", "rajasthan", "icar", "iari",
    ],
    "china": [
        "china", "chinese", "beijing", "shanghai", "guangdong",
        "sichuan", "yunnan", "caas",
    ],
    "usa": [
        "usa", "american", "california", "florida", "iowa", "usda",
        "cornell", "michigan state", "purdue",
    ],
    "europe": [
        "netherlands", "germany", "france", "spain", "italy",
        "wageningen", "european", "eu funded",
    ],
    "africa": [
        "africa", "african", "kenya", "nigeria", "ethiopia",
        "ghana", "tanzania", "cgiar",
    ],
    "brazil": [
        "brazil", "brazilian", "embrapa", "são paulo", "minas",
    ],
    "australia": [
        "australia", "australian", "csiro", "queensland",
    ],
    "japan": [
        "japan", "japanese", "tokyo", "osaka", "naro",
    ],
}


@dataclass
class GeographicBiasResult:
    detected_regions: list[str] = field(default_factory=list)
    is_single_region: bool = False
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    warning_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "detected_regions": self.detected_regions,
            "is_single_region": self.is_single_region,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "warning_message": self.warning_message,
        }


class GeographicBiasDetector:
    """
    Scans dataset metadata (description, tags, title, author) for geographic
    keywords to detect single-region collection bias.
    """

    def detect(
        self,
        description: str = "",
        tags: Optional[list[str]] = None,
        title: str = "",
        author: str = "",
    ) -> GeographicBiasResult:
        """
        Always returns a result — never raises.

        Args:
            description: Full dataset description text.
            tags:        List of dataset tags/keywords.
            title:       Dataset title.
            author:      Dataset author name.
        """
        try:
            return self._detect(description, tags or [], title, author)
        except Exception as e:
            return GeographicBiasResult(
                confidence=0.0,
                evidence=[f"Detection error: {e}"],
            )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _detect(
        self, description: str, tags: list[str], title: str, author: str
    ) -> GeographicBiasResult:
        # Build a single lowercased corpus from all sources
        corpus_parts = [
            ("description", description),
            ("tags", " ".join(tags)),
            ("title", title),
            ("author", author),
        ]

        region_hits: dict[str, list[str]] = {}  # region → evidence strings

        for region, keywords in REGION_KEYWORDS.items():
            for source_name, text in corpus_parts:
                if not text:
                    continue
                lower_text = text.lower()
                for kw in keywords:
                    if kw in lower_text:
                        region_hits.setdefault(region, []).append(
                            f"Found '{kw}' in {source_name}"
                        )

        detected_regions = list(region_hits.keys())
        all_evidence = [ev for evs in region_hits.values() for ev in evs]

        if not detected_regions:
            return GeographicBiasResult(
                detected_regions=[],
                is_single_region=False,
                confidence=0.0,
                evidence=[],
                warning_message=None,
            )

        # Confidence: scale by evidence density (cap at 1.0)
        total_hits = sum(len(v) for v in region_hits.values())
        confidence = min(1.0, 0.3 + total_hits * 0.1)

        if len(detected_regions) == 1:
            region = detected_regions[0]
            return GeographicBiasResult(
                detected_regions=detected_regions,
                is_single_region=True,
                confidence=confidence,
                evidence=all_evidence,
                warning_message=(
                    f"Dataset appears collected in {region} only — "
                    "may not generalize globally."
                ),
            )

        return GeographicBiasResult(
            detected_regions=detected_regions,
            is_single_region=False,
            confidence=confidence,
            evidence=all_evidence,
            warning_message=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Image Dimension Checker
# ══════════════════════════════════════════════════════════════════════════════

_MIN_DIMENSION = 32       # px — smallest usable for CNNs
_MAX_DIMENSION = 4096     # px — memory risk threshold


@dataclass
class DimensionCheckResult:
    is_consistent: Optional[bool] = None   # None if PIL unavailable
    dominant_size: Optional[tuple] = None  # (width, height)
    size_variance: float = 0.0             # 0=all same, 1=all different
    unique_sizes: int = 0
    undersized_count: int = 0              # < 32×32
    oversized_count: int = 0              # > 4096×4096
    warning_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "dominant_size": list(self.dominant_size) if self.dominant_size else None,
            "size_variance": self.size_variance,
            "unique_sizes": self.unique_sizes,
            "undersized_count": self.undersized_count,
            "oversized_count": self.oversized_count,
            "warning_message": self.warning_message,
        }


class ImageDimensionChecker:
    """
    Checks sampled images for dimension consistency and CNN suitability.
    Uses PIL to read image headers — no full decode needed.
    """

    def check(
        self, class_image_paths: dict[str, list[str]]
    ) -> DimensionCheckResult:
        """Always returns a result — never raises."""
        try:
            return self._check(class_image_paths)
        except Exception as e:
            return DimensionCheckResult(
                warning_message=f"Dimension check failed: {e}"
            )

    def _check(
        self, class_image_paths: dict[str, list[str]]
    ) -> DimensionCheckResult:
        if not _HAS_PIL:
            return DimensionCheckResult(
                is_consistent=None,
                warning_message="Pillow not installed — dimension check skipped.",
            )

        all_paths = [
            p for paths in class_image_paths.values() for p in paths
        ]
        if not all_paths:
            return DimensionCheckResult(
                is_consistent=True,
                warning_message=None,
            )

        sizes: list[tuple[int, int]] = []
        for path in all_paths:
            try:
                with _PILImage.open(path) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception:
                continue  # unreadable handled by CorruptionDetector

        if not sizes:
            return DimensionCheckResult(is_consistent=None)

        from collections import Counter

        size_counts = Counter(sizes)
        dominant_size = size_counts.most_common(1)[0][0]
        unique_sizes = len(size_counts)

        # Variance: fraction of images that differ from dominant
        non_dominant = sum(v for k, v in size_counts.items() if k != dominant_size)
        size_variance = non_dominant / len(sizes) if sizes else 0.0

        is_consistent = unique_sizes == 1

        undersized = sum(
            1 for w, h in sizes if w < _MIN_DIMENSION or h < _MIN_DIMENSION
        )
        oversized = sum(
            1 for w, h in sizes if w > _MAX_DIMENSION or h > _MAX_DIMENSION
        )

        warnings = []
        if not is_consistent:
            warnings.append(
                f"Mixed image dimensions detected ({unique_sizes} unique sizes). "
                "Resizing required before training — choose interpolation carefully."
            )
        if undersized:
            warnings.append(
                f"{undersized} image(s) are smaller than {_MIN_DIMENSION}×{_MIN_DIMENSION}px "
                "— too small for meaningful CNN features."
            )
        if oversized:
            warnings.append(
                f"{oversized} image(s) exceed {_MAX_DIMENSION}×{_MAX_DIMENSION}px "
                "— may cause memory issues during training."
            )

        return DimensionCheckResult(
            is_consistent=is_consistent,
            dominant_size=dominant_size,
            size_variance=round(size_variance, 4),
            unique_sizes=unique_sizes,
            undersized_count=undersized,
            oversized_count=oversized,
            warning_message=" | ".join(warnings) if warnings else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Corruption Detector
# ══════════════════════════════════════════════════════════════════════════════

_CORRUPTION_WARNING_THRESHOLD = 5.0  # percent


@dataclass
class CorruptionResult:
    corrupted_count: int = 0
    total_checked: int = 0
    corruption_percentage: float = 0.0
    corrupted_files: list[str] = field(default_factory=list)
    warning_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "corrupted_count": self.corrupted_count,
            "total_checked": self.total_checked,
            "corruption_percentage": self.corruption_percentage,
            "corrupted_files": self.corrupted_files,
            "warning_message": self.warning_message,
        }


class CorruptionDetector:
    """
    Detects unreadable/broken image files using PIL's verify() method.
    verify() reads the full file but does not decode pixels — fast.
    """

    def detect(
        self, class_image_paths: dict[str, list[str]]
    ) -> CorruptionResult:
        """Always returns a result — never raises."""
        try:
            return self._detect(class_image_paths)
        except Exception as e:
            return CorruptionResult(
                warning_message=f"Corruption check failed: {e}"
            )

    def _detect(
        self, class_image_paths: dict[str, list[str]]
    ) -> CorruptionResult:
        if not _HAS_PIL:
            return CorruptionResult(
                warning_message="Pillow not installed — corruption check skipped."
            )

        all_paths = [
            p for paths in class_image_paths.values() for p in paths
        ]
        if not all_paths:
            return CorruptionResult(total_checked=0)

        corrupted: list[str] = []

        for path in all_paths:
            try:
                # verify() reads the whole file but skips pixel decode
                with _PILImage.open(path) as img:
                    img.verify()
            except Exception:
                corrupted.append(path)

        total = len(all_paths)
        pct = (len(corrupted) / total * 100) if total else 0.0

        warning = None
        if pct > _CORRUPTION_WARNING_THRESHOLD:
            warning = (
                f"{len(corrupted)}/{total} sampled images ({pct:.1f}%) are corrupted. "
                "Consider re-downloading the dataset or filtering corrupted files before training."
            )

        return CorruptionResult(
            corrupted_count=len(corrupted),
            total_checked=total,
            corruption_percentage=round(pct, 2),
            corrupted_files=corrupted,
            warning_message=warning,
        )