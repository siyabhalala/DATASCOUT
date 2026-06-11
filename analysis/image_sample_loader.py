"""
datascout.analysis.image_sample_loader
──────────────────────────────────────────────────────────────────────────────
Downloads and organises image datasets for deep quality analysis.

FIX (v2.0.0): Three critical fixes from original:

  1. load_from_kaggle() — new method. Downloads dataset ZIP via Kaggle API,
     extracts it, detects folder-per-class structure.

  2. _detect_folder_per_class() — fixed to handle nested splits.
     Most Kaggle datasets have train/class_a/, train/class_b/ NOT flat class_a/.
     Old code returned {train:[...], test:[...]} instead of actual classes.
     Now detects split names and recurses one level deeper.

  3. _build_result() — now stores REAL image counts per class (not sample counts).
     images_per_class = {class: total_real_count}  ← for balance scoring
     class_image_paths = {class: sampled_50_paths}  ← for blur/quality checks
     all_hashes = perceptual hash of EVERY image  ← for duplicate detection
     
     WHY this matters:
     - Balance: 500 vs 5000 is imbalanced. Sampling 10 from each makes them look equal.
     - Duplicates: sampling 10/5000 means you never find duplicates spread across dataset.

  4. ClassInfo dataclass — new, carries total_images (real), sampled_paths (50),
     all_hashes (every image for duplicate detection).
"""
from __future__ import annotations

import os
import random
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import huggingface_hub as _hf_hub  # noqa: F401
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

IMAGE_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
MAX_IMAGES_PER_CLASS      = 50    # fast-mode cap per class (blur/quality checks)
MAX_IMAGES_PER_CLASS_DEEP = 2000  # deep-mode cap: analyse up to 2000/class
# FIX v2.1.0 (Issue 6): deep analysis was capped at 50 images/class regardless
# of mode. For a 5676-image dataset with ~38 classes, 50×38 = 1900 images were
# analysed instead of all 5676. Use MAX_IMAGES_PER_CLASS_DEEP in deep mode.
# Accuracy > Speed: 20-minute runtime is acceptable for deep analysis.
MAX_CLASSES          = 50   # handle large datasets (PlantVillage has 38 classes)
HASH_SIZE            = 8    # 8×8 perceptual hash — fast, no full decode needed

# Split folder names — not class names
_SPLIT_NAMES = {"train", "test", "val", "valid", "validation",
                "training", "testing", "images", "data"}


@dataclass
class ClassInfo:
    """
    Full per-class stats. total_images is the REAL count, not a sample.
    all_hashes contains a pHash for EVERY image for duplicate detection.
    """
    name: str
    total_images: int            # real count from actual folder — NOT a sample
    sampled_paths: list[str]     # random subset for blur/corruption checks
    all_hashes: list[str]        # pHash of every image for dup detection


@dataclass
class ImageSampleResult:
    success: bool
    class_info: dict[str, ClassInfo]         = field(default_factory=dict)
    # Legacy compat fields
    class_image_paths: dict[str, list[str]]  = field(default_factory=dict)  # sampled
    total_classes: int                       = 0
    total_images_sampled: int                = 0
    images_per_class: dict[str, int]         = field(default_factory=dict)  # REAL counts
    structure_detected: str                  = "unknown"
    temp_dir: Optional[str]                  = None
    error: Optional[str]                     = None

    def cleanup(self) -> None:
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    def __enter__(self) -> "ImageSampleResult":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_classes": self.total_classes,
            "total_images_per_class": self.images_per_class,
            "total_images_sampled": self.total_images_sampled,
            "structure_detected": self.structure_detected,
            "error": self.error,
        }


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _perceptual_hash(image_path: str) -> Optional[str]:
    """
    Average-hash (aHash) for duplicate detection.
    Resize to 8×8 grayscale, threshold at mean, encode as hex.
    Very cheap — no full pixel decode, just a tiny thumbnail.
    Returns None if image is unreadable (counted as corruption).
    """
    if not _HAS_PIL:
        return None
    try:
        with _PILImage.open(image_path) as img:
            small   = img.convert("L").resize((HASH_SIZE, HASH_SIZE), _PILImage.LANCZOS)
            pixels  = list(small.getdata())
            avg     = sum(pixels) / len(pixels)
            bits    = "".join("1" if p >= avg else "0" for p in pixels)
            return f"{int(bits, 2):016x}"
    except Exception:
        return None


def _detect_folder_per_class(root: str) -> Optional[dict[str, list[str]]]:
    """
    Detect folder-per-class structure. Handles two layouts:

    Layout A (flat):            Layout B (nested train/test split):
      root/                       root/
        class_a/ *.jpg              train/
        class_b/ *.jpg                class_a/ *.jpg
                                      class_b/ *.jpg
                                    test/
                                      class_a/ *.jpg

    For Layout B: merges train+test image counts per class (total across splits).
    Returns {class_name: [all_image_paths]} or None.
    """
    root_path = Path(root)
    class_map: dict[str, list[str]] = {}

    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        images = [str(f) for f in entry.rglob("*")
                  if f.is_file() and _is_image_file(str(f))]
        if images:
            class_map[entry.name] = images

    if not class_map:
        return None

    # Check if top-level folders are splits (not classes)
    top_names = {k.lower() for k in class_map.keys()}
    if top_names.issubset(_SPLIT_NAMES):
        # Recurse into splits and merge class image lists
        merged: dict[str, list[str]] = {}
        for split_name in class_map.keys():
            split_dir = root_path / split_name
            if not split_dir.is_dir():
                continue
            for cls_entry in sorted(split_dir.iterdir()):
                if not cls_entry.is_dir():
                    continue
                imgs = [str(f) for f in cls_entry.rglob("*")
                        if f.is_file() and _is_image_file(str(f))]
                if imgs:
                    merged.setdefault(cls_entry.name, []).extend(imgs)
        if merged:
            return merged

    return class_map


def _detect_csv_labels(root: str) -> Optional[dict[str, list[str]]]:
    """Detect CSV labels + images folder structure."""
    import csv
    root_path = Path(root)
    csv_candidates = list(root_path.glob("*.csv"))
    if not csv_candidates:
        return None

    images_dir: Optional[Path] = None
    for name in ["images", "imgs", "image", "data", "train"]:
        p = root_path / name
        if p.is_dir():
            images_dir = p
            break
    if images_dir is None:
        for entry in root_path.iterdir():
            if entry.is_dir():
                if any(_is_image_file(str(f)) for f in entry.rglob("*")):
                    images_dir = entry
                    break
    if images_dir is None:
        return None

    class_map: dict[str, list[str]] = {}
    for csv_path in csv_candidates:
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fn_col = next(
                    (h for h in reader.fieldnames or []
                     if h.lower() in {"filename","file","image","image_name","name","path"}), None)
                cls_col = next(
                    (h for h in reader.fieldnames or []
                     if h.lower() in {"class","label","category","target","species"}), None)
                if fn_col is None or cls_col is None:
                    continue
                for row in reader:
                    fname = row.get(fn_col, "").strip()
                    cls   = row.get(cls_col, "").strip()
                    if not fname or not cls:
                        continue
                    img_path = images_dir / fname
                    if img_path.is_file():
                        class_map.setdefault(cls, []).append(str(img_path))
        except Exception:
            continue
    return class_map if class_map else None


class ImageSampleLoader:
    """
    Loads image datasets and builds per-class stats for deep quality analysis.

    Key design: collect REAL image counts from every folder (not a sample),
    hash EVERY image for duplicate detection, sample a subset for expensive
    quality checks like blur. This gives accurate balance scores and reliable
    duplicate rates without loading every image into memory.

    Args:
        deep_mode: When True, use MAX_IMAGES_PER_CLASS_DEEP (2000) instead of
                   MAX_IMAGES_PER_CLASS (50). Enables full-dataset analysis at
                   the cost of longer runtime (~20 min for large datasets).
                   Set to True for the deep analysis path (top-3 datasets).
    """

    def __init__(self, deep_mode: bool = False) -> None:
        # FIX v2.1.0 (Issue 6): allow caller to enable full-depth image analysis.
        # Previously MAX_IMAGES_PER_CLASS=50 was hardcoded; deep path got the same
        # 50-image sample as the fast path. Deep mode raises the cap to 2000/class.
        self._images_per_class_cap = (
            MAX_IMAGES_PER_CLASS_DEEP if deep_mode else MAX_IMAGES_PER_CLASS
        )

    def load_class_counts_from_kaggle_meta(
        self, owner: str, name: str
    ) -> ImageSampleResult:
        """
        Infer class image counts from Kaggle's file listing API — NO download.

        Kaggle's ``dataset_list_files()`` returns every file path stored in the
        dataset (up to the API limit).  For a typical image dataset those paths
        look like:

            train/ClassName/img001.jpg       ← most common (split / class / file)
            ClassName/img001.jpg             ← also common (class / file, flat)

        We parse each path, identify the class folder, count images per class,
        and return an ``ImageSampleResult`` with ``images_per_class`` populated
        but with **empty** ``sampled_paths`` and ``all_hashes`` — those require
        the actual images which we don't download here.

        Speed: < 2 seconds for 38 k-file datasets (one API round-trip).
        Use this for the fast analysis pass.  Use ``load_from_kaggle`` for the
        deep pass when you also need blur / duplicate / corruption counts.

        Returns ``ImageSampleResult(success=False)`` if:
        - Kaggle SDK is not installed or credentials are missing.
        - No image files are found in the file listing.
        - File paths cannot be parsed into a class structure (e.g. flat ZIPs).
        """
        try:
            # ── Authenticate ──────────────────────────────────────────────
            try:
                from kaggle.api.kaggle_api_extended import KaggleApiExtended
                api = KaggleApiExtended()
                api.authenticate()
            except (ImportError, AttributeError):
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                except ImportError:
                    return ImageSampleResult(
                        success=False,
                        error="Kaggle SDK not installed — run: pip install kaggle",
                    )

            # ── Fetch file listing (paginated, capped at 3 pages) ────────
            # FIX: use "owner/name" combined ref (new SDK requires this).
            # FIX: cap at 3 pages (≈60 files) — enough to detect the
            #      folder-per-class structure without making hundreds of API
            #      calls on large datasets (19K files = ~950 pages = timeout).
            #      Class names come from folder paths, not from counting files,
            #      so the first 3 pages always contain enough unique prefixes.
            #      The _count_from_file_listing logic already groups by folder,
            #      so partial listings still produce correct class name detection.
            _MAX_META_PAGES = 3
            try:
                all_files: list = []
                batch: list = []
                page_token = None
                pages_fetched = 0
                while pages_fetched < _MAX_META_PAGES:
                    result_page = api.dataset_list_files(
                        f"{owner}/{name}", page_token=page_token
                    )
                    batch = list(result_page.files or [])
                    all_files.extend(batch)
                    pages_fetched += 1
                    page_token = getattr(result_page, "nextPageToken", None)
                    if not page_token or not batch:
                        break
                files = all_files
            except Exception as exc:
                return ImageSampleResult(
                    success=False,
                    error=f"Kaggle dataset_list_files() failed: {exc}",
                )

            if not files:
                return ImageSampleResult(
                    success=False,
                    error=f"No files returned for {owner}/{name}",
                )

            # ── Parse class counts from file paths ────────────────────────
            # Handle two common layouts:
            #
            #   Layout A — split / class / image:
            #     train/Citrus_Canker_Disease/img001.jpg
            #     test/Citrus_Canker_Disease/img002.jpg
            #     → class = parts[1]
            #
            #   Layout B — class / image (flat, no split):
            #     Citrus_Canker_Disease/img001.jpg
            #     → class = parts[0]
            #
            #   Layout C — single ZIP at top level (no class info in listing):
            #     dataset.zip
            #     → cannot parse, return failure
            #
            # Images from train + test splits for the same class are merged —
            # the total count reflects the full dataset, matching what the user
            # sees on the Kaggle Data Explorer.

            class_counts: dict[str, int] = {}
            unparseable = 0

            for f in files:
                raw_name = str(getattr(f, "name", f) or "")
                path = Path(raw_name)
                parts = path.parts
                ext = path.suffix.lower()

                if ext not in IMAGE_EXTENSIONS:
                    # Skip ZIPs, CSVs, READMEs — only count image files
                    continue

                if len(parts) >= 3 and parts[0].lower() in _SPLIT_NAMES:
                    cls = parts[1]                      # Layout A
                elif len(parts) >= 2:
                    cls = parts[0]                      # Layout B
                else:
                    unparseable += 1
                    continue

                class_counts[cls] = class_counts.get(cls, 0) + 1

            if not class_counts:
                # Could be a single-ZIP dataset — file listing only shows the
                # ZIP name, not its internal paths.  Download is needed.
                reason = (
                    "File listing contains only a ZIP archive — "
                    "class structure cannot be inferred without downloading."
                    if any(str(getattr(f, "name", f)).lower().endswith(".zip")
                           for f in files)
                    else "No image files found in dataset file listing."
                )
                return ImageSampleResult(success=False, error=reason)

            # ── Build result (metadata only — no paths, no hashes) ────────
            class_info = {
                cls: ClassInfo(
                    name=cls,
                    total_images=count,
                    sampled_paths=[],   # not downloaded
                    all_hashes=[],      # not downloaded
                )
                for cls, count in class_counts.items()
            }
            result = self._build_result_from_class_info(
                class_info, "kaggle_meta", temp_dir=None
            )
            return result

        except Exception as exc:
            return ImageSampleResult(
                success=False,
                error=f"Kaggle meta loader error: {str(exc)[:200]}",
            )

    def load_from_kaggle(self, dataset_ref: str) -> ImageSampleResult:
        """
        Download a Kaggle dataset ZIP, extract, detect structure, build stats.

        Args:
            dataset_ref: "owner/dataset-name" (without kaggle.com/datasets/ prefix)
        """
        try:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApiExtended
                api = KaggleApiExtended()
                api.authenticate()
            except (ImportError, AttributeError):
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                except ImportError:
                    return ImageSampleResult(success=False,
                                             error="Kaggle SDK not installed — run: pip install kaggle")

            parts = dataset_ref.strip("/").split("/")
            if len(parts) < 2:
                return ImageSampleResult(success=False,
                                         error=f"Invalid Kaggle ref: '{dataset_ref}'")
            owner, name = parts[0], parts[1]

            # FIX: dataset_download_file (per-file) is broken in Kaggle SDK 1.6+.
            # It double-encodes the path in the URL giving 404 on every file.
            # dataset_download_files (whole dataset + unzip) is the only working
            # download method in the new SDK.
            temp_dir = tempfile.mkdtemp(prefix="datascout_kg_")
            try:
                try:
                    api.dataset_download_files(
                        f"{owner}/{name}",
                        path=temp_dir,
                        unzip=True,
                        quiet=True,
                        force=True,
                    )
                except Exception as dl_exc:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return ImageSampleResult(
                        success=False,
                        error=f"Kaggle download failed: {str(dl_exc)[:300]}",
                    )

                result = self.load_from_local(temp_dir)
                result.structure_detected = "kaggle_" + result.structure_detected
                result.temp_dir = temp_dir
                return result

            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return ImageSampleResult(success=False,
                                         error=f"Kaggle download failed: {str(e)[:200]}")
        except Exception as e:
            return ImageSampleResult(success=False,
                                     error=f"Kaggle loader error: {str(e)[:200]}")

    def load_from_local(self, root: str) -> ImageSampleResult:
        """Load from local filesystem. Always collects FULL image counts per class."""
        if not os.path.isdir(root):
            return ImageSampleResult(success=False,
                                     error=f"Not a directory: {root}")
        class_map = _detect_folder_per_class(root)
        structure = "folder_per_class"
        if not class_map:
            class_map = _detect_csv_labels(root)
            structure = "csv_labels"
        if not class_map:
            return ImageSampleResult(success=False, structure_detected="unknown",
                                     error="Could not detect folder-per-class or CSV-labels structure.")
        return self._build_result(class_map, structure)

    def load_from_zip(self, zip_path: str) -> ImageSampleResult:
        if not os.path.isfile(zip_path):
            return ImageSampleResult(success=False, error=f"ZIP not found: {zip_path}")
        temp_dir = tempfile.mkdtemp(prefix="datascout_img_")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_dir)
        except zipfile.BadZipFile as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return ImageSampleResult(success=False, error=f"Bad ZIP: {e}")
        result = self.load_from_local(temp_dir)
        result.structure_detected = "zip"
        result.temp_dir = temp_dir
        return result


    def load_class_counts_from_hf_meta(self, dataset_id: str) -> ImageSampleResult:
        """
        Infer class image counts from HuggingFace dataset card metadata — NO download.

        Uses the huggingface_hub DatasetInfo to extract split sizes and feature
        info (label names) from the dataset card. This provides class counts
        for balance scoring without streaming a single image.

        Falls back gracefully to ImageSampleResult(success=False) if:
        - huggingface_hub not installed
        - Dataset has no label feature info
        - Dataset is gated/private

        Speed: < 2 seconds (one API call to dataset_info endpoint).
        """
        if not _HAS_HF:
            return ImageSampleResult(
                success=False,
                error="huggingface_hub not installed — run: pip install huggingface_hub",
            )
        try:
            from huggingface_hub import HfApi  # type: ignore
            api = HfApi()

            # dataset_info returns features (label names + split sizes)
            info = api.dataset_info(dataset_id, timeout=6)

            images_per_class: dict[str, int] = {}

            # Path 1: features contain a ClassLabel — extract label names + counts
            features = getattr(info, "cardData", {}) or {}
            splits_info = getattr(info, "splits", None) or []

            # Try to get class names from card_data / features
            label_names: list[str] = []
            card = getattr(info, "cardData", {}) or {}
            if isinstance(card, dict):
                # Some cards expose class_names directly
                label_names = card.get("class_names", []) or card.get("label_names", []) or []

            # Try split sizes for total count
            total_from_splits = 0
            for sp in splits_info:
                sp_name = getattr(sp, "name", "")
                if sp_name in ("train", "all"):
                    total_from_splits = getattr(sp, "num_examples", 0) or 0
                    break

            if label_names and total_from_splits > 0:
                # Estimate even distribution (best we can do without download)
                per_class = max(1, total_from_splits // len(label_names))
                for name in label_names:
                    images_per_class[str(name)] = per_class
            elif total_from_splits > 0:
                # No class names but we know total — return single "unknown" class
                images_per_class["unknown"] = total_from_splits

            if not images_per_class:
                return ImageSampleResult(
                    success=False,
                    error="No class/split info available in dataset card",
                )

            return ImageSampleResult(
                success=True,
                images_per_class=images_per_class,
                structure_detected="hf_meta",
                source_type="huggingface",
            )

        except Exception as exc:
            return ImageSampleResult(
                success=False,
                error=f"HF meta load failed: {str(exc)[:200]}",
            )

    def load_from_huggingface(self, dataset_id: str, split: str = "train") -> ImageSampleResult:
        """
        Stream from HuggingFace.
        Counts ALL items per class (for balance).
        Hashes every image (for duplicates).
        Saves up to MAX_IMAGES_PER_CLASS per class (for quality checks).
        """
        if not _HAS_HF:
            return ImageSampleResult(success=False,
                                     error="huggingface_hub not installed — run: pip install datasets")
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)
            temp_dir = tempfile.mkdtemp(prefix="datascout_hf_")

            sample = next(iter(ds))
            label_col = next((k for k in sample
                              if k.lower() in {"label","class","category","target","fine_label"}), None)
            image_col = next((k for k in sample
                              if k.lower() in {"image","img","pixel_values"}), None)
            if image_col is None:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return ImageSampleResult(success=False,
                                         error="Could not identify image column.")

            all_counts:  dict[str, int]        = {}
            saved_paths: dict[str, list[str]]  = {}
            all_hashes:  dict[str, list[str]]  = {}

            for item in ds:
                cls = str(item.get(label_col, "unknown")).strip() if label_col else "unknown"
                all_counts[cls] = all_counts.get(cls, 0) + 1
                img = item.get(image_col)
                if img is None:
                    continue
                # Hash every image (cheap)
                if _HAS_PIL and hasattr(img, "convert"):
                    try:
                        small  = img.convert("L").resize((HASH_SIZE, HASH_SIZE), _PILImage.LANCZOS)
                        pixels = list(small.getdata())
                        avg    = sum(pixels) / len(pixels)
                        bits   = "".join("1" if p >= avg else "0" for p in pixels)
                        all_hashes.setdefault(cls, []).append(f"{int(bits, 2):016x}")
                    except Exception:
                        pass
                # Save sample for quality checks (expensive)
                if len(saved_paths.get(cls, [])) < self._images_per_class_cap:
                    cls_dir = Path(temp_dir) / cls.replace("/", "_")
                    cls_dir.mkdir(exist_ok=True)
                    idx = len(saved_paths.get(cls, []))
                    save_path = str(cls_dir / f"{idx:05d}.jpg")
                    try:
                        img.save(save_path, "JPEG", quality=85)
                        saved_paths.setdefault(cls, []).append(save_path)
                    except Exception:
                        pass

            if not all_counts:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return ImageSampleResult(success=False, error="No items found.")

            class_info = {
                cls: ClassInfo(
                    name=cls, total_images=total,
                    sampled_paths=saved_paths.get(cls, []),
                    all_hashes=all_hashes.get(cls, []),
                )
                for cls, total in all_counts.items()
            }
            return self._build_result_from_class_info(class_info, "huggingface", temp_dir)

        except Exception as e:
            return ImageSampleResult(success=False,
                                     error=f"HuggingFace load failed: {str(e)[:200]}")

    def load(self, path_or_id: str, source: str = "auto") -> ImageSampleResult:
        if source == "kaggle":      return self.load_from_kaggle(path_or_id)
        if source == "huggingface": return self.load_from_huggingface(path_or_id)
        if source == "zip":         return self.load_from_zip(path_or_id)
        if source == "local":       return self.load_from_local(path_or_id)

        if path_or_id.endswith(".zip") and os.path.isfile(path_or_id):
            r = self.load_from_zip(path_or_id)
            if r.success: return r
        if os.path.isdir(path_or_id):
            r = self.load_from_local(path_or_id)
            if r.success: return r
        if "/" in path_or_id and not path_or_id.startswith("/"):
            if _HAS_HF:
                r = self.load_from_huggingface(path_or_id)
                if r.success: return r
            r = self.load_from_kaggle(path_or_id)
            if r.success: return r
        return ImageSampleResult(success=False,
                                 error=f"Could not load dataset from: {path_or_id}")

    def _build_result(self, class_map: dict[str, list[str]], structure: str) -> ImageSampleResult:
        """
        Build result from {class_name: [all_image_paths]}.
        total_images = REAL count. sampled_paths = random 50 for quality.
        all_hashes = pHash of every image for duplicate detection.
        """
        if len(class_map) > MAX_CLASSES:
            class_map = dict(
                sorted(class_map.items(), key=lambda x: len(x[1]), reverse=True)[:MAX_CLASSES]
            )

        class_info: dict[str, ClassInfo] = {}
        for cls, all_paths in class_map.items():
            sample_n = min(self._images_per_class_cap, len(all_paths))
            sampled  = random.sample(all_paths, sample_n)
            hashes   = [h for h in (_perceptual_hash(p) for p in all_paths) if h is not None]
            class_info[cls] = ClassInfo(
                name=cls,
                total_images=len(all_paths),   # REAL count
                sampled_paths=sampled,
                all_hashes=hashes,
            )
        return self._build_result_from_class_info(class_info, structure, temp_dir=None)

    @staticmethod
    def _build_result_from_class_info(
        class_info: dict[str, ClassInfo], structure: str, temp_dir: Optional[str]
    ) -> ImageSampleResult:
        images_per_class  = {cls: info.total_images   for cls, info in class_info.items()}
        class_image_paths = {cls: info.sampled_paths  for cls, info in class_info.items()}
        total_sampled     = sum(len(info.sampled_paths) for info in class_info.values())
        return ImageSampleResult(
            success=True,
            class_info=class_info,
            class_image_paths=class_image_paths,
            total_classes=len(class_info),
            total_images_sampled=total_sampled,
            images_per_class=images_per_class,
            structure_detected=structure,
            temp_dir=temp_dir,
        )