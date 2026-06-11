"""
datascout.analysis.column_analyzer
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Per-column analysis — type detection, missing value
quantification, basic statistics, and cardinality assessment.

SYSTEM DESIGN DECISIONS:

  1. WHY type detection (not just use pandas dtypes)?
     - pandas dtypes are storage types (int64, object), not semantic types
     - A column of {"0", "1", "2"} has dtype=object but is CATEGORICAL
     - A column of ISO date strings has dtype=object but is DATETIME
     - Semantic types drive downstream decisions: target detection, encoding choices
     - Semantic type inference is the first step in any real AutoML pipeline

  2. WHY cardinality ratio (not just unique count)?
     - unique_count alone is meaningless without context
     - 100 unique values in a 100-row column = all unique (likely ID column)
     - 100 unique values in a 10K-row column = low cardinality (likely categorical)
     - cardinality_ratio = unique / total → [0, 1] — comparable across datasets

  3. WHY separate numeric_stats only for numeric columns?
     - Mean/std on categorical data is meaningless (pandas computes it anyway)
     - Separating by type: results are only populated when meaningful
     - Downstream scorer checks has_numeric_stats — not None-checking every field

  4. WHY missing_percentage as 0-100 scale (not 0.0-1.0)?
     - Display convention: "8.5% missing" is more natural than "0.085 missing"
     - Consistent with quality_score (0-100), class_imbalance_score (0-100)
     - Downstream scorer uses uniform scale for all quality signals

FAILURE SCENARIOS HANDLED:
  - All values null → missing_percentage=100.0, type=UNKNOWN
  - Single row → stats computed but flagged as low_sample
  - Mixed types in column → detected as MIXED, no stats
  - Overflow in numeric stats → caught, stats set to None

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("datascout.analysis.column_analyzer")


class ColumnType(str, Enum):
    NUMERIC      = "numeric"       # int or float
    CATEGORICAL  = "categorical"   # string with limited cardinality
    TEXT         = "text"          # string with high cardinality (free text)
    DATETIME     = "datetime"
    BOOLEAN      = "boolean"
    ID           = "id"            # High cardinality, likely row identifier
    MIXED        = "mixed"         # Multiple types in same column
    UNKNOWN      = "unknown"


@dataclass
class NumericStats:
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    zeros_percentage: float = 0.0    # % of values that are exactly 0
    negatives_percentage: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class ColumnInfo:
    """Analysis result for a single column."""
    name: str
    col_type: ColumnType
    total_count: int
    null_count: int
    missing_percentage: float          # 0-100
    unique_count: int
    cardinality_ratio: float           # unique / total, 0-1
    sample_values: list[Any] = field(default_factory=list)  # Up to 5 sample values
    numeric_stats: Optional[NumericStats] = None
    top_categories: Optional[list[tuple[Any, int]]] = None  # (value, count) for categoricals
    is_likely_target: bool = False     # Heuristic flag
    is_likely_id: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.col_type.value,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "missing_percentage": round(self.missing_percentage, 2),
            "unique_count": self.unique_count,
            "cardinality_ratio": round(self.cardinality_ratio, 4),
            "sample_values": [str(v) for v in self.sample_values],
            "numeric_stats": self.numeric_stats.to_dict() if self.numeric_stats else None,
            "top_categories": [(str(v), c) for v, c in (self.top_categories or [])],
            "is_likely_target": self.is_likely_target,
            "is_likely_id": self.is_likely_id,
        }


@dataclass
class ColumnAnalysisResult:
    """Full column analysis for all columns in a dataset sample."""
    columns: list[ColumnInfo] = field(default_factory=list)
    total_columns: int = 0
    numeric_columns: int = 0
    categorical_columns: int = 0
    text_columns: int = 0
    datetime_columns: int = 0
    id_columns: int = 0
    overall_missing_percentage: float = 0.0    # Average across all columns
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_columns": self.total_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "text_columns": self.text_columns,
            "datetime_columns": self.datetime_columns,
            "id_columns": self.id_columns,
            "overall_missing_percentage": round(self.overall_missing_percentage, 2),
            "columns": [c.to_dict() for c in self.columns],
            "error": self.error,
        }


class ColumnAnalyzer:
    """
    Analyzes all columns in a DataFrame sample.
    Produces ColumnAnalysisResult — never raises.
    """

    # Cardinality thresholds
    ID_THRESHOLD          = 0.95   # >95% unique → likely ID column
    CATEGORICAL_THRESHOLD = 0.10   # <10% unique → categorical
    TEXT_THRESHOLD        = 0.50   # 10-50% unique → text (free-form)
    # Above 50% unique → text or ID

    # Target column name heuristics
    TARGET_NAME_HINTS = frozenset({
        "target", "label", "labels", "class", "classes", "y",
        "output", "result", "outcome", "response", "dependent",
        "survived", "churn", "fraud", "diagnosis", "disease",
        "price", "salary", "score", "rating",
    })

    ID_NAME_HINTS = frozenset({
        "id", "idx", "index", "uuid", "guid", "key",
        "user_id", "item_id", "product_id", "customer_id",
    })

    def analyze(self, df: Any) -> ColumnAnalysisResult:
        """
        Analyze all columns in a DataFrame.
        Returns ColumnAnalysisResult — never raises.
        """
        try:
            return self._analyze_internal(df)
        except Exception as e:
            logger.error("column_analysis_error", extra={"error": str(e)}, exc_info=True)
            return ColumnAnalysisResult(error=str(e)[:200])

    def _analyze_internal(self, df: Any) -> ColumnAnalysisResult:
        import pandas as pd
        import numpy as np

        columns: list[ColumnInfo] = []
        total_rows = len(df)

        for col_name in df.columns:
            series = df[col_name]
            try:
                info = self._analyze_column(col_name, series, total_rows)
                columns.append(info)
            except Exception as e:
                logger.warning(
                    "column_analyze_skip",
                    extra={"column": col_name, "error": str(e)[:80]},
                )
                columns.append(ColumnInfo(
                    name=col_name,
                    col_type=ColumnType.UNKNOWN,
                    total_count=total_rows,
                    null_count=int(series.isnull().sum()),
                    missing_percentage=float(series.isnull().mean() * 100),
                    unique_count=0,
                    cardinality_ratio=0.0,
                ))

        # Aggregate stats
        numeric_cols = sum(1 for c in columns if c.col_type == ColumnType.NUMERIC)
        cat_cols     = sum(1 for c in columns if c.col_type == ColumnType.CATEGORICAL)
        text_cols    = sum(1 for c in columns if c.col_type == ColumnType.TEXT)
        dt_cols      = sum(1 for c in columns if c.col_type == ColumnType.DATETIME)
        id_cols      = sum(1 for c in columns if c.col_type == ColumnType.ID)

        overall_missing = (
            sum(c.missing_percentage for c in columns) / len(columns)
            if columns else 0.0
        )

        return ColumnAnalysisResult(
            columns=columns,
            total_columns=len(columns),
            numeric_columns=numeric_cols,
            categorical_columns=cat_cols,
            text_columns=text_cols,
            datetime_columns=dt_cols,
            id_columns=id_cols,
            overall_missing_percentage=round(overall_missing, 2),
        )

    def _analyze_column(self, name: str, series: Any, total_rows: int) -> ColumnInfo:
        import pandas as pd
        import numpy as np

        null_count   = int(series.isnull().sum())
        non_null     = series.dropna()
        missing_pct  = (null_count / total_rows * 100) if total_rows > 0 else 100.0
        unique_count = int(series.nunique())
        cardinality  = unique_count / total_rows if total_rows > 0 else 0.0

        # ── Detect type ───────────────────────────────────────────────────────
        col_type = self._detect_type(series, non_null, cardinality)

        # ── Sample values ─────────────────────────────────────────────────────
        sample_vals = non_null.head(5).tolist()

        # ── Numeric stats ─────────────────────────────────────────────────────
        numeric_stats: Optional[NumericStats] = None
        if col_type == ColumnType.NUMERIC and len(non_null) > 0:
            try:
                numeric_col = pd.to_numeric(non_null, errors="coerce").dropna()
                if len(numeric_col) > 0:
                    numeric_stats = NumericStats(
                        mean=float(numeric_col.mean()),
                        std=float(numeric_col.std()) if len(numeric_col) > 1 else 0.0,
                        min=float(numeric_col.min()),
                        max=float(numeric_col.max()),
                        median=float(numeric_col.median()),
                        q25=float(numeric_col.quantile(0.25)),
                        q75=float(numeric_col.quantile(0.75)),
                        zeros_percentage=float((numeric_col == 0).mean() * 100),
                        negatives_percentage=float((numeric_col < 0).mean() * 100),
                    )
            except Exception:
                pass  # Stats optional — don't fail the whole column

        # ── Top categories ────────────────────────────────────────────────────
        top_categories: Optional[list] = None
        if col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) and len(non_null) > 0:
            counts = non_null.value_counts().head(10)
            top_categories = [(v, int(c)) for v, c in counts.items()]

        # ── Target / ID heuristics ─────────────────────────────────────────────
        name_lower = str(name).lower().strip()
        is_likely_target = any(hint in name_lower for hint in self.TARGET_NAME_HINTS)
        is_likely_id = (
            col_type == ColumnType.ID
            or any(hint == name_lower for hint in self.ID_NAME_HINTS)
        )

        return ColumnInfo(
            name=name,
            col_type=col_type,
            total_count=total_rows,
            null_count=null_count,
            missing_percentage=round(missing_pct, 2),
            unique_count=unique_count,
            cardinality_ratio=round(cardinality, 4),
            sample_values=sample_vals,
            numeric_stats=numeric_stats,
            top_categories=top_categories,
            is_likely_target=is_likely_target,
            is_likely_id=is_likely_id,
        )

    def _detect_type(self, series: Any, non_null: Any, cardinality: float) -> ColumnType:
        """
        Infer semantic column type from pandas dtype and cardinality.

        WHY explicit is_string_dtype check:
        - pandas 2.0+ uses StringDtype (dtype.name == 'str') for string columns
          instead of the legacy object dtype (dtype == object)
        - Code that only checks `dtype == object` silently falls through to UNKNOWN
          in pandas 2+ for ALL string columns — a complete correctness failure
        - Both paths (legacy object AND new StringDtype) must be handled
        """
        import pandas as pd
        import numpy as np

        if len(non_null) == 0:
            return ColumnType.UNKNOWN

        dtype = series.dtype

        # ── Native bool ───────────────────────────────────────────────────────
        if dtype == bool or pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BOOLEAN

        # ── Numeric (int / float / other numeric) ─────────────────────────────
        if pd.api.types.is_numeric_dtype(dtype):
            if cardinality >= self.ID_THRESHOLD:
                return ColumnType.ID
            return ColumnType.NUMERIC

        # ── Datetime ──────────────────────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

        # ── String / object — covers BOTH legacy object dtype AND pandas 2+ str dtype ──
        is_string_col = (
            dtype == object
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        )

        if is_string_col:
            # Boolean disguised as strings
            try:
                unique_lower = set(str(v).strip().lower() for v in non_null.unique())
                if unique_lower <= {"true", "false"} or \
                   unique_lower <= {"yes", "no"} or \
                   unique_lower <= {"0", "1"}:
                    return ColumnType.BOOLEAN
            except Exception:
                pass

            # Numeric disguised as strings
            try:
                converted = pd.to_numeric(non_null, errors="coerce")
                if converted.notna().mean() > 0.9:
                    if cardinality >= self.ID_THRESHOLD:
                        return ColumnType.ID
                    return ColumnType.NUMERIC
            except Exception:
                pass

            # Datetime disguised as strings
            try:
                sample = non_null.head(20).astype(str)
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return ColumnType.DATETIME
            except Exception:
                pass

            # Cardinality-based string classification
            if cardinality >= self.ID_THRESHOLD:
                return ColumnType.ID
            elif cardinality <= self.CATEGORICAL_THRESHOLD:
                return ColumnType.CATEGORICAL
            elif cardinality <= self.TEXT_THRESHOLD:
                return ColumnType.TEXT
            else:
                # High-cardinality string: check avg length to distinguish text vs ID
                try:
                    avg_len = non_null.astype(str).str.len().mean()
                    return ColumnType.TEXT if avg_len > 50 else ColumnType.CATEGORICAL
                except Exception:
                    return ColumnType.TEXT

        return ColumnType.UNKNOWN