"""
datascout.evaluation.filter_engine
─────────────────────────────────────────────────────
PRINCIPAL ENGINEER LEVEL: Hard-constraint filtering using the actual
SearchFilters contract from Phase 2 (contracts/requests.py).

SYSTEM DESIGN DECISIONS:

  1. WHY filter BEFORE scoring?
     - Scoring is O(n × dimensions). Pre-filtering 1M → ~50 candidates
       before scoring gives 20,000× speedup at scale.

  2. WHY FilterResult carries per-dataset rejection reasons?
     - "Why wasn't this dataset returned?" is the #1 debug question.
     - LLM explainer can surface rejection reasons to users.

  3. WHY all filter checks return (bool, str)?
     - First failure short-circuits — no wasted work.
     - str carries human-readable rejection reason for the trace.

Author:  Principal Engineer
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from datascout.contracts import RawDataset
from datascout.contracts.requests import SearchFilters
from datascout.contracts.states import LicenseType
from datascout.contracts.task_types import Modality, TaskType, compute_task_compatibility

logger = logging.getLogger("datascout.evaluation.filter_engine")

NON_COMMERCIAL_LICENSES: frozenset[LicenseType] = frozenset({
    LicenseType.CC_BY_NC,
    LicenseType.CC_BY_NC_SA,
})


@dataclass
class FilterResult:
    passed:   list[RawDataset]
    rejected: list[tuple[RawDataset, str]]   # (dataset, rejection_reason)
    total_input: int
    filters_applied: list[str]

    @property
    def pass_count(self) -> int:  return len(self.passed)

    @property
    def reject_count(self) -> int: return len(self.rejected)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.total_input if self.total_input else 0.0

    def to_dict(self) -> dict:
        return {
            "total_input":      self.total_input,
            "passed":           self.pass_count,
            "rejected":         self.reject_count,
            "pass_rate":        round(self.pass_rate, 3),
            "filters_applied":  self.filters_applied,
            "rejection_reasons": [
                {"canonical_id": ds.canonical_id, "reason": r}
                for ds, r in self.rejected
            ],
        }


class FilterEngine:
    """
    Applies hard-constraint filters from SearchFilters contract.
    Fields used from SearchFilters (contracts/requests.py):
      min_rows, max_rows, min_completeness, require_description,
      license_types, task_types, modalities, domains, updated_after.
    Additional runtime args: exclude_duplicates, sources, strict_task_matching.
    """

    def __init__(self, strict_task_matching: bool = False) -> None:
        self.strict_task_matching = strict_task_matching

    def apply(
        self,
        datasets: list[RawDataset],
        filters: SearchFilters,
        query_task:      Optional[TaskType] = None,
        query_modality:  Optional[Modality] = None,
        exclude_duplicates: bool = True,
        sources: Optional[list[str]] = None,
        require_commercial_license: bool = False,
        excluded_licenses: Optional[list[LicenseType]] = None,
        required_modalities: Optional[list[Modality]] = None,
        min_quality_score: Optional[float] = None,
    ) -> FilterResult:
        """Filter datasets against all active hard constraints. Never raises."""
        active = self._active_names(
            filters, query_task, exclude_duplicates, sources,
            require_commercial_license, excluded_licenses,
            required_modalities, min_quality_score,
        )

        if not datasets:
            return FilterResult(passed=[], rejected=[], total_input=0,
                                filters_applied=active)

        passed:   list[RawDataset] = []
        rejected: list[tuple[RawDataset, str]] = []

        for ds in datasets:
            ok, reason = self._check(
                ds, filters, query_task,
                exclude_duplicates, sources,
                require_commercial_license, excluded_licenses,
                required_modalities, min_quality_score,
            )
            if ok:
                passed.append(ds)
            else:
                rejected.append((ds, reason))

        logger.info("filter_complete", extra={
            "input": len(datasets), "passed": len(passed),
            "rejected": len(rejected), "filters": active,
        })
        return FilterResult(passed=passed, rejected=rejected,
                            total_input=len(datasets), filters_applied=active)

    def _check(
        self,
        ds: RawDataset,
        f: SearchFilters,
        query_task: Optional[TaskType],
        exclude_duplicates: bool,
        sources: Optional[list[str]],
        require_commercial_license: bool,
        excluded_licenses: Optional[list[LicenseType]],
        required_modalities: Optional[list[Modality]],
        min_quality_score: Optional[float],
    ) -> tuple[bool, str]:

        # 1. Duplicate
        if exclude_duplicates and ds.is_duplicate:
            return False, "Duplicate record excluded"

        # 2. Source allowlist
        if sources and ds.source not in set(sources):
            return False, f"Source '{ds.source}' not in allowed list"

        # 3. Description required
        if f.require_description and not ds.has_description:
            return False, "No description available"

        # 4. Min rows
        if f.min_rows and ds.row_count is not None and ds.row_count < f.min_rows:
            return False, f"Only {ds.row_count} rows (min {f.min_rows})"

        # 5. Max rows
        if f.max_rows and ds.row_count is not None and ds.row_count > f.max_rows:
            return False, f"{ds.row_count} rows exceeds max {f.max_rows}"

        # 6. Min completeness (0.0–1.0 scale from contract)
        if f.min_completeness and ds.metadata_completeness < f.min_completeness:
            return False, (
                f"Completeness {ds.metadata_completeness:.2f} < "
                f"required {f.min_completeness:.2f}"
            )

        # 7. Min quality score (0–100 scale, caller converts)
        if min_quality_score and (ds.metadata_completeness * 100) < min_quality_score:
            return False, (
                f"Quality {ds.metadata_completeness*100:.1f} < "
                f"required {min_quality_score}"
            )

        # 8. Commercial license
        if require_commercial_license and ds.license_type in NON_COMMERCIAL_LICENSES:
            return False, f"Non-commercial license: {ds.license_type.value}"

        # 9. Excluded licenses (extra list passed by caller)
        if excluded_licenses and ds.license_type in set(excluded_licenses):
            return False, f"License excluded: {ds.license_type.value}"

        # 10. License allowlist from contract
        if f.license_types and ds.license_type not in set(f.license_types):
            return False, f"License {ds.license_type} not in allowed list"

        # 11. Required modalities (extra list)
        if required_modalities:
            ds_mods = set(ds.modalities or [])
            req = set(required_modalities)
            if not ds_mods & req:
                return False, (
                    f"Modality mismatch: need {[m.value for m in req]}, "
                    f"have {[m.value for m in ds_mods]}"
                )

        # 12. Modality allowlist from contract
        if f.modalities:
            ds_mods = set(ds.modalities or [])
            allowed = set(f.modalities)
            if not ds_mods & allowed:
                return False, f"Modality not in allowed list"

        # 13. Updated after
        if f.updated_after and ds.last_updated:
            cutoff = f.updated_after
            last = ds.last_updated
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            if last < cutoff:
                return False, f"Dataset not updated since {cutoff.date()}"

        # 14. Task compatibility (strict mode)
        if (
            self.strict_task_matching
            and query_task
            and query_task != TaskType.OTHER
            and ds.modalities
        ):
            compat = compute_task_compatibility(list(ds.modalities), query_task)
            if not compat.is_compatible:
                return False, (
                    f"Task incompatible: {query_task.value} vs "
                    f"{[m.value for m in ds.modalities]}"
                )

        return True, ""

    @staticmethod
    def _active_names(
        f: SearchFilters,
        query_task: Optional[TaskType],
        exclude_duplicates: bool,
        sources: Optional[list[str]],
        require_commercial_license: bool,
        excluded_licenses: Optional[list],
        required_modalities: Optional[list],
        min_quality_score: Optional[float],
    ) -> list[str]:
        active = []
        if exclude_duplicates:             active.append("exclude_duplicates")
        if sources:                        active.append("sources")
        if f.require_description:          active.append("require_description")
        if f.min_rows:                     active.append("min_rows")
        if f.max_rows:                     active.append("max_rows")
        if f.min_completeness:             active.append("min_completeness")
        if min_quality_score:              active.append("min_quality_score")
        if require_commercial_license:     active.append("commercial_license")
        if excluded_licenses:              active.append("excluded_licenses")
        if f.license_types:               active.append("license_types")
        if required_modalities:            active.append("required_modalities")
        if f.modalities:                   active.append("modalities")
        if f.updated_after:                active.append("updated_after")
        if query_task not in (None, TaskType.OTHER):
            active.append("task_compatibility")
        return active