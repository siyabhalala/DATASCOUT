"""
datascout.infrastructure.metrics
──────────────────────────────────
Compatibility shim for agent metric recording.

The primary monitoring system is in infrastructure.monitoring (Prometheus).
This module provides a lightweight record_metric() interface used by
scout_agent and react_loop for simple numeric event recording.
"""

from __future__ import annotations

import logging
from typing import Union

logger = logging.getLogger("datascout.infrastructure.metrics")

# In-memory counters for demo/hackathon mode (no Prometheus required)
_metric_store: dict[str, list[float]] = {}


def record_metric(name: str, value: Union[int, float]) -> None:
    """
    Record a named numeric metric.

    In production: delegates to the Prometheus metrics system.
    In demo/dev: stores in-memory for inspection via get_metrics().

    Never raises — metric recording must never crash the pipeline.
    """
    try:
        # Attempt to use the full Prometheus metrics if available
        from infrastructure.monitoring import metrics as prom_metrics
        # Map known metric names to Prometheus calls
        if "candidates" in name:
            prom_metrics.adapter_success("agent", int(value), 0.0)
        # For unknown metrics, fall through to in-memory storage
    except Exception:
        pass

    # Always record in-memory for diagnostics
    if name not in _metric_store:
        _metric_store[name] = []
    _metric_store[name].append(float(value))

    logger.debug("metric_recorded", extra={"name": name, "value": value})


def get_metrics() -> dict[str, dict]:
    """
    Return recorded metrics summary for diagnostic inspection.

    Returns:
        Dict of metric_name → {count, total, mean, last}
    """
    result = {}
    for name, values in _metric_store.items():
        if values:
            result[name] = {
                "count": len(values),
                "total": sum(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
            }
    return result


def reset_metrics() -> None:
    """Clear all recorded metrics. Used in tests."""
    _metric_store.clear()