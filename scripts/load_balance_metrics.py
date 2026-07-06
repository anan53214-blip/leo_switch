"""Paper-style load-balance evaluation metrics.

The metric follows the CDF/score design used by Fu Yi-Yang et al.: first record
system load variance over time, then convert the mean variance to a balance
coefficient with (1 - 4B) / (1 + 4B).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np


def _finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def load_variance(
    loads: Iterable[float],
    active_mask: Optional[Iterable[bool]] = None,
) -> float:
    """Return variance over active normalized satellite loads."""

    load_array = np.asarray(list(loads), dtype=np.float64).reshape(-1)
    if active_mask is not None:
        mask = np.asarray(list(active_mask), dtype=bool).reshape(-1)
        if mask.shape != load_array.shape:
            raise ValueError("active_mask must have the same length as loads")
        load_array = load_array[mask]
    load_array = load_array[np.isfinite(load_array)]
    if load_array.size == 0:
        return 0.0
    load_array = np.clip(load_array, 0.0, 1.0)
    mean_load = float(np.mean(load_array))
    return float(np.mean((load_array - mean_load) ** 2))


def load_balance_coefficient(load_variance_value: float) -> float:
    """Map load variance B in [0, 0.25] to a higher-is-better coefficient."""

    variance = float(load_variance_value)
    if not np.isfinite(variance):
        return 0.0
    variance = float(np.clip(variance, 0.0, 0.25))
    return float((1.0 - 4.0 * variance) / (1.0 + 4.0 * variance))


def empirical_cdf(values: Iterable[float]) -> list[dict[str, float]]:
    """Return sorted empirical CDF points for variance samples."""

    samples = np.sort(_finite_array(values))
    count = int(samples.size)
    if count == 0:
        return []
    return [
        {"x": float(value), "cdf": float((index + 1) / count)}
        for index, value in enumerate(samples)
    ]


def summarize_load_variance_samples(samples: Iterable[float]) -> dict[str, object]:
    """Summarize time-point load variance samples for tables and CDF plots."""

    finite_samples = _finite_array(samples)
    if finite_samples.size == 0:
        mean_variance = 0.0
        coefficient = 0.0
    else:
        finite_samples = np.clip(finite_samples, 0.0, 0.25)
        mean_variance = float(np.mean(finite_samples))
        coefficient = load_balance_coefficient(mean_variance)
    return {
        "load_balance_variance": mean_variance,
        "load_balance_coefficient": coefficient,
        "load_variance_sample_count": int(finite_samples.size),
        "load_variance_cdf": empirical_cdf(finite_samples),
    }


def normalize_load_balance_metrics(record: Mapping[str, object]) -> dict[str, object]:
    """Normalize legacy load-balance aliases to the paper-style coefficient."""

    normalized: dict[str, object] = dict(record)
    if "load_variance_samples" in normalized:
        sample_values = normalized.get("load_variance_samples", [])
        summary = summarize_load_variance_samples(
            [] if sample_values is None else sample_values
        )
        normalized.update(summary)
        coefficient = float(summary["load_balance_coefficient"])
    elif "load_balance_variance" in normalized:
        variance = float(normalized.get("load_balance_variance") or 0.0)
        coefficient = load_balance_coefficient(variance)
        normalized["load_balance_variance"] = float(np.clip(variance, 0.0, 0.25))
        normalized["load_balance_coefficient"] = coefficient
        normalized.setdefault("load_variance_sample_count", 1)
    else:
        coefficient = float(
            normalized.get(
                "load_balance_coefficient",
                normalized.get(
                    "mec_load_fairness",
                    normalized.get(
                        "active_load_balance_score",
                        normalized.get("avg_load_balance_score", 0.0),
                    ),
                ),
            )
            or 0.0
        )
        normalized["load_balance_coefficient"] = coefficient

    normalized["mec_load_fairness"] = coefficient
    normalized["active_load_balance_score"] = coefficient
    normalized["avg_load_balance_score"] = coefficient
    return normalized
