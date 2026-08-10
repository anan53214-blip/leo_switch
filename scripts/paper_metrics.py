"""论文图统一指标定义与统计工具。

固定用户对比和多用户扩展图必须从本模块读取指标，避免两套脚本使用
不同字段、单位或优劣方向。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


PRIMARY_COMPARE_METRICS = (
    ("avg_success_delay", "Successful-Task Delay"),
    ("p95_success_delay", "P95 Successful-Task Delay"),
    ("task_success_rate", "Task Success"),
    ("deadline_violation_rate", "Deadline Violation"),
    ("successful_task_throughput", "Successful Task Throughput"),
    ("energy_per_successful_task", "Energy per Successful Task"),
)

FIXED_CORE_METRICS = (
    ("avg_success_delay", "Successful-Task Delay", "Average Delay (ms)"),
    ("p95_success_delay", "P95 Successful-Task Delay", "P95 Delay (ms)"),
    ("task_success_rate", "Task Success Rate", "Task Success Rate (%)"),
    ("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)"),
    ("successful_task_throughput", "Successful Task Throughput", "Tasks / User-Minute"),
    ("energy_per_successful_task", "Energy per Successful Task", "Energy / Successful Task"),
)

ADDITIONAL_METRICS = (
    ("handover_failure_rate", "Handover Failure Rate"),
    ("handovers_per_user_minute", "Handovers per User-Minute"),
    ("blocked_time_ratio", "Blocked User-Time Ratio"),
    ("jain_mec_load_fairness", "MEC Load Jain Fairness"),
)

CORE_SCALING_METRICS = (
    ("avg_success_delay", "Successful-Task Delay", "Average Delay (ms)", 1000.0),
    ("p95_success_delay", "P95 Successful-Task Delay", "P95 Delay (ms)", 1000.0),
    ("task_success_rate", "Task Success Rate", "Task Success Rate (%)", 100.0),
    ("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)", 100.0),
    ("successful_task_throughput", "Successful Task Throughput", "Tasks / User-Minute", 1.0),
    ("handover_failure_rate", "Handover Failure Rate", "Handover Failure Rate (%)", 100.0),
)

RESOURCE_SCALING_METRICS = (
    ("energy_per_successful_task", "Energy per Successful Task", "Energy / Successful Task", 1.0),
    ("jain_mec_load_fairness", "MEC Load Jain Fairness", "Jain Fairness Index", 1.0),
    ("blocked_time_ratio", "Blocked User-Time Ratio", "Blocked User-Time Ratio (%)", 100.0),
    ("handovers_per_user_minute", "Handover Frequency", "Handovers / User-Minute", 1.0),
)

COMBINED_SCALING_METRICS = (
    ("avg_success_delay", "Successful-Task Delay", "Average Delay (ms)", 1000.0),
    ("task_success_rate", "Task Success Rate", "Task Success Rate (%)", 100.0),
    ("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)", 100.0),
    ("successful_task_throughput", "Successful Task Throughput", "Tasks / User-Minute", 1.0),
    ("energy_per_successful_task", "Energy per Successful Task", "Energy / Successful Task", 1.0),
    ("jain_mec_load_fairness", "MEC Load Jain Fairness", "Jain Fairness Index", 1.0),
)

PAPER_METRIC_KEYS = tuple(
    dict.fromkeys(
        spec[0]
        for metric_specs in (
            PRIMARY_COMPARE_METRICS,
            FIXED_CORE_METRICS,
            ADDITIONAL_METRICS,
            CORE_SCALING_METRICS,
            RESOURCE_SCALING_METRICS,
            COMBINED_SCALING_METRICS,
        )
        for spec in metric_specs
    )
)

HIGHER_IS_BETTER = {
    "avg_success_delay": False,
    "p95_success_delay": False,
    "task_success_rate": True,
    "deadline_violation_rate": False,
    "successful_task_throughput": True,
    "handover_failure_rate": False,
    "blocked_time_ratio": False,
    "handovers_per_user_minute": False,
    "energy_per_successful_task": False,
    "jain_mec_load_fairness": True,
}

SUCCESS_DEPENDENT_METRICS = frozenset(
    {
        "avg_success_delay",
        "p95_success_delay",
        "energy_per_successful_task",
    }
)


def _float(record: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = record.get(key, default)
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return float(default)


def derive_paper_metrics(record: Mapping[str, object]) -> dict[str, object]:
    """补齐论文图派生指标，同时保留调用方已有字段。"""

    derived: dict[str, object] = dict(record)
    total_user_seconds = _float(record, "total_user_seconds")
    blocked_user_seconds = _float(record, "blocked_user_seconds")
    completed_tasks = _float(record, "completed_tasks")
    total_energy = _float(record, "total_energy")
    committed_handovers = _float(
        record,
        "handover_committed",
        _float(record, "total_handovers"),
    )

    derived.setdefault(
        "blocked_time_ratio",
        blocked_user_seconds / total_user_seconds if total_user_seconds > 0.0 else 0.0,
    )
    derived.setdefault(
        "handovers_per_user_minute",
        60.0 * committed_handovers / total_user_seconds
        if total_user_seconds > 0.0
        else 0.0,
    )
    if record.get("successful_task_throughput") in (None, ""):
        derived["successful_task_throughput"] = (
            60.0 * completed_tasks / total_user_seconds
            if total_user_seconds > 0.0
            else 0.0
        )
    derived.setdefault(
        "energy_per_successful_task",
        total_energy / completed_tasks if completed_tasks > 0.0 else 0.0,
    )

    delay_samples = np.asarray(
        list(record.get("successful_task_delay_samples", []) or []),
        dtype=np.float64,
    )
    delay_samples = delay_samples[np.isfinite(delay_samples)]
    if delay_samples.size:
        derived.setdefault("avg_success_delay", float(np.mean(delay_samples)))
        derived.setdefault("p95_success_delay", float(np.percentile(delay_samples, 95)))
    else:
        legacy_delay = _float(record, "avg_delay")
        derived.setdefault("avg_success_delay", legacy_delay)
        derived.setdefault(
            "p95_success_delay",
            _float(record, "avg_success_delay", legacy_delay),
        )

    derived.setdefault(
        "jain_mec_load_fairness",
        _float(
            record,
            "mec_load_fairness",
            _float(record, "load_balance_coefficient"),
        ),
    )
    return derived


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 20260728,
) -> tuple[float, float, float]:
    """返回均值和确定性 bootstrap 置信区间。"""

    samples = np.asarray(values, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0, 0.0, 0.0

    mean = float(np.mean(samples))
    if samples.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.size, size=(resamples, samples.size))
    bootstrap_means = np.mean(samples[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def metric_scale(metric_key: str) -> float:
    if metric_key in {"avg_success_delay", "p95_success_delay"}:
        return 1000.0
    if metric_key.endswith("_rate") or metric_key == "blocked_time_ratio":
        return 100.0
    return 1.0
