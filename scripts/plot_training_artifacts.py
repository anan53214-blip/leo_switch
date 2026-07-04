#!/usr/bin/env python3
"""Generate comparison figures from existing training/comparison artifacts.

This script is intentionally plot-only. It reads training_history.json files
or an existing comparison_summary.json, writes normalized comparison summaries,
and regenerates the paper-style comparison figures without launching training
or checkpoint evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HistorySpec = Tuple[Optional[str], Path]
DEFAULT_SELECTION_METRIC = "avg_delay"
PRIMARY_COMPARE_METRICS = (
    ("avg_delay", "Average Delay"),
    ("service_continuity_rate", "Service Continuity"),
    ("task_completion_rate", "Task Completion"),
    ("mec_load_fairness", "MEC Load Fairness"),
)
HIGHER_IS_BETTER = {
    "reward": True,
    "mean_reward": True,
    "avg_delay": False,
    "total_energy": False,
    "service_continuity_rate": True,
    "service_availability_rate": True,
    "handover_failure_rate": False,
    "mec_load_fairness": True,
    "avg_load_balance_score": True,
    "active_load_balance_score": True,
    "energy_per_successful_task": False,
    "task_completion_rate": True,
    "task_success_rate": True,
    "task_failure_rate": False,
    "deadline_violation_rate": False,
    "task_settlement_rate": True,
}
DISPLAY_NAME_MAP = {
    "han_mappo": "HAN+MAPPO",
    "han_attn": "HAN+Attn",
    "attn_mappo": "Attn+MAPPO",
    "mappo_no_han": "MAPPO",
    "maddpg": "MADDPG",
    "pdqn": "PDQN",
    "han_maddpg": "HAN+MADDPG",
    "han_pdqn": "HAN+PDQN",
    "random": "Random",
    "min_distance": "Min-Distance",
    "full_local": "Full-Local",
    "joint_greedy": "Joint Greedy",
}
METHOD_COLORS = [
    "#2F6C9E",
    "#C44E52",
    "#55A868",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
    "#E17C05",
]
CSV_FIELDS = [
    "method",
    "display_name",
    "is_system",
    "episodes",
    "mean_reward",
    "std_reward",
    "avg_delay",
    "total_energy",
    "handover_success_rate",
    "handover_failure_rate",
    "forced_termination_rate",
    "service_continuity_rate",
    "service_availability_rate",
    "task_completion_rate",
    "task_success_rate",
    "task_failure_rate",
    "task_settlement_rate",
    "task_resolution_rate",
    "pending_task_rate",
    "deadline_violation_rate",
    "handover_frequency",
    "mec_load_fairness",
    "active_load_balance_score",
    "avg_load_balance_score",
    "energy_per_successful_task",
    "selection_metric",
    "selection_score",
    "primary_metric_win_count",
    "primary_metric_wins_text",
    "training_history",
    "source",
]


def parse_history_spec(value: str) -> HistorySpec:
    """Parse either PATH or LABEL=PATH."""
    if "=" not in value:
        return None, Path(value)
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"History label is empty in: {value}")
    return label, Path(path.strip())


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def _float(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = record.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _record_reward(record: Dict[str, Any]) -> float:
    return _float(
        record,
        "eval_mean_reward",
        _float(record, "mean_reward", _float(record, "recent_mean_reward", _float(record, "reward", 0.0))),
    )


def pretty_method_name(name: str, is_system: bool) -> str:
    normalized = name.split("(", 1)[0].strip().lower().replace("-", "_")
    for key in sorted(DISPLAY_NAME_MAP, key=len, reverse=True):
        if normalized == key or normalized.startswith(f"{key}_"):
            return DISPLAY_NAME_MAP[key]
    return "HAN+MAPPO" if is_system else name


def compute_deadline_violation_rate(record: Dict[str, Any]) -> float:
    if "deadline_violation_rate" in record:
        return _float(record, "deadline_violation_rate")
    total_tasks = max(_float(record, "total_tasks"), 1.0)
    return _float(record, "deadline_violations") / total_tasks


def compute_handover_frequency(record: Dict[str, Any]) -> float:
    if "handover_frequency" in record:
        return _float(record, "handover_frequency")
    total_user_seconds = _float(record, "total_user_seconds")
    if total_user_seconds <= 0.0:
        return 0.0
    return _float(record, "total_handovers") / total_user_seconds


def energy_per_successful_task(record: Dict[str, Any]) -> float:
    if "energy_per_successful_task" in record:
        return _float(record, "energy_per_successful_task")
    return _float(record, "total_energy") / max(_float(record, "completed_tasks"), 1.0)


def compute_model_selection_score(record: Dict[str, Any], metric_name: str) -> float:
    metric_name = metric_name or "reward"
    if metric_name == "reward":
        return _record_reward(record)
    if metric_name == "total_energy":
        return -_float(record, "total_energy")
    if metric_name == "energy_per_successful_task":
        return -energy_per_successful_task(record)
    if metric_name == "avg_delay":
        return -_float(record, "avg_delay", float("inf"))
    if metric_name == "task_success_rate":
        return _float(record, "task_success_rate", _float(record, "task_completion_rate"))
    if metric_name == "task_failure_rate":
        return -_float(record, "task_failure_rate", _float(record, "deadline_violation_rate"))
    if metric_name == "mec_load_fairness":
        return _float(record, "mec_load_fairness", _float(record, "avg_load_balance_score"))
    if metric_name == "avg_load_balance_score":
        return _float(record, "avg_load_balance_score", _float(record, "mec_load_fairness"))
    if metric_name == "active_load_balance_score":
        return _float(record, "active_load_balance_score", _float(record, "mec_load_fairness"))
    if HIGHER_IS_BETTER.get(metric_name, True):
        return _float(record, metric_name)
    return -_float(record, metric_name)


def extract_training_evaluation_records(payload: Dict[str, Any], training: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = list(payload.get("training_evaluation", []) or [])
    if not records:
        records = [record for record in training if "eval_mean_reward" in record]
    if not records:
        records = list(payload.get("evaluation", []) or [])
    return [
        record
        for record in records
        if "total_steps" in record and ("eval_mean_reward" in record or "mean_reward" in record)
    ]


def _best_record(
    records: Sequence[Dict[str, Any]],
    selection_metric: str,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    return max(records, key=lambda record: compute_model_selection_score(record, selection_metric))


def _method_name(config: Dict[str, Any], history_path: Path) -> str:
    return str(config.get("exp_name") or config.get("algorithm") or history_path.parent.name or "method")


def method_from_history(
    history_path: Path,
    label: Optional[str] = None,
    is_system: bool = False,
    selection_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a comparison method row from one training_history.json."""
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history: {history_path}")

    payload = _load_json(history_path)
    config = dict(payload.get("config", {}))
    metric = str(selection_metric or config.get("best_model_metric") or DEFAULT_SELECTION_METRIC)
    training_records = list(payload.get("training", []) or [])
    evaluation_records = extract_training_evaluation_records(payload, training_records)

    source_section = "evaluation" if evaluation_records else "training"
    selected = _best_record(evaluation_records or training_records, metric)
    if selected is None:
        raise ValueError(f"No training or evaluation records in: {history_path}")

    method_name = _method_name(config, history_path)
    display_name = label or pretty_method_name(method_name, is_system=is_system)
    handover_success_rate = _float(selected, "handover_success_rate")
    service_continuity_rate = _float(selected, "service_continuity_rate")
    load_balance = _float(
        selected,
        "mec_load_fairness",
        _float(selected, "active_load_balance_score", _float(selected, "avg_load_balance_score", 0.0)),
    )
    total_energy = _float(selected, "total_energy")
    completed_tasks = _float(selected, "completed_tasks")
    if completed_tasks <= 0.0 and "energy_per_successful_task" not in selected:
        completed_tasks = 1.0

    method = {
        "method": method_name,
        "display_name": display_name,
        "episodes": int(config.get("eval_episodes", len(evaluation_records))),
        "is_system": bool(is_system),
        "mean_reward": _record_reward(selected),
        "std_reward": _float(selected, "eval_std_reward"),
        "avg_delay": _float(selected, "avg_delay"),
        "total_energy": total_energy,
        "handover_success_rate": handover_success_rate,
        "handover_failure_rate": _float(
            selected,
            "handover_failure_rate",
            max(0.0, 1.0 - handover_success_rate),
        ),
        "forced_termination_rate": _float(
            selected,
            "forced_termination_rate",
            max(0.0, 1.0 - service_continuity_rate),
        ),
        "service_continuity_rate": service_continuity_rate,
        "service_availability_rate": _float(selected, "service_availability_rate", service_continuity_rate),
        "task_completion_rate": _float(selected, "task_completion_rate"),
        "task_success_rate": _float(selected, "task_success_rate", _float(selected, "task_completion_rate")),
        "task_failure_rate": _float(
            selected,
            "task_failure_rate",
            _float(selected, "deadline_violation_rate"),
        ),
        "task_settlement_rate": _float(selected, "task_settlement_rate", _float(selected, "task_resolution_rate")),
        "task_resolution_rate": _float(selected, "task_resolution_rate"),
        "pending_task_rate": _float(selected, "pending_task_rate"),
        "handover_frequency": _float(selected, "handover_frequency", compute_handover_frequency(selected)),
        "mec_load_fairness": load_balance,
        "active_load_balance_score": load_balance,
        "avg_load_balance_score": load_balance,
        "resolved_tasks": _float(selected, "resolved_tasks"),
        "pending_tasks": _float(selected, "pending_tasks"),
        "total_tasks": _float(selected, "total_tasks"),
        "completed_tasks": completed_tasks,
        "deadline_violations": _float(selected, "deadline_violations"),
        "deadline_violation_rate": _float(
            selected,
            "deadline_violation_rate",
            compute_deadline_violation_rate(selected),
        ),
        "energy_per_successful_task": _float(
            selected,
            "energy_per_successful_task",
            energy_per_successful_task({"total_energy": total_energy, "completed_tasks": completed_tasks}),
        ),
        "episode_metrics": list(payload.get("episode_metrics", []) or selected.get("episode_metrics", []) or []),
        "training_history": str(history_path),
        "source": f"training_history_{source_section}_best_{metric}",
    }
    return method


def _normalize_summary_methods(summary_path: Path) -> Tuple[List[Dict[str, Any]], Optional[Path], Dict[str, Any]]:
    payload = _load_json(summary_path)
    methods = [dict(method) for method in payload.get("methods", [])]
    for method in methods:
        method.setdefault("display_name", pretty_method_name(str(method.get("method", "")), bool(method.get("is_system"))))
        history = method.get("training_history") or method.get("history_path")
        if history:
            method["training_history"] = str(Path(str(history)))
    system_history = payload.get("training_history")
    if not system_history:
        for method in methods:
            if method.get("is_system") and method.get("training_history"):
                system_history = method["training_history"]
                break
    return methods, Path(str(system_history)) if system_history else None, payload


def _append_path(paths: List[Path], path: Optional[Path]) -> None:
    if path is not None:
        paths.append(path)


def annotate_priority_metrics(methods: Sequence[Dict[str, Any]], metric_name: str) -> List[Dict[str, Any]]:
    annotated = [dict(method) for method in methods]
    for method in annotated:
        method["selection_metric"] = metric_name
        method["selection_score"] = compute_model_selection_score(method, metric_name)
        wins: List[str] = []
        for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
            values = [_float(candidate, metric_key) for candidate in annotated]
            if not values:
                continue
            best = max(values) if HIGHER_IS_BETTER.get(metric_key, True) else min(values)
            if np.isclose(_float(method, metric_key), best, rtol=1e-9, atol=1e-9):
                wins.append(metric_label)
        method["primary_metric_wins"] = wins
        method["primary_metric_win_count"] = len(wins)
        method["primary_metric_wins_text"] = " | ".join(wins)
    return annotated


def order_methods(methods: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    systems = [method for method in methods if method.get("is_system")]
    baselines = [method for method in methods if not method.get("is_system")]
    baselines.sort(key=lambda method: (_float(method, "selection_score"), _float(method, "mean_reward")), reverse=True)
    return systems + baselines


def primary_metric_leaders(methods: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    leaders: Dict[str, List[str]] = {}
    for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
        values = [_float(method, metric_key) for method in methods]
        if not values:
            continue
        best = max(values) if HIGHER_IS_BETTER.get(metric_key, True) else min(values)
        leaders[metric_label] = [
            str(method.get("display_name", method.get("method", "")))
            for method in methods
            if np.isclose(_float(method, metric_key), best, rtol=1e-9, atol=1e-9)
        ]
    return leaders


def save_results_json(output_dir: Path, payload: Dict[str, Any]) -> Path:
    path = output_dir / "comparison_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def save_results_csv(output_dir: Path, methods: Sequence[Dict[str, Any]]) -> Path:
    path = output_dir / "comparison_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for method in methods:
            writer.writerow({key: method.get(key, "") for key in CSV_FIELDS})
    return path


def _history_training_records(history_path: Optional[Path]) -> List[Dict[str, Any]]:
    if history_path is None or not history_path.exists():
        return []
    return list(_load_json(history_path).get("training", []) or [])


def _training_xy(history_path: Optional[Path], metric_key: str = "mean_reward") -> Tuple[np.ndarray, np.ndarray]:
    records = _history_training_records(history_path)
    pairs = []
    for record in records:
        if "total_steps" not in record:
            continue
        value = _record_reward(record) if metric_key == "mean_reward" else _float(record, metric_key, np.nan)
        if np.isfinite(value):
            pairs.append((_float(record, "total_steps"), value))
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.array([item[0] for item in pairs], dtype=float), np.array([item[1] for item in pairs], dtype=float)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _format_steps(ax) -> None:
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def _method_color(index: int, method: Dict[str, Any]) -> str:
    if method.get("is_system"):
        return "#2F6C9E"
    return METHOD_COLORS[index % len(METHOD_COLORS)]


def _save(fig, output_path: Path) -> Path:
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_method_comparison(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None
    specs = [
        ("avg_delay", "Average Delay", "Average Delay", 1.0),
        ("task_success_rate", "Task Success Rate", "Task Success Rate (%)", 100.0),
        ("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)", 100.0),
        ("service_continuity_rate", "Service Continuity", "Service Continuity (%)", 100.0),
        ("energy_per_successful_task", "Energy per Successful Task", "Energy per Successful Task", 1.0),
        ("mec_load_fairness", "MEC Load Fairness", "MEC Load Fairness", 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = [str(method.get("display_name", method.get("method", ""))) for method in ordered]
    colors = [_method_color(index, method) for index, method in enumerate(ordered)]
    for ax, (key, title, ylabel, scale) in zip(axes.flat, specs):
        values = [_float(method, key) * scale for method in ordered]
        ax.bar(labels, values, color=colors, edgecolor="#303030", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Training Artifact Method Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir / "method_comparison.png")


def plot_training_curve_vs_baselines(
    history_path: Optional[Path],
    methods: Sequence[Dict[str, Any]],
    output_dir: Path,
    window: int,
) -> Optional[Path]:
    del history_path
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    plotted = False
    for index, method in enumerate(order_methods(methods)):
        steps, rewards = _training_xy(Path(str(method["training_history"])) if method.get("training_history") else None)
        if len(steps) == 0:
            continue
        plotted = True
        color = _method_color(index, method)
        ax.plot(
            steps,
            _smooth(rewards, max(window, 1)),
            color=color,
            linewidth=2.4 if method.get("is_system") else 1.8,
            label=str(method.get("display_name", method.get("method", ""))),
        )
        ax.scatter(steps, rewards, s=12, color=color, alpha=0.25)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title("Reward Convergence from Training Histories")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward")
    _format_steps(ax)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_dir / "reward_curve_vs_baselines.png")


def plot_step_metric_curves(history_path: Optional[Path], output_dir: Path, window: int) -> List[Path]:
    records = _history_training_records(history_path)
    if not records:
        return []

    generated: List[Path] = []
    groups = [
        (
            "training_qos_metrics_vs_steps.png",
            "Training QoS Metrics vs. Steps",
            [
                ("avg_delay", "Average Delay", 1.0),
                ("task_completion_rate", "Task Completion (%)", 100.0),
                ("service_continuity_rate", "Service Continuity (%)", 100.0),
                ("mec_load_fairness", "MEC Load Fairness", 1.0),
            ],
        ),
        (
            "reward_components_vs_steps.png",
            "Reward Components vs. Steps",
            [
                ("mean_reward", "Mean Reward", 1.0),
                ("reward_delay", "Delay Reward", 1.0),
                ("reward_energy", "Energy Reward", 1.0),
                ("reward_service_continuity", "Service Term", 1.0),
            ],
        ),
    ]
    for filename, title, specs in groups:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        plotted = False
        for ax, (key, label, scale) in zip(axes.flat, specs):
            steps, values = _training_xy(history_path, key)
            if len(steps) == 0:
                ax.axis("off")
                continue
            plotted = True
            ax.plot(steps, _smooth(values * scale, max(window, 1)), color="#2F6C9E", linewidth=1.8)
            ax.set_title(label)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(label)
            _format_steps(ax)
        if not plotted:
            plt.close(fig)
            continue
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        generated.append(_save(fig, output_dir / filename))
    return generated


def plot_delay_energy_tradeoff(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for index, method in enumerate(ordered):
        ax.scatter(
            _float(method, "avg_delay"),
            _float(method, "energy_per_successful_task"),
            s=150 if method.get("is_system") else 90,
            color=_method_color(index, method),
            edgecolor="#303030",
        )
        ax.annotate(str(method.get("display_name", method.get("method", ""))), (_float(method, "avg_delay"), _float(method, "energy_per_successful_task")), xytext=(6, 6), textcoords="offset points")
    ax.set_title("Delay-Energy Trade-off")
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Energy per Successful Task")
    fig.tight_layout()
    return _save(fig, output_dir / "delay_energy_tradeoff.png")


def plot_success_continuity_scatter(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for index, method in enumerate(ordered):
        ax.scatter(
            _float(method, "task_success_rate", _float(method, "task_completion_rate")) * 100.0,
            _float(method, "service_continuity_rate") * 100.0,
            s=150 if method.get("is_system") else 90,
            color=_method_color(index, method),
            edgecolor="#303030",
        )
        ax.annotate(str(method.get("display_name", method.get("method", ""))), (_float(method, "task_success_rate", _float(method, "task_completion_rate")) * 100.0, _float(method, "service_continuity_rate") * 100.0), xytext=(6, 6), textcoords="offset points")
    ax.set_title("Success-Continuity Trade-off")
    ax.set_xlabel("Task Success Rate (%)")
    ax.set_ylabel("Service Continuity Rate (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, output_dir / "success_continuity_tradeoff.png")


def plot_performance_radar(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None
    specs = [
        ("avg_delay", "Delay", False),
        ("task_success_rate", "Task", True),
        ("service_continuity_rate", "Continuity", True),
        ("mec_load_fairness", "Load", True),
        ("energy_per_successful_task", "Energy", False),
    ]
    raw = np.array([[_float(method, key) for key, _label, _higher in specs] for method in ordered], dtype=float)
    normalized = np.zeros_like(raw)
    for col, (key, _label, higher) in enumerate(specs):
        values = raw[:, col]
        if key.endswith("_rate") or key == "mec_load_fairness":
            normalized[:, col] = np.clip(values, 0.0, 1.0)
            if not higher:
                normalized[:, col] = 1.0 - normalized[:, col]
            continue
        low, high = float(np.min(values)), float(np.max(values))
        if np.isclose(low, high):
            normalized[:, col] = 1.0
        else:
            score = (values - low) / (high - low)
            normalized[:, col] = score if higher else 1.0 - score
    angles = np.linspace(0, 2 * np.pi, len(specs), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for index, method in enumerate(ordered):
        values = np.concatenate([normalized[index], normalized[index, :1]])
        ax.plot(closed_angles, values, label=str(method.get("display_name", method.get("method", ""))), color=_method_color(index, method))
        ax.fill(closed_angles, values, color=_method_color(index, method), alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels([label for _key, label, _higher in specs])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Normalized Multi-Metric Radar")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    return _save(fig, output_dir / "performance_radar.png")


def plot_paper_dashboard(
    history_path: Optional[Path],
    methods: Sequence[Dict[str, Any]],
    output_dir: Path,
    window: int,
) -> Optional[Path]:
    if not methods:
        return None
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    ax = fig.add_subplot(grid[0, :])
    plotted = False
    for index, method in enumerate(order_methods(methods)):
        steps, rewards = _training_xy(Path(str(method["training_history"])) if method.get("training_history") else history_path)
        if len(steps) == 0:
            continue
        plotted = True
        ax.plot(steps, _smooth(rewards, max(window, 1)), label=str(method.get("display_name", method.get("method", ""))), color=_method_color(index, method))
    if plotted:
        ax.set_title("Reward Convergence")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Mean Reward")
        _format_steps(ax)
        ax.legend()
    else:
        ax.axis("off")
    for position, (key, title, scale) in enumerate(
        [
            ("avg_delay", "Average Delay", 1.0),
            ("task_completion_rate", "Task Completion (%)", 100.0),
            ("mec_load_fairness", "MEC Load Fairness", 1.0),
        ]
    ):
        metric_ax = fig.add_subplot(grid[1, position])
        ordered = order_methods(methods)
        metric_ax.bar(
            [str(method.get("display_name", method.get("method", ""))) for method in ordered],
            [_float(method, key) * scale for method in ordered],
            color=[_method_color(index, method) for index, method in enumerate(ordered)],
        )
        metric_ax.set_title(title)
        metric_ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Training Artifact Dashboard", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.90, hspace=0.38, wspace=0.30)
    return _save(fig, output_dir / "paper_baseline_dashboard.png")


def plot_additional_metric_curves(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    plottable = [method for method in order_methods(methods) if method.get("episode_metrics")]
    if not plottable:
        return None
    specs = [
        ("avg_delay", "Average Delay", 1.0),
        ("task_completion_rate", "Task Completion (%)", 100.0),
        ("service_continuity_rate", "Service Continuity (%)", 100.0),
        ("mec_load_fairness", "MEC Load Fairness", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (key, title, scale) in zip(axes.flat, specs):
        for index, method in enumerate(plottable):
            records = list(method.get("episode_metrics", []) or [])
            episodes = [int(record.get("episode", idx + 1)) for idx, record in enumerate(records)]
            values = [_float(record, key) * scale for record in records]
            ax.plot(episodes, values, label=str(method.get("display_name", method.get("method", ""))), color=_method_color(index, method))
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout()
    return _save(fig, output_dir / "additional_metrics_episode_comparison.png")


def plot_reward_distribution(methods: Sequence[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    plottable = [method for method in order_methods(methods) if method.get("episode_metrics")]
    samples = [
        [_float(record, "reward", _record_reward(record)) for record in method.get("episode_metrics", [])]
        for method in plottable
    ]
    if not samples or not any(samples):
        return None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(samples, labels=[str(method.get("display_name", method.get("method", ""))) for method in plottable])
    ax.set_title("Reward Distribution Across Evaluation Episodes")
    ax.set_ylabel("Episode Reward")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, output_dir / "reward_distribution.png")


def write_manifest(
    output_dir: Path,
    sources: Sequence[str],
    generated_paths: Sequence[Path],
    plot_window: int,
    selection_metric: str,
) -> Path:
    manifest_path = output_dir / "plot_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "plot_only",
        "selection_metric": selection_metric,
        "plot_window": int(plot_window),
        "sources": list(sources),
        "generated_artifacts": [str(path) for path in generated_paths],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return manifest_path


def generate_plot_artifacts(
    methods: Sequence[Dict[str, Any]],
    output_dir: Path,
    system_history: Optional[Path],
    selection_metric: str,
    plot_window: int,
    sources: Sequence[str],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_publication_style()
    annotated = order_methods(annotate_priority_metrics(methods, selection_metric))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_mode": "plot_only",
        "best_model_metric": selection_metric,
        "compare_ranking_metric": selection_metric,
        "primary_compare_metrics": [label for _, label in PRIMARY_COMPARE_METRICS],
        "primary_metric_leaders": primary_metric_leaders(annotated),
        "training_history": str(system_history) if system_history else None,
        "methods": annotated,
    }
    if extra_payload:
        payload["source_summary"] = {
            key: value for key, value in extra_payload.items() if key != "methods"
        }

    generated: List[Path] = [
        save_results_json(output_dir, payload),
        save_results_csv(output_dir, annotated),
    ]

    _append_path(generated, plot_method_comparison(annotated, output_dir))
    _append_path(generated, plot_training_curve_vs_baselines(system_history, annotated, output_dir, window=plot_window))
    generated.extend(plot_step_metric_curves(system_history, output_dir, window=plot_window))
    _append_path(generated, plot_additional_metric_curves(annotated, output_dir))
    _append_path(generated, plot_delay_energy_tradeoff(annotated, output_dir))
    _append_path(generated, plot_success_continuity_scatter(annotated, output_dir))
    _append_path(generated, plot_performance_radar(annotated, output_dir))
    _append_path(generated, plot_reward_distribution(annotated, output_dir))
    _append_path(generated, plot_paper_dashboard(system_history, annotated, output_dir, window=plot_window))

    manifest = write_manifest(
        output_dir,
        sources=sources,
        generated_paths=generated,
        plot_window=plot_window,
        selection_metric=selection_metric,
    )
    generated.append(manifest)
    return generated


def generate_from_histories(
    histories: Sequence[HistorySpec],
    output_dir: Path,
    system_history: Optional[Path] = None,
    selection_metric: Optional[str] = None,
    plot_window: int = 5,
) -> List[Path]:
    if not histories:
        raise ValueError("At least one --history entry is required.")

    metric = selection_metric or DEFAULT_SELECTION_METRIC
    methods = [
        method_from_history(path, label=label, is_system=(index == 0), selection_metric=metric)
        for index, (label, path) in enumerate(histories)
    ]
    resolved_system_history = system_history or histories[0][1]
    return generate_plot_artifacts(
        methods=methods,
        output_dir=output_dir,
        system_history=resolved_system_history,
        selection_metric=metric,
        plot_window=plot_window,
        sources=[str(path) for _, path in histories],
    )


def generate_from_summary(
    summary_path: Path,
    output_dir: Path,
    system_history: Optional[Path],
    selection_metric: Optional[str],
    plot_window: int,
) -> List[Path]:
    if summary_path.is_dir():
        summary_path = summary_path / "comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing comparison summary: {summary_path}")

    methods, summary_system_history, payload = _normalize_summary_methods(summary_path)
    metric = selection_metric or str(
        payload.get("compare_ranking_metric") or payload.get("best_model_metric") or DEFAULT_SELECTION_METRIC
    )
    return generate_plot_artifacts(
        methods=methods,
        output_dir=output_dir,
        system_history=system_history or summary_system_history,
        selection_metric=metric,
        plot_window=plot_window,
        sources=[str(summary_path)],
        extra_payload=payload,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate comparison figures from existing training histories or comparison summaries."
    )
    parser.add_argument(
        "histories",
        nargs="*",
        help="training_history.json path, optionally as LABEL=PATH.",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="training_history.json path, optionally as LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--comparison-summary",
        "--compare-summary",
        dest="comparison_summary",
        type=str,
        default=None,
        help="Existing baseline_compare directory or comparison_summary.json to replot.",
    )
    parser.add_argument("--system-history", type=str, default=None)
    parser.add_argument("--output-dir", "-o", type=str, default="results/plot_only_comparison")
    parser.add_argument("--selection-metric", type=str, default=None)
    parser.add_argument("--plot-window", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    system_history = Path(args.system_history) if args.system_history else None

    if args.comparison_summary:
        generated = generate_from_summary(
            summary_path=Path(args.comparison_summary),
            output_dir=output_dir,
            system_history=system_history,
            selection_metric=args.selection_metric,
            plot_window=args.plot_window,
        )
    else:
        history_specs = [parse_history_spec(value) for value in [*args.history, *args.histories]]
        generated = generate_from_histories(
            histories=history_specs,
            output_dir=output_dir,
            system_history=system_history,
            selection_metric=args.selection_metric,
            plot_window=args.plot_window,
        )

    print(f"Plot-only output dir: {output_dir}")
    for path in generated:
        print(f"Generated artifact: {path}")


if __name__ == "__main__":
    main()
