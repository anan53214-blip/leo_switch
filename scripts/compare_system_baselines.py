#!/usr/bin/env python3
"""
Evaluate heuristic baselines and compare them with a trained system method.

Example:
    python scripts/compare_system_baselines.py ^
        --system-run-dir results/full_train_delay_focus ^
        --episodes 5 ^
        --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv, summarize_env_stats
from src.environment.user import UserState
from src.algorithm.replay_buffer import MultiAgentReplayBuffer
from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig

try:
    from scripts.train import (
        BEST_MODEL_METRIC_CHOICES,
        HANMADDPGTrainer,
        HANMAPPOTrainer,
        HANPDQNTrainer,
        TrainConfig,
        compute_model_selection_score,
        energy_per_resolved_task,
    )
except ModuleNotFoundError:
    # Compatible with direct execution: python scripts/compare_system_baselines.py
    from train import (
        BEST_MODEL_METRIC_CHOICES,
        HANMADDPGTrainer,
        HANMAPPOTrainer,
        HANPDQNTrainer,
        TrainConfig,
        compute_model_selection_score,
        energy_per_resolved_task,
    )

DelayOnlyEnv = None
DelayOnlyTrainer = None
try:
    from scripts.train_delay_only import DelayOnlyEnv, DelayOnlyTrainer
except ModuleNotFoundError:
    try:
        from train_delay_only import DelayOnlyEnv, DelayOnlyTrainer
    except ModuleNotFoundError:
        pass

EnergyOnlyEnv = None
EnergyOnlyTrainer = None
try:
    from scripts.train_energy_only import EnergyOnlyEnv, EnergyOnlyTrainer
except ModuleNotFoundError:
    try:
        from train_energy_only import EnergyOnlyEnv, EnergyOnlyTrainer
    except ModuleNotFoundError:
        pass


DEFAULT_BASELINES = [
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
    "maddpg",
    "pdqn",
    "mappo_no_han",
    "han_maddpg",
    "han_pdqn",
]

DEFAULT_SYSTEM_RUN_DIR = PROJECT_ROOT / "results" / "full_train_latency_priority"
DEFAULT_SYSTEM_EXP_NAME = "han_mappo_latency_priority"
DEFAULT_TOTAL_TIMESTEPS = 1_200_000
DEFAULT_PLOT_WINDOW = 5
DEFAULT_SELECTION_METRIC = "effective_latency_score"
TRAIN_ARTIFACT_FILENAMES = (
    "training_history.json",
    "best_model.pt",
    "final_model.pt",
)

PRIMARY_COMPARE_METRICS = [
    ("avg_delay", "Average Delay"),
    ("service_continuity_rate", "Service Continuity"),
    ("service_availability_rate", "Service Availability"),
    ("task_success_rate", "Task Success"),
    ("deadline_violation_rate", "Deadline Violation"),
]

DISPLAY_NAME_MAP = {
    "random": "Random",
    "min_distance": "Min-Distance",
    "full_local": "Full-Local",
    "joint_greedy": "Joint Greedy",
    "dqn": "DQN",
    "maddpg": "MADDPG",
    "pdqn": "PDQN",
    "mappo_no_han": "MAPPO (no HAN)",
    "han_maddpg": "HAN+MADDPG",
    "han_pdqn": "HAN+PDQN",
}

SUMMARY_METRIC_KEYS = [
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
    "avg_load_balance_score",
    "resolved_tasks",
    "pending_tasks",
    "total_tasks",
    "completed_tasks",
    "deadline_violations",
    "deadline_violation_rate",
    "effective_latency_score",
]

REWARD_BREAKDOWN_KEYS = [
    "reward_delay",
    "reward_energy",
    "reward_qos",
    "reward_service_continuity",
    "reward_handover",
    "reward_load_balance",
    "reward_enqueue",
    "penalty_deadline",
    "penalty_queue_full",
    "penalty_invalid_action",
    "penalty_blocked",
    "penalty_failed_handover",
    "penalty_handover_cost",
]

HIGHER_IS_BETTER = {
    "mean_reward": True,
    "reward": True,
    "handover_success_rate": True,
    "service_continuity_rate": True,
    "service_availability_rate": True,
    "task_completion_rate": True,
    "task_success_rate": True,
    "task_failure_rate": False,
    "task_settlement_rate": True,
    "task_resolution_rate": True,
    "effective_latency_score": True,
    "avg_load_balance_score": True,
    "handover_failure_rate": False,
    "forced_termination_rate": False,
    "avg_delay": False,
    "total_energy": False,
    "pending_task_rate": False,
    "deadline_violation_rate": False,
}

PAPER_METRIC_PLOTS = [
    ("avg_delay", "Average Delay", "Avg Delay (s)"),
    ("service_continuity_rate", "Service Continuity", "Rate (%)"),
    ("service_availability_rate", "Service Availability", "Rate (%)"),
    ("task_completion_rate", "Task Completion", "Rate (%)"),
    ("total_energy", "Energy Consumption", "Total Energy (J)"),
    ("handover_failure_rate", "Handover Failure", "Rate (%)"),
]

CORE_BAR_METRICS = [
    ("avg_delay", "Average Delay", "Average Delay (ms)"),
    ("effective_latency_score", "Effective Latency Score", "Effective Latency Score"),
    ("service_continuity_rate", "Service Continuity Rate", "Service Continuity Rate (%)"),
    ("task_success_rate", "Task Success Rate", "Task Success Rate (%)"),
    ("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)"),
    ("avg_load_balance_score", "Avg Load Balance Score", "Avg Load Balance Score"),
]

TRAINING_QOS_STEP_METRICS = [
    ("mean_reward", "Reward", "Mean Episode Reward", 1.0),
    ("avg_delay", "Average Delay", "Average Delay (ms)", 1000.0),
    ("effective_latency_score", "Effective Latency Score", "Score", 1.0),
    ("service_continuity_rate", "Service Continuity", "Rate (%)", 100.0),
    ("task_success_rate", "Task Success", "Rate (%)", 100.0),
    ("task_settlement_rate", "Task Settlement", "Rate (%)", 100.0),
    ("deadline_violation_rate", "Deadline Violation", "Rate (%)", 100.0),
    ("avg_load_balance_score", "Load Balance", "Score", 1.0),
]

REWARD_COMPONENT_STEP_METRICS = [
    ("mean_reward", "Total Reward", "Reward", 1.0),
    ("reward_delay", "Delay Reward", "Reward Term", 1.0),
    ("reward_energy", "Energy Reward", "Reward Term", 1.0),
    ("reward_qos", "QoS Reward", "Reward Term", 1.0),
    ("reward_service_continuity", "Service Continuity Reward", "Reward Term", 1.0),
    ("reward_handover", "Handover Reward", "Reward Term", 1.0),
    ("reward_load_balance", "Load Balance Reward", "Reward Term", 1.0),
    ("reward_enqueue", "Enqueue Reward", "Reward Term", 1.0),
    ("penalty_deadline", "Deadline Penalty", "Penalty Term", 1.0),
    ("penalty_failed_handover", "Failed Handover Penalty", "Penalty Term", 1.0),
    ("penalty_handover_cost", "Handover Cost Penalty", "Penalty Term", 1.0),
    ("penalty_blocked", "Blocked-Service Penalty", "Penalty Term", 1.0),
    ("penalty_queue_full", "Queue-Full Penalty", "Penalty Term", 1.0),
]

RADAR_METRICS = [
    ("effective_latency_score", "Effective\nLatency", True),
    ("service_continuity_rate", "Continuity", True),
    ("task_success_rate", "Task\nSuccess", True),
    ("task_completion_rate", "Completion", True),
    ("avg_load_balance_score", "Load\nBalance", True),
    ("avg_delay", "Low\nDelay", False),
    ("deadline_violation_rate", "Deadline\nReliability", False),
]

PAPER_DASHBOARD_LEFT_METRICS = [
    ("handover_success_rate", "HO Success"),
    ("service_continuity_rate", "Continuity"),
    ("task_completion_rate", "Task Completion"),
]

PAPER_DASHBOARD_RIGHT_METRICS = [
    ("avg_delay", "Delay"),
    ("total_energy", "Energy"),
    ("avg_load_balance_score", "Load Balance"),
]

CORE_EPISODE_PLOTS = [
    ("reward", "Reward Comparison Across Evaluation Episodes", "Episode Reward", "reward_episode_comparison.png"),
    ("avg_delay", "Delay Comparison Across Evaluation Episodes", "Average Delay (s)", "delay_episode_comparison.png"),
    ("total_energy", "Energy Comparison Across Evaluation Episodes", "Total Energy (J)", "energy_episode_comparison.png"),
]

ADDITIONAL_EPISODE_METRICS = [
    ("handover_success_rate", "Handover Success Rate"),
    ("service_continuity_rate", "Service Continuity Rate"),
    ("service_availability_rate", "Service Availability Rate"),
    ("handover_failure_rate", "Handover Failure Rate"),
    ("forced_termination_rate", "Forced Termination Rate"),
    ("deadline_violation_rate", "Deadline Violation Rate"),
    ("avg_load_balance_score", "Load Balance Score"),
]

SYSTEM_STYLE = {
    "color": "#B03A2E",
    "linestyle": "-",
    "marker": "*",
    "linewidth": 3.0,
    "markersize": 11,
    "hatch": "///",
    "scatter_size": 280,
}

BASELINE_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#9C755F",
    "#76B7B2",
    "#EDC948",
    "#BAB0AC",
    "#AF7AA1",
]

BASELINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BASELINE_LINESTYLES = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (7, 2, 1, 2))]
BAR_HATCH_PATTERNS = ["///", "\\\\\\", "xx", "--", "oo", "++", "..", "**"]

LEARNED_BASELINE_COLORS = {
    "dqn": "#4E79A7",
    "maddpg": "#AF7AA1",
    "pdqn": "#EDC948",
    "mappo_no_han": "#59A14F",
    "han_maddpg": "#B07AA1",
    "han_pdqn": "#F28E2B",
}

SCATTER_LABEL_OFFSETS = {
    "HAN+MAPPO": (12, -16),
    "Random": (10, 8),
    "Min-Distance": (10, 12),
    "Full-Local": (10, -10),
    "Joint Greedy": (10, -12),
    "DQN": (10, 10),
    "MADDPG": (10, 12),
    "PDQN": (10, 12),
    "MAPPO (no HAN)": (10, -14),
    "HAN+MADDPG": (10, 12),
    "HAN+PDQN": (10, 12),
}

PAPER_COLORS = {
    "primary": "#0F4C81",
    "secondary": "#B03A2E",
    "success": "#1E8449",
    "warning": "#AF601A",
    "info": "#2471A3",
    "dark": "#283747",
    "muted": "#7B7D7D",
    "fill_alpha": 0.16,
}


def detect_objective(config: Dict) -> str:
    combined = " ".join(
        [
            str(config.get("exp_name", "")).lower(),
            str(config.get("save_path", "")).lower(),
        ]
    )
    if "delay_only" in combined:
        return "delay_only"
    if "energy_only" in combined:
        return "energy_only"
    return "multi_objective"


def setup_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.grid": True,
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
        }
    )


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(float), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def reward_smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    """Use a lighter reward smoothing window so reward curves retain natural fluctuation."""
    if len(values) == 0:
        return values, 0
    if window <= 1 or len(values) < 3:
        return values.astype(float), 1

    effective_window = min(int(window), 7, len(values))
    effective_window = max(effective_window, 3)
    return smooth(values, effective_window), effective_window


def draw_raw_reward_shadow(
    ax,
    steps: np.ndarray,
    raw_rewards: np.ndarray,
    smoothed_rewards: np.ndarray,
    color: str,
    alpha: float = 0.18,
    label: Optional[str] = None,
) -> None:
    """Render raw reward values as a translucent fluctuation area."""
    if len(raw_rewards) == 0:
        return
    fill_alpha = min(alpha * 0.55, 0.12)
    line_alpha = min(alpha + 0.08, 0.30)
    ax.fill_between(
        steps,
        raw_rewards,
        smoothed_rewards,
        color=color,
        alpha=fill_alpha,
        linewidth=0,
        label=label,
        zorder=1,
    )
    ax.plot(steps, raw_rewards, color=color, alpha=line_alpha, linewidth=0.55, zorder=2)


def draw_raw_metric_shadow(
    ax,
    steps: np.ndarray,
    raw_values: np.ndarray,
    smoothed_values: np.ndarray,
    color: str,
    alpha: float = 0.18,
) -> None:
    if len(raw_values) == 0:
        return
    fill_alpha = min(alpha * 0.55, 0.12)
    line_alpha = min(alpha + 0.08, 0.30)
    ax.fill_between(
        steps,
        raw_values,
        smoothed_values,
        color=color,
        alpha=fill_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        steps,
        raw_values,
        color=color,
        linestyle="--",
        alpha=line_alpha,
        linewidth=0.65,
        zorder=2,
    )


def extract_training_reward_curve(training: Sequence[Dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-training-update rewards for a comparable learning curve."""
    source_records = list(training)
    steps = np.array([record.get("total_steps", 0) for record in source_records], dtype=float)
    rewards = np.array(
        [
            record.get("mean_reward", record.get("recent_mean_reward", record.get("eval_mean_reward", 0.0)))
            for record in source_records
        ],
        dtype=float,
    )
    return steps, rewards


def metric_record_value(record: Dict, metric_key: str) -> Optional[float]:
    if metric_key == "mean_reward":
        for key in ("mean_reward", "recent_mean_reward", "eval_mean_reward", "reward"):
            if key in record:
                return float(record.get(key, 0.0))
        return None
    if metric_key in record:
        return float(record.get(metric_key, 0.0))
    return None


def extract_training_metric_curve(records: Sequence[Dict], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for record in records:
        if "total_steps" not in record:
            continue
        value = metric_record_value(record, metric_key)
        if value is None:
            continue
        step = float(record.get("total_steps", 0.0))
        if np.isfinite(step) and np.isfinite(value):
            pairs.append((step, float(value)))

    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)

    pairs.sort(key=lambda item: item[0])
    return (
        np.array([item[0] for item in pairs], dtype=float),
        np.array([item[1] for item in pairs], dtype=float),
    )


def extract_training_evaluation_records(payload: Dict, training: Sequence[Dict]) -> List[Dict]:
    """Return sparse training-evaluation checkpoints, excluding final episode metrics."""
    evaluation_records = payload.get("training_evaluation", [])
    if not evaluation_records:
        evaluation_records = [record for record in training if "eval_mean_reward" in record]
    if not evaluation_records:
        evaluation_records = payload.get("evaluation", [])
    return [
        record
        for record in evaluation_records
        if "total_steps" in record and "eval_mean_reward" in record
    ]


def load_training_curve_from_path(history_path: Optional[Path]) -> tuple[np.ndarray, np.ndarray, List[Dict]]:
    if history_path is None or not history_path.exists():
        return np.array([], dtype=float), np.array([], dtype=float), []

    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training = payload.get("training", [])
    if not training:
        evaluation = extract_training_evaluation_records(payload, training)
        if not evaluation:
            return np.array([], dtype=float), np.array([], dtype=float), []
        steps = np.array([record.get("total_steps", 0) for record in evaluation], dtype=float)
        rewards = np.array([record.get("eval_mean_reward", 0.0) for record in evaluation], dtype=float)
        valid_mask = np.isfinite(steps) & np.isfinite(rewards)
        order = np.argsort(steps[valid_mask])
        return steps[valid_mask][order], rewards[valid_mask][order], evaluation

    steps, rewards = extract_training_reward_curve(training)
    valid_mask = np.isfinite(steps) & np.isfinite(rewards)
    steps = steps[valid_mask]
    rewards = rewards[valid_mask]
    if len(steps) == 0:
        return steps, rewards, extract_training_evaluation_records(payload, training)

    order = np.argsort(steps)
    return steps[order], rewards[order], extract_training_evaluation_records(payload, training)


def load_training_metric_curve_from_path(history_path: Optional[Path], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    if history_path is None or not history_path.exists():
        return np.array([], dtype=float), np.array([], dtype=float)

    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training = payload.get("training", [])
    steps, values = extract_training_metric_curve(training, metric_key)
    if len(steps) > 0:
        return steps, values

    evaluation_records = []
    evaluation_records.extend(payload.get("training_evaluation", []))
    evaluation_records.extend(payload.get("evaluation", []))
    return extract_training_metric_curve(evaluation_records, metric_key)


def load_training_history(history_path: Path) -> Dict:
    with history_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_summary_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find comparison summary: {path}")
    return path


def load_reused_methods(
    sources: Sequence[str],
    include_methods: Sequence[str],
    exclude_methods: Sequence[str],
) -> List[Dict]:
    """Load selected non-system methods from previous comparison summaries."""
    include = {normalize_baseline_name(name) for name in include_methods}
    exclude = {normalize_baseline_name(name) for name in exclude_methods}
    reused: List[Dict] = []
    seen: set[str] = set()

    for source in sources:
        summary_path = resolve_summary_path(source)
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for method in payload.get("methods", []):
            method_name = normalize_baseline_name(str(method.get("method", "")))
            if not method_name or method.get("is_system"):
                continue
            if include and method_name not in include:
                continue
            if method_name in exclude or method_name in seen:
                continue
            reused_method = dict(method)
            reused_method["method"] = method_name
            reused_method.setdefault("display_name", pretty_method_name(method_name, is_system=False))
            reused_method.setdefault("source", f"reused_from_{summary_path.parent.name}")
            reused.append(reused_method)
            seen.add(method_name)

    return reused


def method_training_history_path(method: Dict, output_dir: Optional[Path] = None) -> Optional[Path]:
    for key in ("training_history", "history_path"):
        value = method.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path

    checkpoint = method.get("checkpoint")
    if checkpoint:
        history_path = Path(str(checkpoint)).parent / "training_history.json"
        if history_path.exists():
            return history_path

    method_name = str(method.get("method", ""))
    if output_dir is not None and method_name:
        history_path = output_dir / "learned_baselines" / method_name / "training_history.json"
        if history_path.exists():
            return history_path

    return None


def compute_confidence_band(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = smooth(values, window)
    if len(values) == 0:
        return mean, mean, mean
    std = np.zeros(len(values), dtype=float)
    half_window = max(window // 2, 1)
    for index in range(len(values)):
        lo = max(index - half_window, 0)
        hi = min(index + half_window + 1, len(values))
        std[index] = float(np.std(values[lo:hi]))
    return mean, mean - std, mean + std


def format_steps(value, _position) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def save_figure(fig, output_path: Path) -> Path:
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_env_config_from_train_config(config: Dict, seed: Optional[int], max_steps: Optional[int]) -> EnvConfig:
    return EnvConfig(
        num_planes=int(config.get("num_planes", EnvConfig.num_planes)),
        sats_per_plane=int(config.get("sats_per_plane", EnvConfig.sats_per_plane)),
        altitude_km=float(config.get("altitude_km", EnvConfig.altitude_km)),
        inclination_deg=float(config.get("inclination_deg", EnvConfig.inclination_deg)),
        num_users=int(config.get("num_users", EnvConfig.num_users)),
        max_steps=int(max_steps if max_steps is not None else config.get("max_steps", EnvConfig.max_steps)),
        time_step_sec=float(config.get("time_step_sec", EnvConfig.time_step_sec)),
        min_effective_offload_ratio=float(
            config.get("min_effective_offload_ratio", EnvConfig.min_effective_offload_ratio)
        ),
        reward_delay_weight=float(config.get("reward_delay_weight", EnvConfig.reward_delay_weight)),
        reward_energy_weight=float(config.get("reward_energy_weight", EnvConfig.reward_energy_weight)),
        reward_handover_weight=float(config.get("reward_handover_weight", EnvConfig.reward_handover_weight)),
        reward_load_balance_weight=float(
            config.get("reward_load_balance_weight", EnvConfig.reward_load_balance_weight)
        ),
        reward_qos_weight=float(config.get("reward_qos_weight", EnvConfig.reward_qos_weight)),
        reward_service_continuity_weight=float(
            config.get("reward_service_continuity_weight", EnvConfig.reward_service_continuity_weight)
        ),
        reward_failed_handover_penalty=float(
            config.get("reward_failed_handover_penalty", EnvConfig.reward_failed_handover_penalty)
        ),
        seed=seed if seed is not None else config.get("seed"),
    )


def build_env_for_objective(
    objective: str,
    config: Dict,
    seed: Optional[int],
    max_steps: Optional[int],
) -> LEOSatelliteEnv:
    env_config = build_env_config_from_train_config(config, seed=seed, max_steps=max_steps)
    if objective == "delay_only":
        if DelayOnlyEnv is None:
            raise ModuleNotFoundError("delay_only objective requires scripts/train_delay_only.py")
        env_config.reward_delay_weight = 1.0
        env_config.reward_energy_weight = 0.0
        env_config.reward_handover_weight = 0.0
        env_config.reward_load_balance_weight = 0.0
        env_config.reward_qos_weight = 0.0
        return DelayOnlyEnv(
            env_config,
            delay_violation_penalty=float(config.get("delay_violation_penalty", 5.0)),
            failed_handover_penalty=float(config.get("failed_handover_penalty", 1.0)),
        )
    if objective == "energy_only":
        if EnergyOnlyEnv is None:
            raise ModuleNotFoundError("energy_only objective requires scripts/train_energy_only.py")
        env_config.reward_delay_weight = 0.0
        env_config.reward_energy_weight = 1.0
        env_config.reward_handover_weight = 0.0
        env_config.reward_load_balance_weight = 0.0
        env_config.reward_qos_weight = 0.0
        return EnergyOnlyEnv(
            env_config,
            qos_unmet_task_penalty=float(config.get("qos_unmet_task_penalty", 0.5)),
            delay_violation_penalty=float(config.get("delay_violation_penalty", 1.0)),
            failed_handover_penalty=float(config.get("failed_handover_penalty", 0.5)),
        )
    return LEOSatelliteEnv(env_config)


def build_default_train_config(
    objective: str,
    seed: int,
    max_steps: int,
    num_users: int,
    best_model_metric: str,
) -> Dict:
    config = asdict(TrainConfig())
    config["seed"] = seed
    config["max_steps"] = max_steps
    config["num_users"] = num_users
    config["total_timesteps"] = DEFAULT_TOTAL_TIMESTEPS
    config["save_path"] = str(DEFAULT_SYSTEM_RUN_DIR)
    config["best_model_metric"] = best_model_metric
    if objective == "delay_only":
        config["exp_name"] = "han_mappo_delay_only"
        config["reward_delay_weight"] = 1.0
        config["reward_energy_weight"] = 0.0
        config["reward_handover_weight"] = 0.0
        config["reward_load_balance_weight"] = 0.0
        config["reward_qos_weight"] = 0.0
    elif objective == "energy_only":
        config["exp_name"] = "han_mappo_energy_only"
        config["reward_delay_weight"] = 0.0
        config["reward_energy_weight"] = 1.0
        config["reward_handover_weight"] = 0.0
        config["reward_load_balance_weight"] = 0.0
        config["reward_qos_weight"] = 0.0
    else:
        config["exp_name"] = DEFAULT_SYSTEM_EXP_NAME
        config["reward_delay_weight"] = 1.4
        config["reward_energy_weight"] = 0.4
        config["reward_handover_weight"] = 0.3
        config["reward_load_balance_weight"] = 0.1
        config["reward_qos_weight"] = 0.4
    return config


def infer_system_artifacts(
    system_run_dir: Optional[str],
    system_checkpoint: Optional[str],
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    run_dir = Path(system_run_dir).resolve() if system_run_dir else None
    checkpoint = Path(system_checkpoint).resolve() if system_checkpoint else None

    if checkpoint and run_dir is None:
        run_dir = checkpoint.parent

    history_path = None
    if run_dir is not None:
        candidate = run_dir / "training_history.json"
        if candidate.exists():
            history_path = candidate

        if checkpoint is None:
            best_candidate = run_dir / "best_model.pt"
            final_candidate = run_dir / "final_model.pt"
            if best_candidate.exists():
                checkpoint = best_candidate
            elif final_candidate.exists():
                checkpoint = final_candidate

    return run_dir, checkpoint, history_path


def list_training_artifacts(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []

    artifacts: List[Path] = []
    for filename in TRAIN_ARTIFACT_FILENAMES:
        candidate = run_dir / filename
        if candidate.exists():
            artifacts.append(candidate)
    artifacts.extend(sorted(run_dir.glob("checkpoint_*.pt")))
    return artifacts


def prepare_system_run_dir(
    requested_run_dir: str,
    timestamp: str,
    resume_system: bool,
    overwrite_system_run_dir: bool,
) -> Path:
    run_dir = Path(requested_run_dir).resolve()
    if resume_system or overwrite_system_run_dir:
        return run_dir

    artifacts = list_training_artifacts(run_dir)
    if not artifacts:
        return run_dir

    fresh_run_dir = run_dir.parent / f"{run_dir.name}_{timestamp}"
    suffix = 1
    while fresh_run_dir.exists():
        fresh_run_dir = run_dir.parent / f"{run_dir.name}_{timestamp}_{suffix:02d}"
        suffix += 1

    print(
        "Existing training artifacts detected in "
        f"{run_dir}. Starting a fresh train_compare run in {fresh_run_dir} "
        "instead. Use --resume-system to continue training there, or "
        "--overwrite-system-run-dir to intentionally reuse the existing path."
    )
    return fresh_run_dir


def torch_load_trusted_checkpoint(path: Path, map_location):
    """Load a trusted project checkpoint across PyTorch versions.

    PyTorch 2.6 changed torch.load's default to weights_only=True. These
    project checkpoints contain Python/numpy metadata in addition to tensors,
    so trusted local checkpoints need weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_run_config(checkpoint: Optional[Path], history_path: Optional[Path]) -> Dict:
    if history_path and history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload.get("config", {}))
    if checkpoint and checkpoint.exists():
        payload = torch_load_trusted_checkpoint(checkpoint, map_location="cpu")
        return dict(payload.get("config", {}))
    raise FileNotFoundError("No checkpoint or training_history.json could be loaded.")


def pretty_method_name(name: str, is_system: bool) -> str:
    if is_system:
        return "HAN+MAPPO"
    base_name = name.split("(", 1)[0]
    return DISPLAY_NAME_MAP.get(name, DISPLAY_NAME_MAP.get(base_name, name))


def normalize_baseline_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def compute_deadline_violation_rate(summary: Dict) -> float:
    total_tasks = float(summary.get("total_tasks", 0.0))
    return float(summary.get("deadline_violations", 0.0)) / max(total_tasks, 1.0)


def build_episode_records(rewards: Sequence[float], summaries: Sequence[Dict]) -> List[Dict]:
    episode_records: List[Dict] = []
    for episode_index, (reward, summary) in enumerate(zip(rewards, summaries), start=1):
        record = {
            "episode": episode_index,
            "reward": float(reward),
        }
        for key in SUMMARY_METRIC_KEYS:
            if key == "deadline_violation_rate":
                continue
            record[key] = float(summary.get(key, 0.0))
        for key in REWARD_BREAKDOWN_KEYS:
            record[key] = float(summary.get(key, 0.0))
        record["deadline_violation_rate"] = compute_deadline_violation_rate(summary)
        episode_records.append(record)
    return episode_records


def summarize_results(
    name: str,
    rewards: Sequence[float],
    summaries: Sequence[Dict],
    extra: Optional[Dict] = None,
    is_system: bool = False,
) -> Dict:
    episode_metrics = build_episode_records(rewards, summaries)
    result = {
        "method": name,
        "display_name": pretty_method_name(name, is_system=is_system),
        "episodes": len(rewards),
        "is_system": bool(is_system),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "episode_metrics": episode_metrics,
    }
    for key in SUMMARY_METRIC_KEYS:
        values = [float(record.get(key, 0.0)) for record in episode_metrics]
        result[key] = float(np.mean(values)) if values else 0.0
    for key in REWARD_BREAKDOWN_KEYS:
        values = [float(record.get(key, 0.0)) for record in episode_metrics]
        result[key] = float(np.mean(values)) if values else 0.0
    if extra:
        result.update(extra)
    return result


def selection_score(method: Dict, metric_name: str) -> float:
    return float(compute_model_selection_score(method, metric_name))


def annotate_priority_metrics(methods: Sequence[Dict], metric_name: str) -> List[Dict]:
    annotated = [dict(method) for method in methods]
    for method in annotated:
        method["selection_metric"] = metric_name
        method["selection_score"] = selection_score(method, metric_name)
        method["energy_per_resolved_task"] = float(energy_per_resolved_task(method))
        method["primary_metric_wins"] = []

    for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
        values = [float(method.get(metric_key, 0.0)) for method in annotated]
        if not values:
            continue
        best_value = (
            max(values)
            if HIGHER_IS_BETTER.get(metric_key, True)
            else min(values)
        )
        for method, value in zip(annotated, values):
            if np.isclose(value, best_value, rtol=1e-9, atol=1e-9):
                method["primary_metric_wins"].append(metric_label)

    for method in annotated:
        method["primary_metric_win_count"] = len(method["primary_metric_wins"])
        method["primary_metric_wins_text"] = " | ".join(method["primary_metric_wins"])

    return annotated


def primary_metric_leaders(methods: Sequence[Dict]) -> Dict[str, List[str]]:
    leaders: Dict[str, List[str]] = {}
    for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
        values = [float(method.get(metric_key, 0.0)) for method in methods]
        if not values:
            continue
        best_value = (
            max(values)
            if HIGHER_IS_BETTER.get(metric_key, True)
            else min(values)
        )
        leaders[metric_label] = [
            method.get("display_name", method.get("method", ""))
            for method, value in zip(methods, values)
            if np.isclose(value, best_value, rtol=1e-9, atol=1e-9)
        ]
    return leaders


def build_method_styles(methods: Sequence[Dict]) -> Dict[str, Dict]:
    styles: Dict[str, Dict] = {}
    baseline_index = 0
    for method in order_methods(methods):
        method_key = str(method.get("method", ""))
        if method.get("is_system"):
            styles[method_key] = dict(SYSTEM_STYLE)
            continue

        method_name = str(method.get("method", ""))
        display_name = str(method.get("display_name", method_name))
        if method_name in LEARNED_BASELINE_COLORS:
            color = LEARNED_BASELINE_COLORS[method_name]
        elif method_name == "joint_greedy" or display_name == "Joint Greedy":
            color = PAPER_COLORS["primary"]
        else:
            color = BASELINE_COLORS[baseline_index % len(BASELINE_COLORS)]
        style = {
            "color": color,
            "linestyle": BASELINE_LINESTYLES[baseline_index % len(BASELINE_LINESTYLES)],
            "marker": BASELINE_MARKERS[baseline_index % len(BASELINE_MARKERS)],
            "linewidth": 1.8,
            "markersize": 5.5,
            "hatch": BAR_HATCH_PATTERNS[(baseline_index + 1) % len(BAR_HATCH_PATTERNS)],
            "scatter_size": 150,
        }
        styles[method_key] = style
        baseline_index += 1
    return styles


def order_methods(methods: Sequence[Dict]) -> List[Dict]:
    systems = [method for method in methods if method.get("is_system")]
    baselines = [method for method in methods if not method.get("is_system")]
    baselines.sort(
        key=lambda item: (item.get("selection_score", item.get("mean_reward", 0.0)), item.get("mean_reward", 0.0)),
        reverse=True,
    )
    return systems + baselines


def choose_best_index(values: Sequence[float], higher_is_better: bool) -> int:
    if not values:
        return -1
    fn = np.argmax if higher_is_better else np.argmin
    return int(fn(np.asarray(values, dtype=float)))


def current_visibility(env: LEOSatelliteEnv, user) -> Optional[object]:
    if user.serving_satellite < 0:
        return None
    visible = env._get_satellite_visibility(user, user.serving_satellite)
    return visible if visible is not None and visible.is_visible else None


def active_queue_remaining_cycles(server) -> float:
    if server is None:
        return 0.0
    return float(
        sum(
            task.get("remaining_cycles", 0.0)
            for task in server.task_queue
            if task.get("status") in ("queued", "processing")
        )
    )


class BasePolicy:
    name = "base"

    def begin_episode(self, env: LEOSatelliteEnv) -> None:
        del env

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        raise NotImplementedError


class SimpleHeuristicPolicy(BasePolicy):
    def __init__(self, strategy: str, offload_ratio: float):
        self.strategy = strategy
        self.offload_ratio = float(offload_ratio)
        self.name = f"{strategy}(offload={self.offload_ratio:.2f})"

    def _select_handover(self, env: LEOSatelliteEnv, user, visible_sats: Sequence[object]) -> int:
        if not visible_sats:
            return 0
        if self.strategy == "random":
            return int(env.rng.integers(0, len(visible_sats) + 1))
        if self.strategy == "min_distance":
            target_idx = int(np.argmin([sat.distance_km for sat in visible_sats]))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if user.serving_satellite == visible_sats[target_idx].sat_id:
            return 0
        return target_idx + 1

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        actions = np.zeros((env.num_users, 2), dtype=np.float32)
        for user_id, user in enumerate(env.user_manager.users):
            visible_sats = env._get_visible_satellites(user)
            actions[user_id, 0] = self._select_handover(env, user, visible_sats)
            actions[user_id, 1] = self.offload_ratio
        return actions


class FullLocalPolicy(BasePolicy):
    name = "full_local"

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        return np.zeros((env.num_users, 2), dtype=np.float32)


class JointGreedyPolicy(BasePolicy):
    def __init__(self, objective: str, offload_grid: Sequence[float]):
        self.objective = objective
        self.offload_grid = [float(np.clip(value, 0.0, 1.0)) for value in offload_grid]
        self.name = "joint_greedy"
        self._planned_queue_increments: Dict[int, int] = {}
        self._planned_cycle_increments: Dict[int, float] = {}

    def begin_episode(self, env: LEOSatelliteEnv) -> None:
        del env
        self._planned_queue_increments = defaultdict(int)
        self._planned_cycle_increments = defaultdict(float)

    def _effective_target_for_action(self, env: LEOSatelliteEnv, user, handover_action: int, visible_sats: Sequence[object]):
        if handover_action > 0 and len(visible_sats) >= handover_action:
            return visible_sats[handover_action - 1]
        current_vis = current_visibility(env, user)
        if current_vis is not None:
            return current_vis
        if visible_sats:
            return max(visible_sats, key=lambda sat: sat.elevation_deg)
        return None

    def _expected_handover_value(
        self,
        env: LEOSatelliteEnv,
        user,
        target_vis,
        predicted_queue_len: int,
    ) -> float:
        if target_vis is None or target_vis.sat_id == user.serving_satellite:
            return 0.0

        target_server = env.mec_manager.get_server(target_vis.sat_id)
        if target_server is None:
            return -float(env.config.reward_failed_handover_penalty)

        old_server = env.mec_manager.get_server(user.serving_satellite) if user.serving_satellite >= 0 else None
        migration_load = 0
        if old_server is not None:
            migration_load = sum(1 for task in old_server.task_queue if task.get("user_id") == user.user_id)

        queue_ratio = predicted_queue_len / max(target_server.config.max_queue_size, 1)
        utilization = np.clip(
            target_server.utilization + 0.15 * self._planned_queue_increments[target_vis.sat_id],
            0.0,
            1.0,
        )
        snr_db = env.channel.compute_snr_db(target_vis.distance_km, target_vis.elevation_deg)
        success_prob = env._compute_handover_success_probability(
            elevation_deg=target_vis.elevation_deg,
            rvt_seconds=target_vis.rvt_seconds,
            snr_db=snr_db,
            utilization=utilization,
            queue_ratio=queue_ratio,
            migration_load=migration_load,
        )

        elevation_score = np.clip(target_vis.elevation_deg / 90.0, 0.0, 1.0)
        rvt_score = np.clip(
            target_vis.rvt_seconds / max(env.config.rvt_threshold_sec, 1.0),
            0.0,
            1.0,
        )
        handover_gain = env.config.reward_handover_weight * (0.5 * elevation_score + 0.5 * rvt_score)
        migration_penalty = 0.05 * migration_load
        delay_penalty = min(env.config.handover_delay_sec / 2.0, 1.0)

        load_bonus = 0.0
        if user.serving_satellite >= 0:
            current_server = env.mec_manager.get_server(user.serving_satellite)
            if current_server is not None:
                current_queue_ratio = current_server.queue_length / max(current_server.config.max_queue_size, 1)
                load_bonus = env.config.reward_load_balance_weight * np.clip(current_queue_ratio - queue_ratio, -1.0, 1.0)

        success_value = handover_gain - env.config.reward_handover_weight * (delay_penalty + migration_penalty) + load_bonus
        failure_value = -float(env.config.reward_failed_handover_penalty)
        return float(success_prob * success_value + (1.0 - success_prob) * failure_value)

    def _task_score(self, env: LEOSatelliteEnv, total_delay: float, total_energy: float, max_delay: float) -> float:
        if self.objective == "delay_only":
            penalty = getattr(env, "delay_violation_penalty", 5.0) if total_delay > max_delay else 0.0
            return -float(total_delay + penalty)
        if self.objective == "energy_only":
            penalty = getattr(env, "delay_violation_penalty", 1.0) if total_delay > max_delay else 0.0
            return -float(total_energy + penalty)
        reward_value, _ = env._compute_task_reward(total_delay, total_energy, max_delay)
        return float(reward_value)

    def _estimate_task_value(
        self,
        env: LEOSatelliteEnv,
        user,
        task,
        target_vis,
        offload_ratio: float,
        predicted_queue_len: int,
    ) -> tuple[float, float]:
        if task is None:
            return 0.0, 0.0

        target_sat_id = target_vis.sat_id if target_vis is not None else user.serving_satellite
        server = env.mec_manager.get_server(target_sat_id) if target_sat_id is not None and target_sat_id >= 0 else None

        local_ratio = 1.0 - offload_ratio
        local_cycles = local_ratio * task.computation
        local_delay = env.offload_calc.compute_local_delay(local_cycles) if local_cycles > 0 else 0.0
        local_energy = env.offload_calc.compute_local_energy(local_cycles) if local_cycles > 0 else 0.0

        if offload_ratio <= 0.0 or server is None or target_vis is None:
            reward_value = self._task_score(env, local_delay, local_energy, task.max_delay)
            return float(reward_value), 0.0

        if predicted_queue_len >= server.config.max_queue_size:
            fallback_delay = env.offload_calc.compute_local_delay(task.computation)
            fallback_energy = env.offload_calc.compute_local_energy(task.computation)
            reward_value = self._task_score(env, fallback_delay, fallback_energy, task.max_delay)
            if self.objective == "multi_objective":
                reward_value -= float(env.config.reward_queue_full_penalty)
            return float(reward_value), 0.0

        offload_cycles = offload_ratio * task.computation
        offload_bits = offload_ratio * task.data_size
        upload_delay, download_delay = env.offload_calc.compute_transmission_delay(
            offload_bits,
            target_vis.distance_km,
            target_vis.elevation_deg,
        )
        upload_energy = env.offload_calc.compute_transmission_energy(
            offload_bits,
            target_vis.distance_km,
            target_vis.elevation_deg,
        )

        existing_cycles = active_queue_remaining_cycles(server)
        planned_cycles = self._planned_cycle_increments[target_vis.sat_id]
        capacity_hz = max(server.total_capacity_ghz * 1e9, 1e-6)
        queue_wait = (existing_cycles + planned_cycles) / capacity_hz
        satellite_compute_delay = offload_cycles / capacity_hz
        offload_total_delay = upload_delay + queue_wait + satellite_compute_delay + download_delay
        total_delay = max(local_delay, offload_total_delay)
        total_energy = local_energy + upload_energy

        reward_value = self._task_score(env, total_delay, total_energy, task.max_delay)
        if self.objective == "multi_objective":
            reward_value += env.config.reward_enqueue_bonus * max(
                1.0 - (predicted_queue_len / max(server.config.max_queue_size, 1)),
                0.0,
            )
        return float(reward_value), float(offload_cycles)

    def _score_candidate(
        self,
        env: LEOSatelliteEnv,
        user,
        visible_sats: Sequence[object],
        handover_action: int,
        offload_ratio: float,
    ) -> tuple[float, float]:
        task = env.user_tasks.get(user.user_id)
        target_vis = self._effective_target_for_action(env, user, handover_action, visible_sats)
        target_sat_id = target_vis.sat_id if target_vis is not None else user.serving_satellite
        target_server = env.mec_manager.get_server(target_sat_id) if target_sat_id is not None and target_sat_id >= 0 else None
        predicted_queue_len = (
            target_server.queue_length + self._planned_queue_increments[target_sat_id]
            if target_server is not None and target_sat_id is not None and target_sat_id >= 0
            else 0
        )

        score = 0.0
        if user.state == UserState.BLOCKED and target_vis is None:
            score -= float(env.config.reward_blocked_penalty)

        score += self._expected_handover_value(env, user, target_vis, predicted_queue_len)
        task_value, extra_cycles = self._estimate_task_value(
            env,
            user,
            task,
            target_vis,
            offload_ratio,
            predicted_queue_len,
        )
        score += task_value
        return score, extra_cycles

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        self._planned_queue_increments = defaultdict(int)
        self._planned_cycle_increments = defaultdict(float)
        actions = np.zeros((env.num_users, 2), dtype=np.float32)

        order = list(range(env.num_users))
        order.sort(
            key=lambda user_id: (
                env.user_tasks[user_id].max_delay if env.user_tasks.get(user_id) is not None else float("inf"),
                user_id,
            )
        )

        for user_id in order:
            user = env.user_manager.users[user_id]
            visible_sats = env._get_visible_satellites(user)
            candidate_actions = [0] + [index + 1 for index in range(len(visible_sats))]

            best_score = -float("inf")
            best_handover = 0
            best_offload = 0.0
            best_cycles = 0.0

            for handover_action in candidate_actions:
                ratios = self.offload_grid if env.user_tasks.get(user_id) is not None else [0.0]
                for offload_ratio in ratios:
                    score, extra_cycles = self._score_candidate(
                        env,
                        user,
                        visible_sats,
                        handover_action,
                        offload_ratio,
                    )
                    if score > best_score:
                        best_score = score
                        best_handover = handover_action
                        best_offload = offload_ratio
                        best_cycles = extra_cycles

            actions[user_id, 0] = best_handover
            actions[user_id, 1] = best_offload

            task = env.user_tasks.get(user_id)
            target_vis = self._effective_target_for_action(env, user, int(best_handover), visible_sats)
            target_sat_id = target_vis.sat_id if target_vis is not None else user.serving_satellite
            target_server = env.mec_manager.get_server(target_sat_id) if target_sat_id is not None and target_sat_id >= 0 else None
            if task is not None and best_offload > 0.0 and target_server is not None:
                self._planned_queue_increments[target_sat_id] += 1
                self._planned_cycle_increments[target_sat_id] += best_cycles

        return actions


class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def dqn_action_mask(env: LEOSatelliteEnv, offload_bins: Sequence[float]) -> np.ndarray:
    action_dim = (env.max_visible_sats + 1) * len(offload_bins)
    masks = np.zeros((env.num_users, action_dim), dtype=bool)
    for user_id, user in enumerate(env.user_manager.users):
        visible_sats = env._get_visible_satellites(user)
        valid_handover_count = min(len(visible_sats), env.max_visible_sats)
        for handover_action in range(valid_handover_count + 1):
            start = handover_action * len(offload_bins)
            masks[user_id, start:start + len(offload_bins)] = True
    return masks


def dqn_indices_to_env_actions(indices: np.ndarray, offload_bins: Sequence[float]) -> np.ndarray:
    bins = np.asarray(offload_bins, dtype=np.float32)
    num_bins = len(bins)
    actions = np.zeros((len(indices), 2), dtype=np.float32)
    actions[:, 0] = indices // num_bins
    actions[:, 1] = bins[indices % num_bins]
    return actions


def select_dqn_indices(
    q_net: DQNNetwork,
    observations: np.ndarray,
    masks: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    if rng.random() < epsilon:
        sampled = []
        for mask in masks:
            valid = np.flatnonzero(mask)
            sampled.append(int(rng.choice(valid)) if len(valid) else 0)
        return np.asarray(sampled, dtype=np.int64)

    with torch.no_grad():
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
        q_values = q_net(obs_tensor).detach().cpu().numpy()
    q_values = np.where(masks, q_values, -np.inf)
    return np.argmax(q_values, axis=1).astype(np.int64)


def evaluate_dqn_policy(
    q_net: DQNNetwork,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    offload_bins: Sequence[float],
    device: torch.device,
) -> Dict:
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    rewards: List[float] = []
    summaries: List[Dict] = []
    rng = np.random.default_rng(seed)

    q_net.eval()
    for episode_idx in range(episodes):
        observations, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        while not done:
            masks = dqn_action_mask(env, offload_bins)
            action_indices = select_dqn_indices(q_net, observations, masks, 0.0, rng, device)
            env_actions = dqn_indices_to_env_actions(action_indices, offload_bins)
            observations, reward, terminated, truncated, _ = env.step(env_actions)
            episode_reward += float(reward)
            done = terminated or truncated
        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())

    return summarize_results("dqn", rewards, summaries, is_system=False)


def train_and_evaluate_dqn_baseline(
    config: Dict,
    objective: str,
    output_dir: Path,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    total_timesteps: int,
    offload_bins: Sequence[float],
    device_name: str,
) -> Dict:
    device = torch.device(resolve_device(device_name))
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    obs_dim = int(env.user_obs_dim)
    clean_bins = [float(np.clip(value, 0.0, 1.0)) for value in offload_bins]
    clean_bins = sorted(set(clean_bins))
    if not clean_bins:
        clean_bins = [0.0]
    action_dim = (env.max_visible_sats + 1) * len(clean_bins)

    q_net = DQNNetwork(obs_dim, action_dim).to(device)
    target_net = DQNNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    replay = deque(maxlen=50_000)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 128
    gamma = 0.99
    warmup = min(1_000, max(64, total_timesteps // 20))
    target_update_interval = 500
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay_steps = max(total_timesteps * 0.7, 1)
    training_records: List[Dict] = []
    recent_episode_rewards: deque = deque(maxlen=10)
    recent_losses: deque = deque(maxlen=100)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    observations, _ = env.reset(seed=seed)
    q_net.train()
    for step_idx in range(max(int(total_timesteps), 0)):
        progress = min(step_idx / epsilon_decay_steps, 1.0)
        epsilon = epsilon_start + progress * (epsilon_final - epsilon_start)
        masks = dqn_action_mask(env, clean_bins)
        action_indices = select_dqn_indices(q_net, observations, masks, epsilon, rng, device)
        env_actions = dqn_indices_to_env_actions(action_indices, clean_bins)
        next_observations, reward, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)
        next_masks = dqn_action_mask(env, clean_bins) if not done else np.zeros_like(masks)
        episode_reward += float(reward)
        episode_length += 1

        for user_id in range(env.num_users):
            replay.append(
                (
                    observations[user_id].astype(np.float32, copy=True),
                    int(action_indices[user_id]),
                    float(reward),
                    next_observations[user_id].astype(np.float32, copy=True),
                    bool(done),
                    next_masks[user_id].astype(bool, copy=True),
                )
            )

        observations = next_observations
        if done:
            observations, _ = env.reset(seed=seed + step_idx + 1)

        if len(replay) >= max(batch_size, warmup):
            batch = random.sample(replay, batch_size)
            obs_b = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
            action_b = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            reward_b = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            next_obs_b = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=device)
            done_b = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
            next_mask_b = torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool, device=device)

            q_selected = q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_obs_b).masked_fill(~next_mask_b, -1e9).max(dim=1).values
                target = reward_b + gamma * (1.0 - done_b) * next_q
            loss = F.smooth_l1_loss(q_selected, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            recent_losses.append(float(loss.detach().cpu().item()))

        if (step_idx + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            episode_count += 1
            recent_episode_rewards.append(episode_reward)
            training_records.append(
                {
                    "update": episode_count,
                    "total_steps": step_idx + 1,
                    "episodes": episode_count,
                    "mean_reward": episode_reward,
                    "recent_mean_reward": float(np.mean(recent_episode_rewards)),
                    "mean_length": float(episode_length),
                    "epsilon": float(epsilon),
                    "loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
                }
            )
            episode_reward = 0.0
            episode_length = 0

    if episode_length > 0:
        episode_count += 1
        recent_episode_rewards.append(episode_reward)
        training_records.append(
            {
                "update": episode_count,
                "total_steps": int(total_timesteps),
                "episodes": episode_count,
                "mean_reward": episode_reward,
                "recent_mean_reward": float(np.mean(recent_episode_rewards)),
                "mean_length": float(episode_length),
                "epsilon": float(epsilon_final),
                "loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
                "partial_episode": True,
            }
        )

    result = evaluate_dqn_policy(
        q_net=q_net,
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        offload_bins=clean_bins,
        device=device,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["offload_grid"] = clean_bins
    checkpoint_dir = output_dir / "learned_baselines" / "dqn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "training_history.json"
    checkpoint_path = checkpoint_dir / "dqn_model.pt"
    torch.save(
        {
            "q_state_dict": q_net.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "offload_bins": clean_bins,
            "trained_timesteps": int(total_timesteps),
            "training_history": str(history_path),
        },
        checkpoint_path,
    )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "method": "dqn",
                    "objective": objective,
                    "total_timesteps": int(total_timesteps),
                    "seed": int(seed),
                    "max_steps": int(max_steps) if max_steps is not None else None,
                    "offload_grid": clean_bins,
                },
                "training": training_records,
                "evaluation": result.get("episode_metrics", []),
                "summary": {
                    "mean_reward": float(result.get("mean_reward", 0.0)),
                    "std_reward": float(result.get("std_reward", 0.0)),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    result["checkpoint"] = str(checkpoint_path)
    result["training_history"] = str(history_path)
    return result


class MADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, handover_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.handover_head = nn.Linear(in_dim, int(handover_dim))
        self.offload_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        handover_logits = self.handover_head(features)
        offload = torch.sigmoid(self.offload_head(features)).squeeze(-1)
        return handover_logits, offload


class MADDPGCritic(nn.Module):
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_feature_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
    ):
        super().__init__()
        input_dim = int(num_agents) * (int(obs_dim) + int(action_feature_dim))
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        if observations.dim() != 3 or action_features.dim() != 3:
            raise ValueError("MADDPG critic expects batched tensors shaped (batch, agents, dim).")
        joint_input = torch.cat(
            [
                observations.reshape(observations.shape[0], -1),
                action_features.reshape(action_features.shape[0], -1),
            ],
            dim=-1,
        )
        return self.net(joint_input).squeeze(-1)


def maddpg_action_mask(env: LEOSatelliteEnv) -> np.ndarray:
    masks = np.zeros((env.num_users, env.max_visible_sats + 1), dtype=bool)
    masks[:, 0] = True
    for user_id, user in enumerate(env.user_manager.users):
        visible_sats = env._get_visible_satellites(user)
        valid_count = min(len(visible_sats), env.max_visible_sats)
        if valid_count > 0:
            masks[user_id, 1:valid_count + 1] = True
    return masks


def safe_mask_tensor(masks: torch.Tensor) -> torch.Tensor:
    safe_masks = masks.bool().clone()
    flat_masks = safe_masks.reshape(-1, safe_masks.shape[-1])
    empty_rows = ~flat_masks.any(dim=-1)
    if torch.any(empty_rows):
        flat_masks[empty_rows, 0] = True
    return safe_masks


def maddpg_actor_action_features(
    actor: MADDPGActor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    straight_through: bool = True,
) -> torch.Tensor:
    batch_size, num_agents, obs_dim = observations.shape
    handover_dim = masks.shape[-1]
    flat_obs = observations.reshape(batch_size * num_agents, obs_dim)
    flat_masks = safe_mask_tensor(masks).reshape(batch_size * num_agents, handover_dim)
    logits, offload = actor(flat_obs)
    probs = torch.softmax(logits.masked_fill(~flat_masks, -1e9), dim=-1)
    if straight_through:
        hard_indices = torch.argmax(probs, dim=-1)
        hard_handover = F.one_hot(hard_indices, num_classes=handover_dim).to(probs.dtype)
        handover_features = hard_handover + probs - probs.detach()
    else:
        handover_features = probs
    features = torch.cat([handover_features, offload.unsqueeze(-1)], dim=-1)
    return features.reshape(batch_size, num_agents, handover_dim + 1)


def maddpg_one_hot_action_features(
    handover_actions: np.ndarray,
    offload_ratios: np.ndarray,
    handover_dim: int,
) -> np.ndarray:
    handover = np.clip(np.asarray(handover_actions, dtype=np.int64), 0, handover_dim - 1)
    offload = np.clip(np.asarray(offload_ratios, dtype=np.float32), 0.0, 1.0)
    features = np.zeros((len(handover), handover_dim + 1), dtype=np.float32)
    features[np.arange(len(handover)), handover] = 1.0
    features[:, -1] = offload
    return features


def random_maddpg_actions(
    masks: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    handover_actions = []
    for mask in masks:
        valid = np.flatnonzero(mask)
        handover_actions.append(int(rng.choice(valid)) if len(valid) else 0)
    handover = np.asarray(handover_actions, dtype=np.int64)
    offload = rng.random(len(handover), dtype=np.float32)
    features = maddpg_one_hot_action_features(handover, offload, masks.shape[1])
    env_actions = np.column_stack([handover, offload]).astype(np.float32)
    return env_actions, features, handover


def select_maddpg_env_actions(
    actor: MADDPGActor,
    observations: np.ndarray,
    masks: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
        logits, offload = actor(obs_tensor)
        logits_np = logits.detach().cpu().numpy()
        offload_np = offload.detach().cpu().numpy()

    if noise_std > 0.0:
        logits_np = logits_np + rng.normal(0.0, noise_std, size=logits_np.shape)
        offload_np = offload_np + rng.normal(0.0, noise_std, size=offload_np.shape)

    logits_np = np.where(masks, logits_np, -np.inf)
    handover = np.argmax(logits_np, axis=1).astype(np.int64)
    offload_np = np.clip(offload_np, 0.0, 1.0).astype(np.float32)
    features = maddpg_one_hot_action_features(handover, offload_np, masks.shape[1])
    env_actions = np.column_stack([handover, offload_np]).astype(np.float32)
    actor.train(was_training)
    return env_actions, features, handover


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def scalar_reward_value(reward) -> float:
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        return float(np.mean(list(reward.values()))) if reward else 0.0
    return float(np.mean(np.asarray(reward, dtype=float)))


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a module state for in-memory best-checkpoint tracking."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def evaluate_maddpg_policy(
    actor: MADDPGActor,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    device: torch.device,
) -> Dict:
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    rewards: List[float] = []
    summaries: List[Dict] = []
    rng = np.random.default_rng(seed)

    actor.eval()
    for episode_idx in range(episodes):
        observations, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        while not done:
            masks = maddpg_action_mask(env)
            env_actions, _, _ = select_maddpg_env_actions(
                actor=actor,
                observations=observations,
                masks=masks,
                noise_std=0.0,
                rng=rng,
                device=device,
            )
            observations, reward, terminated, truncated, _ = env.step(env_actions)
            episode_reward += scalar_reward_value(reward)
            done = terminated or truncated
        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())

    return summarize_results("maddpg", rewards, summaries, is_system=False)


def train_and_evaluate_maddpg_baseline(
    config: Dict,
    objective: str,
    output_dir: Path,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    total_timesteps: int,
    device_name: str,
) -> Dict:
    device = torch.device(resolve_device(device_name))
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    obs_dim = int(env.user_obs_dim)
    num_agents = int(env.num_users)
    handover_dim = int(env.max_visible_sats + 1)
    action_feature_dim = handover_dim + 1

    actor = MADDPGActor(obs_dim, handover_dim).to(device)
    target_actor = MADDPGActor(obs_dim, handover_dim).to(device)
    critic = MADDPGCritic(num_agents, obs_dim, action_feature_dim).to(device)
    target_critic = MADDPGCritic(num_agents, obs_dim, action_feature_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    replay = deque(maxlen=50_000)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 128
    gamma = 0.99
    tau = 0.01
    warmup = min(1_000, max(64, total_timesteps // 20))
    noise_start = 0.35
    noise_final = 0.05
    noise_decay_steps = max(total_timesteps * 0.7, 1)
    training_records: List[Dict] = []
    evaluation_records: List[Dict] = []
    recent_episode_rewards: deque = deque(maxlen=10)
    recent_actor_losses: deque = deque(maxlen=100)
    recent_critic_losses: deque = deque(maxlen=100)
    eval_interval_episodes = 10
    train_eval_episodes = min(3, max(1, int(episodes)))
    best_eval_reward = -float("inf")
    best_actor_state = clone_state_dict(actor)
    best_critic_state = clone_state_dict(critic)
    best_target_actor_state = clone_state_dict(target_actor)
    best_target_critic_state = clone_state_dict(target_critic)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    observations, _ = env.reset(seed=seed)
    actor.train()
    critic.train()
    for step_idx in range(max(int(total_timesteps), 0)):
        progress = min(step_idx / noise_decay_steps, 1.0)
        noise_std = noise_start + progress * (noise_final - noise_start)
        masks = maddpg_action_mask(env)
        if step_idx < warmup:
            env_actions, action_features, _ = random_maddpg_actions(masks, rng)
        else:
            env_actions, action_features, _ = select_maddpg_env_actions(
                actor=actor,
                observations=observations,
                masks=masks,
                noise_std=float(noise_std),
                rng=rng,
                device=device,
            )

        next_observations, reward, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)
        reward_value = scalar_reward_value(reward)
        next_masks = maddpg_action_mask(env) if not done else np.zeros_like(masks)
        replay.append(
            (
                observations.astype(np.float32, copy=True),
                action_features.astype(np.float32, copy=True),
                float(reward_value),
                next_observations.astype(np.float32, copy=True),
                bool(done),
                masks.astype(bool, copy=True),
                next_masks.astype(bool, copy=True),
            )
        )

        observations = next_observations
        episode_reward += reward_value
        episode_length += 1

        if len(replay) >= max(batch_size, warmup):
            batch = random.sample(replay, batch_size)
            obs_b = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
            action_b = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
            reward_b = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            next_obs_b = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=device)
            done_b = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
            mask_b = torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool, device=device)
            next_mask_b = torch.tensor(np.stack([item[6] for item in batch]), dtype=torch.bool, device=device)

            with torch.no_grad():
                next_action_b = maddpg_actor_action_features(target_actor, next_obs_b, next_mask_b)
                target_q = target_critic(next_obs_b, next_action_b)
                target = reward_b + gamma * (1.0 - done_b) * target_q

            q_values = critic(obs_b, action_b)
            critic_loss = F.mse_loss(q_values, target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            for param in critic.parameters():
                param.requires_grad_(False)
            actor_action_b = maddpg_actor_action_features(actor, obs_b, mask_b)
            actor_loss = -critic(obs_b, actor_action_b).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            for param in critic.parameters():
                param.requires_grad_(True)

            soft_update(actor, target_actor, tau)
            soft_update(critic, target_critic, tau)
            recent_actor_losses.append(float(actor_loss.detach().cpu().item()))
            recent_critic_losses.append(float(critic_loss.detach().cpu().item()))

        if done:
            episode_count += 1
            recent_episode_rewards.append(episode_reward)
            record = {
                "update": episode_count,
                "total_steps": step_idx + 1,
                "episodes": episode_count,
                "mean_reward": episode_reward,
                "recent_mean_reward": float(np.mean(recent_episode_rewards)),
                "mean_length": float(episode_length),
                "exploration_noise": float(noise_std),
                "actor_loss": float(np.mean(recent_actor_losses)) if recent_actor_losses else 0.0,
                "critic_loss": float(np.mean(recent_critic_losses)) if recent_critic_losses else 0.0,
            }
            if episode_count % eval_interval_episodes == 0:
                eval_result = evaluate_maddpg_policy(
                    actor=actor,
                    objective=objective,
                    config=config,
                    episodes=train_eval_episodes,
                    seed=seed + 10_000,
                    max_steps=max_steps,
                    device=device,
                )
                record["eval_mean_reward"] = float(eval_result.get("mean_reward", 0.0))
                record["eval_std_reward"] = float(eval_result.get("std_reward", 0.0))
                eval_record = {
                    "update": episode_count,
                    "total_steps": step_idx + 1,
                    "eval_mean_reward": record["eval_mean_reward"],
                    "eval_std_reward": record["eval_std_reward"],
                    "eval_episodes": train_eval_episodes,
                }
                evaluation_records.append(eval_record)
                if record["eval_mean_reward"] > best_eval_reward:
                    best_eval_reward = record["eval_mean_reward"]
                    best_actor_state = clone_state_dict(actor)
                    best_critic_state = clone_state_dict(critic)
                    best_target_actor_state = clone_state_dict(target_actor)
                    best_target_critic_state = clone_state_dict(target_critic)
                actor.train()
                critic.train()
            training_records.append(record)
            observations, _ = env.reset(seed=seed + step_idx + 1)
            episode_reward = 0.0
            episode_length = 0

    if episode_length > 0:
        episode_count += 1
        recent_episode_rewards.append(episode_reward)
        training_records.append(
            {
                "update": episode_count,
                "total_steps": int(total_timesteps),
                "episodes": episode_count,
                "mean_reward": episode_reward,
                "recent_mean_reward": float(np.mean(recent_episode_rewards)),
                "mean_length": float(episode_length),
                "exploration_noise": float(noise_final),
                "actor_loss": float(np.mean(recent_actor_losses)) if recent_actor_losses else 0.0,
                "critic_loss": float(np.mean(recent_critic_losses)) if recent_critic_losses else 0.0,
                "partial_episode": True,
            }
        )

    final_eval = evaluate_maddpg_policy(
        actor=actor,
        objective=objective,
        config=config,
        episodes=train_eval_episodes,
        seed=seed + 10_000,
        max_steps=max_steps,
        device=device,
    )
    final_eval_record = {
        "update": episode_count,
        "total_steps": int(total_timesteps),
        "eval_mean_reward": float(final_eval.get("mean_reward", 0.0)),
        "eval_std_reward": float(final_eval.get("std_reward", 0.0)),
        "eval_episodes": train_eval_episodes,
        "final_training_eval": True,
    }
    if training_records and (
        int(training_records[-1].get("total_steps", -1)) == int(total_timesteps)
        and "eval_mean_reward" not in training_records[-1]
    ):
        training_records[-1]["eval_mean_reward"] = final_eval_record["eval_mean_reward"]
        training_records[-1]["eval_std_reward"] = final_eval_record["eval_std_reward"]
    if not evaluation_records or int(evaluation_records[-1].get("total_steps", -1)) != int(total_timesteps):
        evaluation_records.append(final_eval_record)
    if final_eval_record["eval_mean_reward"] > best_eval_reward:
        best_eval_reward = final_eval_record["eval_mean_reward"]
        best_actor_state = clone_state_dict(actor)
        best_critic_state = clone_state_dict(critic)
        best_target_actor_state = clone_state_dict(target_actor)
        best_target_critic_state = clone_state_dict(target_critic)

    final_actor_state = clone_state_dict(actor)
    final_critic_state = clone_state_dict(critic)
    final_target_actor_state = clone_state_dict(target_actor)
    final_target_critic_state = clone_state_dict(target_critic)
    actor.load_state_dict(best_actor_state)
    critic.load_state_dict(best_critic_state)
    target_actor.load_state_dict(best_target_actor_state)
    target_critic.load_state_dict(best_target_critic_state)

    result = evaluate_maddpg_policy(
        actor=actor,
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["best_training_eval_reward"] = float(best_eval_reward)
    checkpoint_dir = output_dir / "learned_baselines" / "maddpg"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "training_history.json"
    checkpoint_path = checkpoint_dir / "maddpg_model.pt"
    final_checkpoint_path = checkpoint_dir / "maddpg_final_model.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "target_actor_state_dict": target_actor.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "obs_dim": obs_dim,
            "num_agents": num_agents,
            "handover_dim": handover_dim,
            "action_feature_dim": action_feature_dim,
            "trained_timesteps": int(total_timesteps),
            "best_training_eval_reward": float(best_eval_reward),
            "training_history": str(history_path),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "actor_state_dict": final_actor_state,
            "critic_state_dict": final_critic_state,
            "target_actor_state_dict": final_target_actor_state,
            "target_critic_state_dict": final_target_critic_state,
            "obs_dim": obs_dim,
            "num_agents": num_agents,
            "handover_dim": handover_dim,
            "action_feature_dim": action_feature_dim,
            "trained_timesteps": int(total_timesteps),
            "training_history": str(history_path),
        },
        final_checkpoint_path,
    )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "method": "maddpg",
                    "objective": objective,
                    "total_timesteps": int(total_timesteps),
                    "seed": int(seed),
                    "max_steps": int(max_steps) if max_steps is not None else None,
                    "actor_lr": 5e-4,
                    "critic_lr": 1e-3,
                    "gamma": gamma,
                    "tau": tau,
                    "noise_start": noise_start,
                    "noise_final": noise_final,
                    "train_eval_interval_episodes": eval_interval_episodes,
                    "train_eval_episodes": train_eval_episodes,
                    "best_training_eval_reward": float(best_eval_reward),
                    "parameter_sharing": True,
                    "discrete_actor_update": "straight_through_one_hot",
                },
                "training": training_records,
                "training_evaluation": evaluation_records,
                "evaluation": result.get("episode_metrics", []),
                "summary": {
                    "mean_reward": float(result.get("mean_reward", 0.0)),
                    "std_reward": float(result.get("std_reward", 0.0)),
                    "best_training_eval_reward": float(best_eval_reward),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    result["checkpoint"] = str(checkpoint_path)
    result["training_history"] = str(history_path)
    return result


def pdqn_action_mask(env: LEOSatelliteEnv) -> np.ndarray:
    masks = np.zeros((env.num_users, env.max_visible_sats + 1), dtype=bool)
    masks[:, 0] = True
    for user_id, user in enumerate(env.user_manager.users):
        visible_sats = env._get_visible_satellites(user)
        valid_count = min(len(visible_sats), env.max_visible_sats)
        if valid_count > 0:
            masks[user_id, 1:valid_count + 1] = True
    return masks


def evaluate_pdqn_policy(
    algorithm: PDQNAlgorithm,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
) -> Dict:
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    rewards: List[float] = []
    summaries: List[Dict] = []

    for episode_idx in range(episodes):
        observations, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        while not done:
            masks = pdqn_action_mask(env)
            env_actions, _, _ = algorithm.act(observations, masks, epsilon=0.0)
            observations, reward, terminated, truncated, _ = env.step(env_actions)
            episode_reward += scalar_reward_value(reward)
            done = terminated or truncated
        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())

    return summarize_results("pdqn", rewards, summaries, is_system=False)


def train_and_evaluate_pdqn_baseline(
    config: Dict,
    objective: str,
    output_dir: Path,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    total_timesteps: int,
    device_name: str,
) -> Dict:
    device = resolve_device(device_name)
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    obs_dim = int(env.user_obs_dim)
    num_agents = int(env.num_users)
    max_candidates = int(env.max_visible_sats)
    handover_dim = max_candidates + 1
    algo = PDQNAlgorithm(
        PDQNConfig(
            num_agents=num_agents,
            obs_dim=obs_dim,
            max_candidates=max_candidates,
            batch_size=128,
            warmup_steps=1_000,
            replay_size=50_000,
            epsilon_decay_steps=max(int(total_timesteps), 1),
            device=device,
        )
    )
    replay = MultiAgentReplayBuffer(
        capacity=algo.config.replay_size,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_feature_dim=handover_dim + 1,
        mask_dim=handover_dim,
        device=device,
    )
    training_records: List[Dict] = []
    recent_episode_rewards: deque = deque(maxlen=10)
    recent_q_losses: deque = deque(maxlen=100)
    recent_param_losses: deque = deque(maxlen=100)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    observations, _ = env.reset(seed=seed)
    for step_idx in range(max(int(total_timesteps), 0)):
        masks = pdqn_action_mask(env)
        if step_idx < algo.config.warmup_steps:
            env_actions, action_features, _ = algo.random_actions(masks)
        else:
            env_actions, action_features, _ = algo.act(observations, masks)

        next_observations, reward, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)
        reward_value = scalar_reward_value(reward)
        next_masks = pdqn_action_mask(env) if not done else np.zeros_like(masks)
        replay.add(observations, action_features, reward_value, next_observations, done, masks, next_masks)

        if len(replay) >= max(algo.config.batch_size, algo.config.warmup_steps):
            stats = algo.update(replay)
            if stats:
                recent_q_losses.append(float(stats.get("q_loss", 0.0)))
                recent_param_losses.append(float(stats.get("param_loss", 0.0)))

        observations = next_observations
        episode_reward += reward_value
        episode_length += 1

        if done:
            episode_count += 1
            recent_episode_rewards.append(episode_reward)
            env_stats = env.get_stats_summary()
            summary = summarize_env_stats(env_stats)
            training_records.append(
                {
                    "update": episode_count,
                    "total_steps": step_idx + 1,
                    "episodes": episode_count,
                    "mean_reward": episode_reward,
                    "recent_mean_reward": float(np.mean(recent_episode_rewards)),
                    "mean_length": float(episode_length),
                    "q_loss": float(np.mean(recent_q_losses)) if recent_q_losses else 0.0,
                    "param_loss": float(np.mean(recent_param_losses)) if recent_param_losses else 0.0,
                    "epsilon": algo.current_epsilon(),
                    "avg_delay": summary.get("avg_delay", 0.0),
                    "effective_latency_score": summary.get("effective_latency_score", 0.0),
                    "service_continuity_rate": summary.get("service_continuity_rate", 0.0),
                    "task_completion_rate": summary.get("task_completion_rate", 0.0),
                    "task_success_rate": summary.get("task_success_rate", 0.0),
                    "avg_load_balance_score": summary.get("avg_load_balance_score", 0.0),
                    "total_energy": env_stats.get("total_energy", 0.0),
                }
            )
            observations, _ = env.reset(seed=seed + step_idx + 1)
            episode_reward = 0.0
            episode_length = 0

    save_dir = output_dir / "learned_baselines" / "pdqn"
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "training_history.json"
    checkpoint_path = save_dir / "pdqn_model.pt"
    algo.save(checkpoint_path)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    **config,
                    "algorithm": "pdqn",
                    "total_timesteps": int(total_timesteps),
                    "device": device,
                },
                "training": training_records,
                "evaluation": [],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    result = evaluate_pdqn_policy(
        algorithm=algo,
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["checkpoint"] = str(checkpoint_path)
    result["training_history"] = str(history_path)
    return result


def build_policy(name: str, objective: str, fixed_offload: float, joint_offload_grid: Sequence[float]) -> BasePolicy:
    if name in {"random", "min_distance"}:
        return SimpleHeuristicPolicy(name, fixed_offload)
    if name == "full_local":
        return FullLocalPolicy()
    if name == "joint_greedy":
        return JointGreedyPolicy(objective=objective, offload_grid=joint_offload_grid)
    raise ValueError(f"Unknown baseline: {name}")


def evaluate_policy(
    policy: BasePolicy,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
) -> Dict:
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    rewards: List[float] = []
    summaries: List[Dict] = []

    for episode_idx in range(episodes):
        policy.begin_episode(env)
        env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0

        while not done:
            actions = policy.select_actions(env)
            _, reward, terminated, truncated, _ = env.step(
                actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += float(reward)
            done = terminated or truncated

        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())

    extra = {}
    if isinstance(policy, SimpleHeuristicPolicy):
        extra["selected_offload"] = policy.offload_ratio
    return summarize_results(policy.name, rewards, summaries, extra=extra, is_system=False)


def evaluate_simple_heuristic_with_offload_search(
    strategy: str,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    offload_grid: Sequence[float],
    selection_metric_name: str,
) -> Dict:
    candidates = []
    for offload_ratio in offload_grid:
        policy = SimpleHeuristicPolicy(strategy, offload_ratio)
        result = evaluate_policy(
            policy=policy,
            objective=objective,
            config=config,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
        )
        candidates.append(result)

    best_index = choose_best_index(
        [selection_score(candidate, selection_metric_name) for candidate in candidates],
        higher_is_better=True,
    )
    best = dict(candidates[best_index])
    best["method"] = strategy
    best["display_name"] = pretty_method_name(strategy, is_system=False)
    best["method_variant"] = best["method"] + f"(best_offload={best.get('selected_offload', 0.0):.2f})"
    return best


def trainer_class_for_objective(objective: str):
    if objective == "delay_only":
        if DelayOnlyTrainer is None:
            raise ModuleNotFoundError("delay_only objective requires scripts/train_delay_only.py")
        return DelayOnlyTrainer
    if objective == "energy_only":
        if EnergyOnlyTrainer is None:
            raise ModuleNotFoundError("energy_only objective requires scripts/train_energy_only.py")
        return EnergyOnlyTrainer
    return HANMAPPOTrainer


class NoHANTrainerMixin:
    """Mixin for MAPPO ablations that bypass the HAN graph encoder."""

    def _init_environment(self):
        super()._init_environment()
        self.han_out_dim = self.raw_obs_dim
        self.obs_dim = self.raw_obs_dim
        self.global_state_dim = self.num_agents * self.obs_dim
        self.logger.info("  - HAN ablation: using raw environment observations")
        self.logger.info(f"  - No-HAN observation dim: {self.obs_dim}")

    def _init_han_encoder(self):
        self.han_encoder = nn.Identity().to(self.device)

    def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = self.env._get_observation().astype(np.float32, copy=False)
        available_actions = np.zeros((self.num_agents, self.max_candidates + 1), dtype=np.float32)
        available_actions[:, 0] = 1.0
        for uid, user in enumerate(self.env.user_manager.users):
            visible_sats = self.env._get_visible_satellites(user)
            valid_count = min(len(visible_sats), self.max_candidates)
            if valid_count > 0:
                available_actions[uid, 1:valid_count + 1] = 1.0

        satellite_embeddings = np.zeros((self.env.num_satellites, self.han_out_dim), dtype=np.float32)
        return observations, satellite_embeddings, available_actions


class NoHANMAPPOTrainer(NoHANTrainerMixin, HANMAPPOTrainer):
    """Default multi-objective MAPPO ablation without HAN embeddings."""


def no_han_trainer_class_for_objective(objective: str):
    base_cls = trainer_class_for_objective(objective)
    if base_cls is HANMAPPOTrainer:
        return NoHANMAPPOTrainer

    class ObjectiveNoHANTrainer(NoHANTrainerMixin, base_cls):
        """Objective-specific MAPPO ablation without HAN embeddings."""

    ObjectiveNoHANTrainer.__name__ = f"NoHAN{base_cls.__name__}"
    return ObjectiveNoHANTrainer


def train_config_from_dict(
    config_data: Dict,
    device: str,
    max_steps: Optional[int],
    episodes: int,
    total_timesteps: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    save_path: Optional[Path] = None,
    exp_name: Optional[str] = None,
    load_path: Optional[Path] = None,
) -> TrainConfig:
    config = TrainConfig()
    for key, value in config_data.items():
        setattr(config, key, value)
    config.device = resolve_device(device)
    config.eval_episodes = episodes
    if max_steps is not None:
        config.max_steps = int(max_steps)
    if total_timesteps is not None:
        config.total_timesteps = int(total_timesteps)
    if early_stop_patience is not None:
        config.early_stop_patience = int(early_stop_patience)
    if save_path is not None:
        config.save_path = str(save_path)
    if exp_name:
        config.exp_name = exp_name
    if load_path is not None:
        config.load_path = str(load_path)
    return config


def run_system_training(
    config_data: Dict,
    objective: str,
    system_run_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    exp_name: Optional[str],
    resume_checkpoint: Optional[Path],
) -> tuple[Path, Optional[Path], Optional[Path], Dict]:
    trainer_cls = trainer_class_for_objective(objective)
    system_run_dir.mkdir(parents=True, exist_ok=True)
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        save_path=system_run_dir,
        exp_name=exp_name,
        load_path=resume_checkpoint,
    )
    trainer = trainer_cls(config)
    if resume_checkpoint is not None:
        trainer.load_checkpoint(str(resume_checkpoint))
    trainer.train()
    checkpoint = system_run_dir / "best_model.pt"
    if not checkpoint.exists():
        checkpoint = system_run_dir / "final_model.pt"
    history_path = system_run_dir / "training_history.json"
    return (
        system_run_dir,
        checkpoint if checkpoint.exists() else None,
        history_path if history_path.exists() else None,
        asdict(config),
    )


def evaluate_system_checkpoint(
    checkpoint: Path,
    config_data: Dict,
    objective: str,
    episodes: int,
    device: str,
    max_steps: Optional[int],
) -> Dict:
    trainer_cls = trainer_class_for_objective(objective)
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        save_path=checkpoint.parent,
        load_path=checkpoint,
    )
    trainer = trainer_cls(config)
    checkpoint_payload = torch_load_trusted_checkpoint(checkpoint, map_location=trainer.device)
    trainer.total_steps = checkpoint_payload.get("total_steps", trainer.total_steps)
    trainer.episodes = checkpoint_payload.get("episodes", trainer.episodes)
    trainer.best_reward = checkpoint_payload.get("best_reward", trainer.best_reward)
    if "best_model_metric" in checkpoint_payload:
        trainer.config.best_model_metric = checkpoint_payload["best_model_metric"]
    trainer.mappo.actor.load_state_dict(checkpoint_payload["actor_state_dict"])
    trainer.mappo.critic.load_state_dict(checkpoint_payload["critic_state_dict"])
    if "han_state_dict" in checkpoint_payload:
        trainer.han_encoder.load_state_dict(checkpoint_payload["han_state_dict"])

    rewards: List[float] = []
    summaries: List[Dict] = []

    for episode_idx in range(episodes):
        trainer.env.reset(seed=int(config.seed) + episode_idx if config.seed is not None else None)
        observations, satellite_embeddings, available_actions = trainer._encode_graph_state()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                actions, _, _ = trainer.mappo.act(
                    observations,
                    available_actions,
                    satellite_embeddings=satellite_embeddings,
                    deterministic=True,
                )

            env_actions = trainer._process_actions(actions)
            _, reward, terminated, truncated, _ = trainer.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += float(reward)
            done = terminated or truncated

            if not done:
                observations, satellite_embeddings, available_actions = trainer._encode_graph_state()

        rewards.append(episode_reward)
        summaries.append(trainer.env.get_stats_summary())

    method_name = str(config_data.get("exp_name", checkpoint.parent.name or "system"))
    return summarize_results(method_name, rewards, summaries, is_system=True)


def evaluate_mappo_checkpoint_with_trainer(
    checkpoint: Path,
    config_data: Dict,
    episodes: int,
    device: str,
    max_steps: Optional[int],
    trainer_cls,
    method_name: str,
    is_system: bool = False,
) -> Dict:
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        save_path=checkpoint.parent,
        load_path=checkpoint,
    )
    trainer = trainer_cls(config)
    checkpoint_payload = torch_load_trusted_checkpoint(checkpoint, map_location=trainer.device)
    trainer.total_steps = checkpoint_payload.get("total_steps", trainer.total_steps)
    trainer.episodes = checkpoint_payload.get("episodes", trainer.episodes)
    trainer.best_reward = checkpoint_payload.get("best_reward", trainer.best_reward)
    if "best_model_metric" in checkpoint_payload:
        trainer.config.best_model_metric = checkpoint_payload["best_model_metric"]
    trainer.mappo.actor.load_state_dict(checkpoint_payload["actor_state_dict"])
    trainer.mappo.critic.load_state_dict(checkpoint_payload["critic_state_dict"])
    if "han_state_dict" in checkpoint_payload and hasattr(trainer.han_encoder, "load_state_dict"):
        trainer.han_encoder.load_state_dict(checkpoint_payload["han_state_dict"], strict=False)

    rewards: List[float] = []
    summaries: List[Dict] = []

    for episode_idx in range(episodes):
        trainer.env.reset(seed=int(config.seed) + episode_idx if config.seed is not None else None)
        observations, satellite_embeddings, available_actions = trainer._encode_graph_state()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                actions, _, _ = trainer.mappo.act(
                    observations,
                    available_actions,
                    satellite_embeddings=satellite_embeddings,
                    deterministic=True,
                )

            env_actions = trainer._process_actions(actions)
            _, reward, terminated, truncated, _ = trainer.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += float(reward)
            done = terminated or truncated

            if not done:
                observations, satellite_embeddings, available_actions = trainer._encode_graph_state()

        rewards.append(episode_reward)
        summaries.append(trainer.env.get_stats_summary())

    return summarize_results(method_name, rewards, summaries, is_system=is_system)


def train_and_evaluate_no_han_mappo(
    config_data: Dict,
    objective: str,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
) -> Dict:
    save_dir = output_dir / "learned_baselines" / "mappo_no_han"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer_cls = no_han_trainer_class_for_objective(objective)
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        save_path=save_dir,
        exp_name="mappo_no_han",
    )
    trainer = trainer_cls(config)
    trainer.train()
    checkpoint = save_dir / "best_model.pt"
    if not checkpoint.exists():
        checkpoint = save_dir / "final_model.pt"
    result = evaluate_mappo_checkpoint_with_trainer(
        checkpoint=checkpoint,
        config_data=asdict(config),
        episodes=episodes,
        device=resolve_device(device),
        max_steps=max_steps,
        trainer_cls=trainer_cls,
        method_name="mappo_no_han",
        is_system=False,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["checkpoint"] = str(checkpoint)
    history_path = save_dir / "training_history.json"
    if history_path.exists():
        result["training_history"] = str(history_path)
    return result


def evaluate_han_offpolicy_checkpoint(
    checkpoint: Path,
    config_data: Dict,
    episodes: int,
    device: str,
    max_steps: Optional[int],
    trainer_cls,
    method_name: str,
) -> Dict:
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        save_path=checkpoint.parent,
        load_path=checkpoint,
    )
    trainer = trainer_cls(config)
    trainer.load_checkpoint(str(checkpoint))
    rewards: List[float] = []
    summaries: List[Dict] = []

    for episode_idx in range(episodes):
        observations, _, masks = trainer._reset_encoded_env(seed=int(config.seed) + episode_idx)
        done = False
        episode_reward = 0.0
        while not done:
            env_actions, _, _ = trainer._select_eval_action(observations, masks)
            _, reward, terminated, truncated, _ = trainer.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += trainer._scalar_reward(reward)
            done = terminated or truncated
            if not done:
                observations, _, masks = trainer._encode_graph_state()
        rewards.append(episode_reward)
        summaries.append(trainer.env.get_stats_summary())

    return summarize_results(method_name, rewards, summaries, is_system=False)


def train_and_evaluate_han_offpolicy_baseline(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    trainer_cls,
    method_name: str,
    algorithm_name: str,
) -> Dict:
    save_dir = output_dir / "learned_baselines" / method_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config = train_config_from_dict(
        config_data,
        device=device,
        max_steps=max_steps,
        episodes=episodes,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        save_path=save_dir,
        exp_name=method_name,
    )
    config.algorithm = algorithm_name
    trainer = trainer_cls(config)
    trainer.train()
    checkpoint = save_dir / "best_model.pt"
    if not checkpoint.exists():
        checkpoint = save_dir / "final_model.pt"
    result = evaluate_han_offpolicy_checkpoint(
        checkpoint=checkpoint,
        config_data=asdict(config),
        episodes=episodes,
        device=resolve_device(device),
        max_steps=max_steps,
        trainer_cls=trainer_cls,
        method_name=method_name,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["checkpoint"] = str(checkpoint)
    history_path = save_dir / "training_history.json"
    if history_path.exists():
        result["training_history"] = str(history_path)
    return result


def train_and_evaluate_han_maddpg_baseline(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
) -> Dict:
    return train_and_evaluate_han_offpolicy_baseline(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        trainer_cls=HANMADDPGTrainer,
        method_name="han_maddpg",
        algorithm_name="maddpg",
    )


def train_and_evaluate_han_pdqn_baseline(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
) -> Dict:
    return train_and_evaluate_han_offpolicy_baseline(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        trainer_cls=HANPDQNTrainer,
        method_name="han_pdqn",
        algorithm_name="pdqn",
    )


def extract_history_method(history_path: Path) -> tuple[Dict, Optional[Dict]]:
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config_data = dict(payload.get("config", {}))
    evaluation_records = payload.get("evaluation", [])
    if not evaluation_records:
        return config_data, None

    selection_metric_name = str(config_data.get("best_model_metric", DEFAULT_SELECTION_METRIC))
    best_record = max(
        evaluation_records,
        key=lambda record: compute_model_selection_score(record, selection_metric_name),
    )
    handover_success_rate = float(best_record.get("handover_success_rate", 0.0))
    service_continuity_rate = float(best_record.get("service_continuity_rate", 0.0))
    effective_latency_score = float(
        best_record.get(
            "effective_latency_score",
            compute_model_selection_score(best_record, "effective_latency_score"),
        )
    )
    result = {
        "method": str(config_data.get("exp_name", history_path.parent.name or "system")),
        "display_name": pretty_method_name(
            str(config_data.get("exp_name", history_path.parent.name or "system")),
            is_system=True,
        ),
        "episodes": int(config_data.get("eval_episodes", 0)),
        "is_system": True,
        "mean_reward": float(best_record.get("eval_mean_reward", 0.0)),
        "std_reward": float(best_record.get("eval_std_reward", 0.0)),
        "avg_delay": float(best_record.get("avg_delay", 0.0)),
        "total_energy": float(best_record.get("total_energy", 0.0)),
        "handover_success_rate": handover_success_rate,
        "handover_failure_rate": float(best_record.get("handover_failure_rate", max(0.0, 1.0 - handover_success_rate))),
        "forced_termination_rate": float(best_record.get("forced_termination_rate", max(0.0, 1.0 - service_continuity_rate))),
        "service_continuity_rate": service_continuity_rate,
        "service_availability_rate": float(best_record.get("service_availability_rate", service_continuity_rate)),
        "task_completion_rate": float(best_record.get("task_completion_rate", 0.0)),
        "task_success_rate": float(best_record.get("task_success_rate", best_record.get("task_completion_rate", 0.0))),
        "task_failure_rate": float(best_record.get("task_failure_rate", best_record.get("deadline_violation_rate", 0.0))),
        "task_settlement_rate": float(best_record.get("task_settlement_rate", best_record.get("task_resolution_rate", 0.0))),
        "task_resolution_rate": float(best_record.get("task_resolution_rate", 0.0)),
        "pending_task_rate": float(best_record.get("pending_task_rate", 0.0)),
        "avg_load_balance_score": float(best_record.get("avg_load_balance_score", 0.0)),
        "resolved_tasks": float(best_record.get("resolved_tasks", 0.0)),
        "pending_tasks": float(best_record.get("pending_tasks", 0.0)),
        "total_tasks": float(best_record.get("total_tasks", 0.0)),
        "completed_tasks": float(best_record.get("completed_tasks", 0.0)),
        "deadline_violations": float(best_record.get("deadline_violations", 0.0)),
        "deadline_violation_rate": compute_deadline_violation_rate(best_record),
        "effective_latency_score": effective_latency_score,
        "episode_metrics": [],
        "training_history": str(history_path),
        "source": f"training_history_best_{selection_metric_name}",
    }
    return config_data, result


def save_results_json(output_dir: Path, payload: Dict) -> Path:
    path = output_dir / "comparison_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def save_results_csv(output_dir: Path, methods: Sequence[Dict]) -> Path:
    path = output_dir / "comparison_summary.csv"
    fieldnames = [
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
        "effective_latency_score",
        "avg_load_balance_score",
        "energy_per_resolved_task",
        *REWARD_BREAKDOWN_KEYS,
        "selection_metric",
        "selection_score",
        "primary_metric_win_count",
        "primary_metric_wins_text",
        "selected_offload",
        "method_variant",
        "training_history",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {key: method.get(key, "") for key in fieldnames}
            writer.writerow(row)
    return path


def save_episode_metrics_csv(output_dir: Path, methods: Sequence[Dict]) -> Optional[Path]:
    rows = []
    for method in order_methods(methods):
        for record in method.get("episode_metrics", []):
            row = {
                "method": method.get("method", ""),
                "display_name": method.get("display_name", method.get("method", "")),
                "episode": int(record.get("episode", 0)),
                "reward": float(record.get("reward", 0.0)),
                "avg_delay": float(record.get("avg_delay", 0.0)),
                "total_energy": float(record.get("total_energy", 0.0)),
                "handover_success_rate": float(record.get("handover_success_rate", 0.0)),
                "handover_failure_rate": float(record.get("handover_failure_rate", 0.0)),
                "forced_termination_rate": float(record.get("forced_termination_rate", 0.0)),
                "service_continuity_rate": float(record.get("service_continuity_rate", 0.0)),
                "service_availability_rate": float(record.get("service_availability_rate", 0.0)),
                "task_completion_rate": float(record.get("task_completion_rate", 0.0)),
                "task_success_rate": float(record.get("task_success_rate", record.get("task_completion_rate", 0.0))),
                "task_failure_rate": float(record.get("task_failure_rate", record.get("deadline_violation_rate", 0.0))),
                "task_settlement_rate": float(record.get("task_settlement_rate", record.get("task_resolution_rate", 0.0))),
                "task_resolution_rate": float(record.get("task_resolution_rate", 0.0)),
                "pending_task_rate": float(record.get("pending_task_rate", 0.0)),
                "effective_latency_score": float(record.get("effective_latency_score", 0.0)),
                "avg_load_balance_score": float(record.get("avg_load_balance_score", 0.0)),
                "deadline_violation_rate": float(record.get("deadline_violation_rate", 0.0)),
                "resolved_tasks": float(record.get("resolved_tasks", 0.0)),
                "completed_tasks": float(record.get("completed_tasks", 0.0)),
                "deadline_violations": float(record.get("deadline_violations", 0.0)),
                **{key: float(record.get(key, 0.0)) for key in REWARD_BREAKDOWN_KEYS},
                "selected_offload": method.get("selected_offload", ""),
            }
            rows.append(row)

    if not rows:
        return None

    path = output_dir / "episode_metrics.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def metric_scale(metric_key: str) -> float:
    return 100.0 if metric_key.endswith("_rate") else 1.0


def metric_display_value(value: float, metric_key: str) -> str:
    if metric_key.endswith("_rate"):
        return f"{value:.1f}%"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def metric_episode_samples(method: Dict, metric_key: str) -> np.ndarray:
    records = method.get("episode_metrics", [])
    if not records:
        return np.array([], dtype=float)
    scale = metric_scale(metric_key)
    return np.array([float(record.get(metric_key, 0.0)) * scale for record in records], dtype=float)


def paper_metric_scale(metric_key: str) -> float:
    if metric_key == "avg_delay":
        return 1000.0
    return metric_scale(metric_key)


def paper_metric_value(method: Dict, metric_key: str) -> float:
    return float(method.get(metric_key, 0.0)) * paper_metric_scale(metric_key)


def paper_metric_samples(method: Dict, metric_key: str) -> np.ndarray:
    records = method.get("episode_metrics", [])
    if not records:
        return np.array([], dtype=float)
    scale = paper_metric_scale(metric_key)
    return np.array([float(record.get(metric_key, 0.0)) * scale for record in records], dtype=float)


def method_tick_label(method: Dict) -> str:
    label = str(method.get("display_name", method.get("method", "")))
    replacements = {
        "HAN+MAPPO": "HAN+\nMAPPO",
        "Joint Greedy": "Joint\nGreedy",
        "Min-Distance": "Min-\nDistance",
        "Full-Local": "Full-\nLocal",
        "MAPPO (no HAN)": "MAPPO\n(no HAN)",
        "MADDPG": "MADDPG",
    }
    return replacements.get(label, label)


def style_axes_frame(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.6, color="#BDBDBD")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.9)


def is_pareto_efficient(x_values: Sequence[float], y_values: Sequence[float]) -> np.ndarray:
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    efficient = np.ones(len(xs), dtype=bool)
    for index in range(len(xs)):
        if not efficient[index]:
            continue
        dominated = (
            (xs <= xs[index]) &
            (ys <= ys[index]) &
            ((xs < xs[index]) | (ys < ys[index]))
        )
        dominated[index] = False
        if np.any(dominated):
            efficient[index] = False
    return efficient


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=PAPER_COLORS["dark"],
    )


def draw_metric_bar_panel(ax, methods: Sequence[Dict], metric_key: str, title: str, ylabel: str, compact: bool = False) -> None:
    ordered = order_methods(methods)
    styles = build_method_styles(ordered)
    labels = [method_tick_label(method) for method in ordered]
    values = [paper_metric_value(method, metric_key) for method in ordered]
    errors = []
    for method in ordered:
        samples = paper_metric_samples(method, metric_key)
        errors.append(float(np.std(samples)) if len(samples) > 1 else 0.0)

    positions = np.arange(len(ordered), dtype=float)
    colors = [styles[str(method.get("method", ""))].get("color", PAPER_COLORS["muted"]) for method in ordered]
    bars = ax.bar(
        positions,
        values,
        yerr=errors if any(error > 0 for error in errors) else None,
        width=0.72,
        color=colors,
        edgecolor=PAPER_COLORS["dark"],
        linewidth=1.0,
        alpha=0.92,
        error_kw={"elinewidth": 1.1, "capsize": 2.8, "capthick": 1.1},
    )

    best_index = choose_best_index(values, HIGHER_IS_BETTER.get(metric_key, True))
    for index, (bar, value, method) in enumerate(zip(bars, values, ordered)):
        style = styles[str(method.get("method", ""))]
        bar.set_hatch(style.get("hatch", ""))
        if method.get("is_system") or str(method.get("method", "")) == "joint_greedy":
            bar.set_linewidth(1.35)
        if index == best_index:
            bar.set_edgecolor(PAPER_COLORS["dark"])
            bar.set_linewidth(2.1)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(max(values) * 0.02, 0.015),
            metric_display_value(value, metric_key),
            va="bottom",
            ha="center",
            fontsize=10 if compact else 10.5,
            rotation=90 if compact else 0,
        )

    ax.set_xticks(positions, labels=labels)
    ax.tick_params(axis="x", rotation=25 if compact else 28)
    direction = "Higher is better" if HIGHER_IS_BETTER.get(metric_key, True) else "Lower is better"
    ax.set_title(f"{title} ({direction})")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#BDBDBD")
    if values:
        ax.set_ylim(0.0, max(values) * 1.20 + 1e-9)
    style_axes_frame(ax)


def draw_reward_curve_panel(
    ax,
    history_path: Optional[Path],
    methods: Sequence[Dict],
    window: int,
    compact: bool = False,
    output_dir: Optional[Path] = None,
) -> bool:
    steps, rewards, evaluation = load_training_curve_from_path(history_path)
    if len(steps) == 0:
        return False

    mean_reward, _ = reward_smooth(rewards, window=max(window, 3))
    system_color = SYSTEM_STYLE["color"]
    draw_raw_reward_shadow(
        ax,
        steps,
        rewards,
        mean_reward,
        system_color,
        alpha=0.20,
    )
    ax.plot(
        steps,
        mean_reward,
        color=system_color,
        linewidth=3.0,
        zorder=3,
        label="HAN+MAPPO",
    )

    if evaluation:
        eval_steps = np.array([record.get("total_steps", 0) for record in evaluation], dtype=float)
        eval_rewards = np.array([record.get("eval_mean_reward", 0.0) for record in evaluation], dtype=float)
        ax.scatter(
            eval_steps,
            eval_rewards,
            s=30 if compact else 36,
            facecolors="white",
            edgecolors=system_color,
            linewidths=1.0,
            zorder=4,
        )

    baseline_methods = [method for method in order_methods(methods) if not method.get("is_system")]
    baseline_styles = build_method_styles(baseline_methods)
    for method in baseline_methods:
        style = baseline_styles[str(method.get("method", ""))]
        baseline_history_path = method_training_history_path(method, output_dir=output_dir)
        baseline_steps, baseline_rewards, _ = load_training_curve_from_path(baseline_history_path)
        if len(baseline_steps) > 0:
            baseline_mean, _ = reward_smooth(baseline_rewards, window=max(window, 3))
            draw_raw_reward_shadow(
                ax,
                baseline_steps,
                baseline_rewards,
                baseline_mean,
                style["color"],
                alpha=0.13 if compact else 0.15,
            )
            ax.plot(
                baseline_steps,
                baseline_mean,
                color=style["color"],
                linestyle="-",
                linewidth=2.2 if compact else 2.4,
                alpha=0.97,
                zorder=3,
                label=method.get("display_name", method.get("method", "")),
            )
            continue

        mean_value = float(method.get("mean_reward", 0.0))
        std_value = float(method.get("std_reward", 0.0))
        if std_value > 0.0:
            ax.axhspan(
                mean_value - std_value,
                mean_value + std_value,
                color=style["color"],
                alpha=0.05,
                linewidth=0,
            )
        ax.axhline(
            y=mean_value,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            alpha=0.95,
            label=method.get("display_name", method.get("method", "")),
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Convergence vs. Baseline Levels")
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(
        loc="lower right" if compact else "best",
        fontsize=10.5 if compact else 11.5,
        ncol=1 if compact else 2,
    )
    style_axes_frame(ax)
    return True


def plot_method_comparison(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    if not methods:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=220)
    for axis, (metric_key, title, ylabel) in zip(axes.flatten(), CORE_BAR_METRICS):
        draw_metric_bar_panel(axis, methods, metric_key=metric_key, title=title, ylabel=ylabel)

    fig.suptitle("HAN+MAPPO vs. Heuristic Baselines: Core Metrics", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir / "method_comparison.pdf")


def draw_training_metric_step_panel(
    ax,
    history_path: Optional[Path],
    metric_key: str,
    title: str,
    ylabel: str,
    scale: float,
    window: int,
    color: str,
) -> bool:
    steps, values = load_training_metric_curve_from_path(history_path, metric_key)
    if len(steps) == 0:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No step data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=PAPER_COLORS["muted"],
        )
        ax.set_title(title)
        return False

    scaled_values = values.astype(float) * float(scale)
    smoothed_values, effective_window = reward_smooth(scaled_values, window=max(window, 3))
    draw_raw_metric_shadow(ax, steps, scaled_values, smoothed_values, color=color, alpha=0.20)
    ax.plot(
        steps,
        smoothed_values,
        color=color,
        linewidth=2.0,
        label=f"Smoothed (w={effective_window})",
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    style_axes_frame(ax)
    return True


def load_training_metric_per_task_curve_from_path(history_path: Optional[Path], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    steps, values = load_training_metric_curve_from_path(history_path, metric_key)
    if history_path is None or not history_path.exists() or len(steps) == 0:
        return steps, values
    history = load_training_history(history_path)
    training = history.get("training", []) if isinstance(history, dict) else []
    totals = np.array([max(float(record.get("total_tasks", 0.0)), 1.0) for record in training if "total_steps" in record and metric_key in record], dtype=float)
    if len(totals) != len(values):
        return steps, values
    return steps, values / totals


def draw_training_metric_per_task_step_panel(
    ax,
    history_path: Optional[Path],
    metric_key: str,
    title: str,
    ylabel: str,
    scale: float,
    window: int,
    color: str,
) -> bool:
    steps, values = load_training_metric_per_task_curve_from_path(history_path, metric_key)
    if len(steps) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No step data", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=PAPER_COLORS["muted"])
        ax.set_title(title)
        return False

    scaled_values = values.astype(float) * float(scale)
    smoothed_values, effective_window = reward_smooth(scaled_values, window=max(window, 3))
    draw_raw_metric_shadow(ax, steps, scaled_values, smoothed_values, color=color, alpha=0.20)
    ax.plot(steps, smoothed_values, color=color, linewidth=2.0, label=f"Smoothed (w={effective_window})", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    style_axes_frame(ax)
    return True


def plot_step_metric_group(
    history_path: Optional[Path],
    output_dir: Path,
    window: int,
    metric_specs: Sequence[tuple[str, str, str, float]],
    output_name: str,
    title: str,
) -> Optional[Path]:
    if history_path is None or not history_path.exists():
        return None

    ncols = 2
    nrows = int(np.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows), dpi=220)
    axes = np.asarray(axes).reshape(-1)
    colors = [
        PAPER_COLORS["primary"],
        PAPER_COLORS["secondary"],
        PAPER_COLORS["info"],
        PAPER_COLORS["success"],
        PAPER_COLORS["warning"],
        "#5DA5DA",
        "#60BD68",
        "#F17CB0",
        "#B2912F",
        "#B276B2",
    ]

    has_any_data = False
    for index, (axis, spec) in enumerate(zip(axes, metric_specs)):
        metric_key, metric_title, ylabel, scale = spec
        has_data = draw_training_metric_step_panel(
            axis,
            history_path,
            metric_key,
            metric_title,
            ylabel,
            scale,
            window,
            colors[index % len(colors)],
        )
        has_any_data = has_any_data or has_data

    for axis in axes[len(metric_specs):]:
        axis.axis("off")

    if not has_any_data:
        plt.close(fig)
        return None

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_figure(fig, output_dir / output_name)


def plot_reward_component_per_task_curves(history_path: Optional[Path], output_dir: Path, window: int) -> Optional[Path]:
    if history_path is None or not history_path.exists():
        return None

    metric_specs = [spec for spec in REWARD_COMPONENT_STEP_METRICS if spec[0] != "mean_reward"]
    ncols = 2
    nrows = int(np.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows), dpi=220)
    axes = np.asarray(axes).reshape(-1)
    colors = [
        PAPER_COLORS["primary"],
        PAPER_COLORS["secondary"],
        PAPER_COLORS["info"],
        PAPER_COLORS["success"],
        PAPER_COLORS["warning"],
        "#5DA5DA",
        "#60BD68",
        "#F17CB0",
        "#B2912F",
        "#B276B2",
    ]

    has_any_data = False
    for index, (axis, spec) in enumerate(zip(axes, metric_specs)):
        metric_key, metric_title, _ylabel, scale = spec
        has_data = draw_training_metric_per_task_step_panel(
            axis,
            history_path,
            metric_key,
            f"{metric_title} per Task",
            "Reward Term / Task",
            scale,
            window,
            colors[index % len(colors)],
        )
        has_any_data = has_any_data or has_data

    for axis in axes[len(metric_specs):]:
        axis.axis("off")

    if not has_any_data:
        plt.close(fig)
        return None

    fig.suptitle("Reward Components per Task vs. Steps", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_figure(fig, output_dir / "reward_components_per_task_vs_steps.pdf")


def plot_step_metric_curves(history_path: Optional[Path], output_dir: Path, window: int) -> List[Path]:
    paths: List[Path] = []
    qos_path = plot_step_metric_group(
        history_path,
        output_dir,
        window,
        TRAINING_QOS_STEP_METRICS,
        "training_qos_metrics_vs_steps.pdf",
        "Training QoS Metrics vs. Steps",
    )
    if qos_path is not None:
        paths.append(qos_path)

    reward_component_path = plot_step_metric_group(
        history_path,
        output_dir,
        window,
        REWARD_COMPONENT_STEP_METRICS,
        "reward_components_vs_steps.pdf",
        "Reward Components vs. Steps",
    )
    if reward_component_path is not None:
        paths.append(reward_component_path)
    reward_component_per_task_path = plot_reward_component_per_task_curves(
        history_path,
        output_dir,
        window,
    )
    if reward_component_per_task_path is not None:
        paths.append(reward_component_per_task_path)
    return paths


def methods_with_episode_metrics(methods: Sequence[Dict]) -> List[Dict]:
    return [method for method in order_methods(methods) if method.get("episode_metrics")]


def plot_episode_metric_curve(
    methods: Sequence[Dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Optional[Path]:
    plottable = methods_with_episode_metrics(methods)
    if not plottable:
        return None

    styles = build_method_styles(plottable)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=220)

    for method in plottable:
        records = method.get("episode_metrics", [])
        episodes = [int(record.get("episode", idx + 1)) for idx, record in enumerate(records)]
        values = [float(record.get(metric_key, 0.0)) * metric_scale(metric_key) for record in records]
        style = styles[str(method.get("method", ""))]
        ax.plot(
            episodes,
            values,
            label=method.get("display_name", method.get("method", "")),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=0.95,
        )

    direction = "Higher is better" if HIGHER_IS_BETTER.get(metric_key, True) else "Lower is better"
    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({direction})")
    style_axes_frame(ax)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_additional_metric_curves(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    plottable = methods_with_episode_metrics(methods)
    if not plottable:
        return None

    styles = build_method_styles(plottable)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    axes = axes.flatten()

    for axis, (metric_key, title) in zip(axes, ADDITIONAL_EPISODE_METRICS):
        for method in plottable:
            records = method.get("episode_metrics", [])
            episodes = [int(record.get("episode", idx + 1)) for idx, record in enumerate(records)]
            values = [float(record.get(metric_key, 0.0)) * metric_scale(metric_key) for record in records]
            style = styles[str(method.get("method", ""))]
            axis.plot(
                episodes,
                values,
                label=method.get("display_name", method.get("method", "")),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=max(style["markersize"] - 0.5, 4.0),
                alpha=0.95,
            )

        direction = "Higher is better" if HIGHER_IS_BETTER.get(metric_key, True) else "Lower is better"
        axis.set_title(f"{title} ({direction})")
        axis.set_xlabel("Evaluation Episode")
        axis.set_ylabel("Rate (%)" if metric_key.endswith("_rate") else "Score")
        style_axes_frame(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir / "additional_metrics_episode_comparison.pdf")


def plot_delay_energy_tradeoff(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None

    styles = build_method_styles(ordered)
    x_values = np.array([paper_metric_value(method, "avg_delay") for method in ordered], dtype=float)
    y_values = np.array([float(method.get("energy_per_resolved_task", 0.0)) for method in ordered], dtype=float)
    if len(x_values) == 0 or len(y_values) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    pareto_mask = is_pareto_efficient(x_values, y_values)
    pareto_points = sorted(
        [(x, y) for x, y, keep in zip(x_values, y_values, pareto_mask) if keep],
        key=lambda item: item[0],
    )
    if len(pareto_points) >= 2:
        ax.plot(
            [point[0] for point in pareto_points],
            [point[1] for point in pareto_points],
            color=PAPER_COLORS["muted"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label="Pareto frontier",
        )

    for method, x_value, y_value, is_pareto in zip(ordered, x_values, y_values, pareto_mask):
        style = styles[str(method.get("method", ""))]
        display_name = str(method.get("display_name", method.get("method", "")))
        scatter_size = float(style.get("scatter_size", 160))
        if method.get("is_system"):
            scatter_size = max(scatter_size, 280.0)
        ax.scatter(
            x_value,
            y_value,
            s=scatter_size,
            marker=style.get("marker", "o"),
            color=style.get("color", PAPER_COLORS["muted"]),
            edgecolors=PAPER_COLORS["dark"],
            linewidths=1.1 if is_pareto else 0.9,
            alpha=0.96,
            zorder=4 if method.get("is_system") else 3,
        )
        dx, dy = SCATTER_LABEL_OFFSETS.get(display_name, (9, 9))
        label = display_name
        if method.get("is_system") and is_pareto:
            label = f"{display_name} (Pareto)"
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            color=PAPER_COLORS["dark"],
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "alpha": 0.88,
            },
        )

    ax.scatter(
        np.min(x_values),
        np.min(y_values),
        s=0,
        alpha=0.0,
    )
    ax.annotate(
        "Ideal region\n(lower delay, lower energy)",
        xy=(np.min(x_values), np.min(y_values)),
        xytext=(18, -4),
        textcoords="offset points",
        fontsize=11,
        color=PAPER_COLORS["secondary"],
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["secondary"], "lw": 1.1},
    )

    ax.set_xlabel("Average Delay (ms)")
    ax.set_ylabel("Energy per Resolved Task")
    ax.set_title("Delay-Energy Trade-off Across System and Heuristic Methods")
    style_axes_frame(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir / "delay_energy_tradeoff.pdf")


def plot_success_continuity_scatter(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None

    styles = build_method_styles(ordered)
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    for method in ordered:
        style = styles[str(method.get("method", ""))]
        x_value = float(method.get("task_success_rate", method.get("task_completion_rate", 0.0))) * 100.0
        y_value = float(method.get("service_continuity_rate", 0.0)) * 100.0
        load_balance = float(method.get("avg_load_balance_score", 0.0))
        effective = float(method.get("effective_latency_score", 0.0))
        size = 90.0 + 360.0 * np.clip(load_balance, 0.0, 1.0)
        if method.get("is_system"):
            size *= 1.20

        ax.scatter(
            x_value,
            y_value,
            s=size,
            marker=style.get("marker", "o"),
            color=style.get("color", PAPER_COLORS["muted"]),
            alpha=0.88,
            edgecolors=PAPER_COLORS["dark"],
            linewidths=1.0,
            zorder=4 if method.get("is_system") else 3,
        )
        label = method.get("display_name", method.get("method", ""))
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=10,
            color=PAPER_COLORS["dark"],
        )
        ax.text(
            x_value,
            y_value,
            f"{effective:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    ax.set_xlabel("Task Success Rate (%)")
    ax.set_ylabel("Service Continuity Rate (%)")
    ax.set_title("Success-Continuity Trade-off (marker size: load balance, label: effective latency)")
    ax.set_xlim(left=0.0, right=max(100.0, ax.get_xlim()[1]))
    ax.set_ylim(bottom=0.0, top=max(100.0, ax.get_ylim()[1]))
    style_axes_frame(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir / "success_continuity_tradeoff.pdf")


def normalized_metric_values(methods: Sequence[Dict], metric_key: str, higher_is_better: bool) -> np.ndarray:
    values = np.array([float(method.get(metric_key, 0.0)) for method in methods], dtype=float)
    if len(values) == 0:
        return values
    if metric_key.endswith("_rate") or metric_key in {"effective_latency_score", "avg_load_balance_score"}:
        bounded = np.clip(values, 0.0, 1.0)
        return bounded if higher_is_better else 1.0 - bounded

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        return np.ones_like(values)
    normalized = (values - min_value) / (max_value - min_value)
    return normalized if higher_is_better else 1.0 - normalized


def plot_performance_radar(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None

    styles = build_method_styles(ordered)
    labels = [label for _, label, _ in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    normalized_by_metric = [
        normalized_metric_values(ordered, metric_key, higher_is_better)
        for metric_key, _label, higher_is_better in RADAR_METRICS
    ]

    fig, ax = plt.subplots(figsize=(9.2, 9.2), dpi=220, subplot_kw={"polar": True})
    for method_index, method in enumerate(ordered):
        values = np.array(
            [metric_values[method_index] for metric_values in normalized_by_metric],
            dtype=float,
        )
        closed_values = np.concatenate([values, values[:1]])
        style = styles[str(method.get("method", ""))]
        linewidth = 2.3 if method.get("is_system") else 1.45
        alpha = 0.18 if method.get("is_system") else 0.08
        ax.plot(
            closed_angles,
            closed_values,
            color=style.get("color", PAPER_COLORS["muted"]),
            linewidth=linewidth,
            linestyle=style.get("linestyle", "-"),
            label=method.get("display_name", method.get("method", "")),
        )
        ax.fill(
            closed_angles,
            closed_values,
            color=style.get("color", PAPER_COLORS["muted"]),
            alpha=alpha,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Normalized Multi-Metric Performance Radar", fontsize=14, fontweight="bold", pad=22)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_dir / "performance_radar.pdf")


def plot_paper_dashboard(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
    window: int,
) -> Optional[Path]:
    if not methods:
        return None

    fig = plt.figure(figsize=(16, 10), dpi=220)
    grid = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.26, height_ratios=[1.15, 0.95])

    ax_reward = fig.add_subplot(grid[0, :])
    has_reward_curve = draw_reward_curve_panel(
        ax_reward,
        history_path,
        methods,
        window=window,
        compact=True,
        output_dir=output_dir,
    )
    if not has_reward_curve:
        ax_reward.axis("off")
        ax_reward.text(
            0.5,
            0.5,
            "No training history available.",
            transform=ax_reward.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    add_panel_label(ax_reward, "(a)")

    ax_delay = fig.add_subplot(grid[1, 0])
    draw_metric_bar_panel(
        ax_delay,
        methods,
        metric_key="avg_delay",
        title="Average Delay",
        ylabel="Average Delay (ms)",
        compact=True,
    )
    add_panel_label(ax_delay, "(b)")

    ax_task = fig.add_subplot(grid[1, 1])
    draw_metric_bar_panel(
        ax_task,
        methods,
        metric_key="task_completion_rate",
        title="Task Completion Rate",
        ylabel="Task Completion Rate (%)",
        compact=True,
    )
    add_panel_label(ax_task, "(c)")

    ax_energy = fig.add_subplot(grid[1, 2])
    draw_metric_bar_panel(
        ax_energy,
        methods,
        metric_key="avg_load_balance_score",
        title="Avg Load Balance Score",
        ylabel="Avg Load Balance Score",
        compact=True,
    )
    add_panel_label(ax_energy, "(d)")

    fig.suptitle("Publication-Style Baseline Comparison", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.92, wspace=0.26, hspace=0.32)
    return save_figure(fig, output_dir / "paper_baseline_dashboard.pdf")


def plot_training_curve_vs_baselines(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
    window: int,
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=220)
    has_curve = draw_reward_curve_panel(ax, history_path, methods, window=window, output_dir=output_dir)
    if not has_curve:
        plt.close(fig)
        return None
    fig.tight_layout()
    return save_figure(fig, output_dir / "reward_curve_vs_baselines.pdf")


def plot_reward_distribution(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    plottable = methods_with_episode_metrics(methods)
    if not plottable:
        return None

    ordered = order_methods(plottable)
    styles = build_method_styles(ordered)
    labels = [method.get("display_name", method.get("method", "")) for method in ordered]
    samples = [
        [float(record.get("reward", 0.0)) for record in method.get("episode_metrics", [])]
        for method in ordered
    ]
    if not any(samples):
        return None

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=220)
    box = ax.boxplot(
        samples,
        vert=False,
        patch_artist=True,
        tick_labels=labels,
        showfliers=False,
        medianprops={"color": PAPER_COLORS["dark"], "linewidth": 1.2},
        whiskerprops={"color": PAPER_COLORS["dark"], "linewidth": 1.0},
        capprops={"color": PAPER_COLORS["dark"], "linewidth": 1.0},
    )

    rng = np.random.default_rng(0)
    for position, method, patch in zip(range(1, len(ordered) + 1), ordered, box["boxes"]):
        style = styles[str(method.get("method", ""))]
        color = style["color"]
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(PAPER_COLORS["dark"])
        patch.set_linewidth(1.0)

        values = np.array([float(record.get("reward", 0.0)) for record in method.get("episode_metrics", [])], dtype=float)
        if len(values) == 0:
            continue

        jitter = rng.normal(0.0, 0.05, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), position, dtype=float) + jitter,
            s=26,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter(
            float(np.mean(values)),
            position,
            s=76 if method.get("is_system") else 52,
            marker="*" if method.get("is_system") else "D",
            color=color,
            edgecolors=PAPER_COLORS["dark"],
            linewidths=0.8,
            zorder=4,
        )

    ax.set_xlabel("Episode Reward")
    ax.set_title("Reward Distribution Across Evaluation Episodes")
    style_axes_frame(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    return save_figure(fig, output_dir / "reward_distribution.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HAN+MAPPO and compare it against the heuristic baselines in BASELINE_STRATEGIES."
    )
    parser.add_argument("--run-mode", type=str, default="train_compare",
                        choices=["train_compare", "compare_only"],
                        help="Whether to train the system first or only compare against an existing run.")
    parser.add_argument("--system-run-dir", type=str, default=str(DEFAULT_SYSTEM_RUN_DIR),
                        help=(
                            "Directory used for training outputs and/or existing system artifacts. "
                            "Fresh train_compare runs now protect existing artifacts by switching to a "
                            "timestamped sibling directory unless --resume-system or "
                            "--overwrite-system-run-dir is used."
                        ))
    parser.add_argument("--system-checkpoint", type=str, default=None,
                        help="Path to best_model.pt or final_model.pt.")
    parser.add_argument("--resume-system", action="store_true",
                        help="Resume training from an existing checkpoint in --system-run-dir or --system-checkpoint.")
    parser.add_argument("--overwrite-system-run-dir", action="store_true",
                        help="Allow a fresh train_compare run to write into an existing --system-run-dir that already contains training artifacts.")
    parser.add_argument("--exp-name", type=str, default=DEFAULT_SYSTEM_EXP_NAME,
                        help="Experiment name used when training from this unified entry script.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes for each method.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override episode length for training/evaluation/baselines.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Total system training steps. Default is 1,200,000 steps.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Early stopping patience for MAPPO training. Default 0 disables early stopping.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for baseline evaluation.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device used for training/evaluating the system method.")
    parser.add_argument("--best-model-metric", type=str, default=DEFAULT_SELECTION_METRIC,
                        choices=list(BEST_MODEL_METRIC_CHOICES),
                        help="Metric used to save best_model.pt during training and to read history best records.")
    parser.add_argument("--compare-ranking-metric", type=str, default=DEFAULT_SELECTION_METRIC,
                        choices=list(BEST_MODEL_METRIC_CHOICES),
                        help="Metric used to pick heuristic offload variants and sort methods in the report.")
    parser.add_argument("--objective", type=str, default="multi_objective",
                        choices=["multi_objective", "delay_only", "energy_only"],
                        help="Objective used when no system run is provided.")
    parser.add_argument("--num-users", type=int, default=10,
                        help="User count used when no system run is provided.")
    parser.add_argument("--baselines", type=str, nargs="+", default=["all"],
                        help="Baselines to evaluate. Use 'all' for the default suite.")
    parser.add_argument("--fixed-offload-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="Offload candidates for simple heuristic baselines.")
    parser.add_argument("--joint-offload-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Offload grid used by the joint greedy baseline.")
    parser.add_argument("--dqn-offload-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="Discrete offload grid used by the DQN baseline.")
    parser.add_argument("--dqn-timesteps", type=int, default=None,
                        help="Training steps for the DQN baseline. Defaults to --total-timesteps.")
    parser.add_argument("--maddpg-timesteps", type=int, default=None,
                        help="Training steps for the MADDPG baseline. Defaults to --total-timesteps.")
    parser.add_argument("--pdqn-timesteps", type=int, default=None,
                        help="Training steps for the PDQN baseline. Defaults to --total-timesteps.")
    parser.add_argument("--no-han-total-timesteps", type=int, default=None,
                        help="Training steps for the MAPPO(no-HAN) ablation. Defaults to --total-timesteps.")
    parser.add_argument("--skip-system-eval", action="store_true",
                        help="Skip checkpoint evaluation and only use history summary when available.")
    parser.add_argument("--plot-window", type=int, default=DEFAULT_PLOT_WINDOW,
                        help="Smoothing window used by the publication-style reward figure.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for JSON/CSV summaries and figures.")
    parser.add_argument("--reuse-methods-from", type=str, nargs="*", default=[],
                        help="Previous baseline_compare directory or comparison_summary.json to reuse methods from.")
    parser.add_argument("--reuse-methods", type=str, nargs="*", default=[],
                        help="Method names to reuse from --reuse-methods-from. Empty means all non-system methods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_publication_style()
    requested_baselines = [normalize_baseline_name(name) for name in args.baselines]
    baselines = DEFAULT_BASELINES if "all" in requested_baselines else requested_baselines
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "baseline_compare" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir, checkpoint, history_path = infer_system_artifacts(args.system_run_dir, args.system_checkpoint)
    objective = args.objective
    system_method: Optional[Dict] = None

    if checkpoint or history_path:
        config_data = load_run_config(checkpoint, history_path)
        objective = detect_objective(config_data)
    else:
        config_data = build_default_train_config(
            objective=objective,
            seed=args.seed,
            max_steps=args.max_steps or TrainConfig().max_steps,
            num_users=args.num_users,
            best_model_metric=args.best_model_metric,
        )

    config_data["best_model_metric"] = args.best_model_metric

    if args.run_mode == "train_compare":
        resume_checkpoint = checkpoint if args.resume_system else None
        training_run_dir = prepare_system_run_dir(
            requested_run_dir=args.system_run_dir,
            timestamp=timestamp,
            resume_system=args.resume_system,
            overwrite_system_run_dir=args.overwrite_system_run_dir,
        )
        run_dir, checkpoint, history_path, config_data = run_system_training(
            config_data=config_data,
            objective=objective,
            system_run_dir=training_run_dir,
            device=args.device,
            episodes=args.episodes,
            max_steps=args.max_steps,
            total_timesteps=args.total_timesteps,
            early_stop_patience=args.early_stop_patience,
            exp_name=args.exp_name,
            resume_checkpoint=resume_checkpoint,
        )
    elif checkpoint is None and history_path is None:
        raise FileNotFoundError(
            "compare_only mode requires an existing --system-run-dir or --system-checkpoint with training artifacts."
        )

    if checkpoint and not args.skip_system_eval:
        system_method = evaluate_system_checkpoint(
            checkpoint=checkpoint,
            config_data=config_data,
            objective=objective,
            episodes=args.episodes,
            device=resolve_device(args.device),
            max_steps=args.max_steps,
        )
        system_method["source"] = "checkpoint_eval"
        if history_path and history_path.exists():
            system_method["training_history"] = str(history_path)
    elif history_path and history_path.exists():
        _, system_method = extract_history_method(history_path)

    methods: List[Dict] = []
    if system_method is not None:
        methods.append(system_method)

    if args.reuse_methods_from:
        methods.extend(
            load_reused_methods(
                sources=args.reuse_methods_from,
                include_methods=args.reuse_methods,
                exclude_methods=baselines,
            )
        )

    for baseline_name in baselines:
        if baseline_name in {"random", "min_distance"}:
            result = evaluate_simple_heuristic_with_offload_search(
                strategy=baseline_name,
                objective=objective,
                config=config_data,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                offload_grid=args.fixed_offload_grid,
                selection_metric_name=args.compare_ranking_metric,
            )
            result["source"] = "heuristic_eval"
        elif baseline_name == "dqn":
            result = train_and_evaluate_dqn_baseline(
                config=config_data,
                objective=objective,
                output_dir=output_dir,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                total_timesteps=args.dqn_timesteps or args.total_timesteps,
                offload_bins=args.dqn_offload_grid,
                device_name=args.device,
            )
            result["source"] = "dqn_train_eval"
        elif baseline_name == "maddpg":
            result = train_and_evaluate_maddpg_baseline(
                config=config_data,
                objective=objective,
                output_dir=output_dir,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                total_timesteps=args.maddpg_timesteps or args.total_timesteps,
                device_name=args.device,
            )
            result["source"] = "maddpg_train_eval"
        elif baseline_name == "pdqn":
            result = train_and_evaluate_pdqn_baseline(
                config=config_data,
                objective=objective,
                output_dir=output_dir,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                total_timesteps=args.pdqn_timesteps or args.total_timesteps,
                device_name=args.device,
            )
            result["source"] = "pdqn_train_eval"
        elif baseline_name == "mappo_no_han":
            result = train_and_evaluate_no_han_mappo(
                config_data=config_data,
                objective=objective,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.no_han_total_timesteps or args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
            )
            result["source"] = "mappo_no_han_train_eval"
        elif baseline_name == "han_maddpg":
            result = train_and_evaluate_han_maddpg_baseline(
                config_data=config_data,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.maddpg_timesteps or args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
            )
            result["source"] = "han_maddpg_train_eval"
        elif baseline_name == "han_pdqn":
            result = train_and_evaluate_han_pdqn_baseline(
                config_data=config_data,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.pdqn_timesteps or args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
            )
            result["source"] = "han_pdqn_train_eval"
        else:
            policy = build_policy(
                name=baseline_name,
                objective=objective,
                fixed_offload=args.fixed_offload_grid[0],
                joint_offload_grid=args.joint_offload_grid,
            )
            result = evaluate_policy(
                policy=policy,
                objective=objective,
                config=config_data,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            result["source"] = "heuristic_eval"
        methods.append(result)

    methods = annotate_priority_metrics(methods, metric_name=args.compare_ranking_metric)
    methods = order_methods(methods)
    leaders = primary_metric_leaders(methods)
    json_path = save_results_json(
        output_dir,
        {
            "generated_at": timestamp,
            "run_mode": args.run_mode,
            "objective": objective,
            "best_model_metric": args.best_model_metric,
            "compare_ranking_metric": args.compare_ranking_metric,
            "primary_compare_metrics": [label for _, label in PRIMARY_COMPARE_METRICS],
            "primary_metric_leaders": leaders,
            "system_run_dir": str(run_dir) if run_dir else None,
            "system_checkpoint": str(checkpoint) if checkpoint else None,
            "training_history": str(history_path) if history_path else None,
            "total_timesteps": int(config_data.get("total_timesteps", args.total_timesteps)),
            "env_config": asdict(build_env_config_from_train_config(config_data, seed=args.seed, max_steps=args.max_steps)),
            "methods": methods,
        },
    )
    csv_path = save_results_csv(output_dir, methods)
    metrics_plot = plot_method_comparison(methods, output_dir)
    episode_csv_path = save_episode_metrics_csv(output_dir, methods)
    reward_curve_plot = plot_training_curve_vs_baselines(
        history_path,
        methods,
        output_dir,
        window=args.plot_window,
    )
    step_metric_plots = plot_step_metric_curves(history_path, output_dir, window=args.plot_window)
    episode_metric_plot = plot_additional_metric_curves(methods, output_dir)
    tradeoff_plot = plot_delay_energy_tradeoff(methods, output_dir)
    reliability_plot = plot_success_continuity_scatter(methods, output_dir)
    radar_plot = plot_performance_radar(methods, output_dir)
    reward_distribution_plot = plot_reward_distribution(methods, output_dir)
    dashboard_plot = plot_paper_dashboard(history_path, methods, output_dir, window=args.plot_window)

    print(json.dumps(methods, ensure_ascii=False, indent=2))
    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")
    if episode_csv_path is not None:
        print(f"Episode metrics CSV saved to: {episode_csv_path}")
    if metrics_plot is not None:
        print(f"Metric comparison figure: {metrics_plot}")
    if reward_curve_plot is not None:
        print(f"Reward curve comparison figure: {reward_curve_plot}")
    for step_metric_plot in step_metric_plots:
        print(f"Step metric curve figure: {step_metric_plot}")
    if episode_metric_plot is not None:
        print(f"Episode metric comparison figure: {episode_metric_plot}")
    if tradeoff_plot is not None:
        print(f"Delay-energy trade-off figure: {tradeoff_plot}")
    if reliability_plot is not None:
        print(f"Success-continuity trade-off figure: {reliability_plot}")
    if radar_plot is not None:
        print(f"Performance radar figure: {radar_plot}")
    if reward_distribution_plot is not None:
        print(f"Reward distribution figure: {reward_distribution_plot}")
    if dashboard_plot is not None:
        print(f"Paper dashboard figure: {dashboard_plot}")


if __name__ == "__main__":
    main()
