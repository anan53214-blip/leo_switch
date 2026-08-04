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
import logging
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

from src.environment.gym_env import (
    EnvConfig,
    LEOSatelliteEnv,
    REWARD_BREAKDOWN_KEYS,
    build_env_config,
    summarize_env_stats,
)
from src.environment.user import UserState
from src.algorithm.replay_buffer import MultiAgentReplayBuffer
from src.algorithm.maddpg import (
    MADDPGAlgorithm,
    MADDPGConfig,
)
from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig
from src.features.satellite_load import (
    SATELLITE_CONTEXT_FEATURE_DIM,
    build_satellite_context_features,
)

from scripts.load_balance_metrics import (
    empirical_cdf,
    normalize_load_balance_metrics,
    summarize_load_variance_samples,
)
from scripts.paper_metrics import (
    PRIMARY_COMPARE_METRICS as PAPER_PRIMARY_COMPARE_METRICS,
    SUCCESS_DEPENDENT_METRICS,
    bootstrap_mean_ci,
    derive_paper_metrics,
    metric_scale as unified_metric_scale,
)
from scripts.baseline_plot_config import (
    ADDITIONAL_EPISODE_METRICS,
    BAR_HATCH_PATTERNS,
    BASELINE_COLORS,
    BASELINE_LINESTYLES,
    BASELINE_MARKERS,
    CORE_BAR_METRICS,
    LEARNED_BASELINE_COLORS,
    PAPER_COLORS,
    RADAR_METRICS,
    REWARD_COMPONENT_STEP_METRICS,
    SCATTER_LABEL_OFFSETS,
    SYSTEM_STYLE,
    TRAINING_QOS_STEP_METRICS,
)

try:
    from scripts.train import (
        BEST_MODEL_METRIC_CHOICES,
        AttentionMAPPOTrainer,
        HANCandidateAttentionMAPPOTrainer,
        HANMADDPGTrainer,
        HANMAPPOTrainer,
        HANPDQNTrainer,
        TrainConfig,
        MODEL_SCHEMA_VERSION,
        GEOMETRY_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        compute_model_selection_score,
        energy_per_resolved_task,
        energy_per_successful_task,
    )
except ModuleNotFoundError:
    # Compatible with direct execution: python scripts/compare_system_baselines.py
    from train import (
        BEST_MODEL_METRIC_CHOICES,
        AttentionMAPPOTrainer,
        HANCandidateAttentionMAPPOTrainer,
        HANMADDPGTrainer,
        HANMAPPOTrainer,
        HANPDQNTrainer,
        TrainConfig,
        MODEL_SCHEMA_VERSION,
        GEOMETRY_SCHEMA_VERSION,
        ENVIRONMENT_SCHEMA_VERSION,
        compute_model_selection_score,
        energy_per_resolved_task,
        energy_per_successful_task,
    )

DEFAULT_BASELINES = [
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
    "maddpg",
    "pdqn",
    "han_mappo",
    "mappo_no_han",
    "attn_mappo",
    "han_attn",
    "han_maddpg",
    "han_pdqn",
]

DEFAULT_SYSTEM_RUN_DIR = PROJECT_ROOT / "results" / "full_train_latency_priority"
DEFAULT_SYSTEM_EXP_NAME = "han_mappo_latency_priority"
DEFAULT_TOTAL_TIMESTEPS = TrainConfig.total_timesteps
DEFAULT_EVAL_EPISODES = TrainConfig.eval_episodes
DEFAULT_PLOT_WINDOW = 3
DEFAULT_SELECTION_METRIC = TrainConfig.best_model_metric
EVALUATION_SEED_OFFSET = 1_000_000
HEURISTIC_TUNING_SEED_OFFSET = 500_000
TRAIN_ARTIFACT_FILENAMES = (
    "training_history.json",
    "best_model.pt",
    "final_model.pt",
)

PRIMARY_COMPARE_METRICS = list(PAPER_PRIMARY_COMPARE_METRICS)

DISPLAY_NAME_MAP = {
    "random": "Random",
    "min_distance": "Min-Distance",
    "full_local": "Full-Local",
    "joint_greedy": "Joint Greedy",
    "dqn": "DQN",
    "maddpg": "MADDPG",
    "pdqn": "PDQN",
    "han_mappo": "HAN+MAPPO",
    "mappo_no_han": "MAPPO",
    "attn_mappo": "Attn+MAPPO",
    "han_attn": "HAN+Attn",
    "han_maddpg": "HAN+MADDPG",
    "han_pdqn": "HAN+PDQN",
}

SUMMARY_METRIC_KEYS = [
    "avg_delay",
    "avg_success_delay",
    "p95_success_delay",
    "total_energy",
    "handover_success_rate",
    "handover_failure_rate",
    "forced_termination_rate",
    "total_user_seconds",
    "blocked_user_seconds",
    "blocked_time_ratio",
    "handover_interruption_seconds",
    "service_interruption_seconds",
    "total_handovers",
    "handover_attempts",
    "handover_committed",
    "handover_aborted",
    "handover_radio_failures",
    "migration_rejections",
    "reconnection_attempts",
    "reconnections",
    "failed_handovers",
    "service_continuity_rate",
    "service_availability_rate",
    "task_completion_rate",
    "task_success_rate",
    "task_failure_rate",
    "task_settlement_rate",
    "task_resolution_rate",
    "pending_task_rate",
    "handover_frequency",
    "handovers_per_user_minute",
    "load_balance_variance",
    "load_balance_coefficient",
    "load_variance_sample_count",
    "mec_load_fairness",
    "jain_mec_load_fairness",
    "active_mec_load_fairness",
    "reachable_jain_mec_load_fairness",
    "active_load_balance_score",
    "avg_load_balance_score",
    "resolved_tasks",
    "pending_tasks",
    "total_tasks",
    "completed_tasks",
    "deadline_violations",
    "failed_tasks",
    "deadline_violation_rate",
    "energy_per_successful_task",
]

ACTION_DIAGNOSTIC_KEYS = [
    "handover_action_rate",
    "local_compute_rate",
    "mean_offload_ratio",
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
    "load_balance_variance": False,
    "load_balance_coefficient": True,
    "mec_load_fairness": True,
    "jain_mec_load_fairness": True,
    "active_load_balance_score": True,
    "avg_load_balance_score": True,
    "handover_failure_rate": False,
    "handover_frequency": False,
    "handovers_per_user_minute": False,
    "forced_termination_rate": False,
    "avg_delay": False,
    "avg_success_delay": False,
    "p95_success_delay": False,
    "total_energy": False,
    "energy_per_resolved_task": False,
    "energy_per_successful_task": False,
    "pending_task_rate": False,
    "blocked_time_ratio": False,
    "deadline_violation_rate": False,
}

def detect_objective(config: Dict) -> str:
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
    source_records = [
        record
        for record in training
        if not bool(record.get("partial_episode", False))
    ]
    steps = np.array([record.get("total_steps", 0) for record in source_records], dtype=float)
    rewards = np.array(
        [
            record.get(
                "mean_reward",
                record.get("eval_mean_reward", record.get("reward", 0.0)),
            )
            for record in source_records
        ],
        dtype=float,
    )
    return steps, rewards


def metric_record_value(record: Dict, metric_key: str) -> Optional[float]:
    if metric_key == "mean_reward":
        for key in ("mean_reward", "eval_mean_reward", "reward"):
            if key in record:
                return float(record.get(key, 0.0))
        return None
    if metric_key in record:
        return float(record.get(metric_key, 0.0))
    return None


def extract_training_metric_curve(records: Sequence[Dict], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for record in records:
        if bool(record.get("partial_episode", False)):
            continue
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
    evaluation = extract_training_evaluation_records(payload, training)

    if training:
        steps, rewards = extract_training_reward_curve(training)
        valid_mask = np.isfinite(steps) & np.isfinite(rewards)
        steps = steps[valid_mask]
        rewards = rewards[valid_mask]
        if len(steps) > 0:
            order = np.argsort(steps)
            return steps[order], rewards[order], evaluation

    if evaluation:
        steps = np.array([record.get("total_steps", 0) for record in evaluation], dtype=float)
        rewards = np.array([record.get("eval_mean_reward", 0.0) for record in evaluation], dtype=float)
        valid_mask = np.isfinite(steps) & np.isfinite(rewards)
        order = np.argsort(steps[valid_mask])
        return steps[valid_mask][order], rewards[valid_mask][order], evaluation

    return np.array([], dtype=float), np.array([], dtype=float), evaluation


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


def reward_component_step_metrics_for_history(history_path: Optional[Path]) -> List[tuple[str, str, str, float]]:
    """返回当前 reward 分项；旧环境结果不再与 schema v5 混合绘制。"""
    del history_path
    return list(REWARD_COMPONENT_STEP_METRICS)


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


def method_training_history_paths(
    method: Dict,
    output_dir: Optional[Path] = None,
    primary_history_path: Optional[Path] = None,
) -> List[Path]:
    """Return de-duplicated histories for one learning method."""
    candidates: List[Path] = []
    if method.get("is_system") and primary_history_path is not None:
        candidates.append(primary_history_path)
    for value in method.get("training_history_paths", []) or []:
        candidates.append(Path(str(value)))
    single_path = method_training_history_path(method, output_dir=output_dir)
    if single_path is not None:
        candidates.append(single_path)

    paths: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if path.exists() and resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return paths


def aggregate_reward_curves(
    history_paths: Sequence[Path],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align seed curves on common steps and return mean with 95% CI."""
    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for path in history_paths:
        steps, rewards, _ = load_training_curve_from_path(path)
        if len(steps) == 0:
            continue
        smoothed, _ = reward_smooth(rewards, window=max(window, 3))
        curves.append((steps, smoothed))
    if not curves:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    common_steps = set(float(value) for value in curves[0][0])
    for steps, _ in curves[1:]:
        common_steps.intersection_update(float(value) for value in steps)
    ordered_steps = np.array(sorted(common_steps), dtype=float)
    if len(ordered_steps) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    reward_rows = []
    for steps, rewards in curves:
        reward_by_step = {
            float(step): float(reward)
            for step, reward in zip(steps, rewards)
        }
        reward_rows.append([reward_by_step[float(step)] for step in ordered_steps])
    reward_matrix = np.asarray(reward_rows, dtype=float)
    means = np.mean(reward_matrix, axis=0)
    lows = np.empty(len(ordered_steps), dtype=float)
    highs = np.empty(len(ordered_steps), dtype=float)
    for index in range(len(ordered_steps)):
        _, lows[index], highs[index] = bootstrap_mean_ci(
            reward_matrix[:, index],
            seed=20260728 + index,
        )
    return ordered_steps, means, lows, highs


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


def figure_output_path(
    output_dir: Path,
    filename: str,
    output_suffix: str = "",
) -> Path:
    suffix = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in output_suffix.strip().strip("_").replace("-", "_").replace(" ", "_")
    )
    path = Path(filename)
    if not suffix:
        return output_dir / path.name
    return output_dir / f"{path.stem}_{suffix}{path.suffix}"


def build_env_config_from_train_config(config: Dict, seed: Optional[int], max_steps: Optional[int]) -> EnvConfig:
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    return build_env_config(config, **overrides)


def build_env_for_objective(
    objective: str,
    config: Dict,
    seed: Optional[int],
    max_steps: Optional[int],
) -> LEOSatelliteEnv:
    env_config = build_env_config_from_train_config(config, seed=seed, max_steps=max_steps)
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
    config["exp_name"] = DEFAULT_SYSTEM_EXP_NAME
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


def find_existing_checkpoint(run_dir: Path) -> Optional[Path]:
    for filename in ("best_model.pt", "final_model.pt"):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


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
    base_name = name.split("(", 1)[0]
    normalized_name = base_name.strip().lower().replace("-", "_")
    for method_name in sorted(DISPLAY_NAME_MAP, key=len, reverse=True):
        display_name = DISPLAY_NAME_MAP[method_name]
        if normalized_name == method_name or normalized_name.startswith(f"{method_name}_"):
            return display_name
    if is_system:
        return "HAN+MAPPO"
    return DISPLAY_NAME_MAP.get(name, DISPLAY_NAME_MAP.get(base_name, name))


def system_display_name(methods: Sequence[Dict], fallback: str = "HAN+MAPPO") -> str:
    for method in methods:
        if method.get("is_system"):
            return str(method.get("display_name", method.get("method", fallback)))
    return fallback


def normalize_baseline_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def filter_duplicate_system_baselines(
    baselines: Sequence[str],
    config_data: Dict,
) -> List[str]:
    system_algorithm = normalize_baseline_name(str(config_data.get("algorithm", "")))
    if not system_algorithm:
        return list(baselines)
    system_method = {
        "mappo": "han_mappo",
        "attn_mappo": "attn_mappo",
        "han_attn": "han_attn",
    }.get(system_algorithm, system_algorithm)
    return [
        baseline
        for baseline in baselines
        if normalize_baseline_name(baseline) != system_method
    ]


def compute_deadline_violation_rate(summary: Dict) -> float:
    total_tasks = float(summary.get("total_tasks", 0.0))
    return float(summary.get("deadline_violations", 0.0)) / max(total_tasks, 1.0)


def compute_handover_frequency(summary: Dict) -> float:
    total_user_seconds = float(summary.get("total_user_seconds", 0.0))
    if total_user_seconds <= 0.0:
        return 0.0
    committed = summary.get(
        "handover_committed",
        summary.get("total_handovers", 0.0),
    )
    return float(committed) / total_user_seconds


def summarize_env_stats_with_load_balance(env_stats: Dict) -> Dict:
    summary = summarize_env_stats(env_stats)
    if "load_variance_samples" in env_stats:
        summary["load_variance_samples"] = env_stats.get("load_variance_samples", [])
    return derive_paper_metrics(normalize_load_balance_metrics(summary))


def build_episode_records(rewards: Sequence[float], summaries: Sequence[Dict]) -> List[Dict]:
    episode_records: List[Dict] = []
    for episode_index, (reward, summary) in enumerate(zip(rewards, summaries), start=1):
        summary = derive_paper_metrics(normalize_load_balance_metrics(summary))
        record = {
            "episode": episode_index,
            "reward": float(reward),
        }
        for key in SUMMARY_METRIC_KEYS:
            if key in {"deadline_violation_rate", "handover_frequency"}:
                continue
            record[key] = float(summary.get(key, 0.0))
        for key in REWARD_BREAKDOWN_KEYS:
            record[key] = float(summary.get(key, 0.0))
        record["deadline_violation_rate"] = compute_deadline_violation_rate(summary)
        record["handover_frequency"] = float(
            summary.get("handover_frequency", compute_handover_frequency(summary))
        )
        record["mec_load_fairness"] = float(
            summary.get(
                "mec_load_fairness",
                summary.get("active_load_balance_score", summary.get("avg_load_balance_score", 0.0)),
            )
        )
        record["active_load_balance_score"] = record["mec_load_fairness"]
        record["avg_load_balance_score"] = record["mec_load_fairness"]
        record["load_balance_coefficient"] = record["mec_load_fairness"]
        record["energy_per_successful_task"] = float(energy_per_successful_task(record))
        record = derive_paper_metrics(record)
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
    variance_samples: List[float] = []
    for summary in summaries:
        samples = summary.get("load_variance_samples")
        if samples is not None:
            variance_samples.extend(float(value) for value in samples)
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
    if variance_samples:
        result.update(summarize_load_variance_samples(variance_samples))
    return derive_paper_metrics(normalize_load_balance_metrics(result))


def action_diagnostics(
    action_batches: Sequence[np.ndarray],
    min_effective_offload_ratio: float = EnvConfig.min_effective_offload_ratio,
) -> Dict[str, float]:
    if not action_batches:
        return {key: 0.0 for key in ACTION_DIAGNOSTIC_KEYS}

    actions = np.concatenate([np.asarray(batch, dtype=np.float32).reshape(-1, 2) for batch in action_batches], axis=0)
    if actions.size == 0:
        return {key: 0.0 for key in ACTION_DIAGNOSTIC_KEYS}

    handover = actions[:, 0]
    offload = np.clip(actions[:, 1], 0.0, 1.0)
    return {
        "handover_action_rate": float(np.mean(handover > 0.0)),
        "local_compute_rate": float(np.mean(offload < float(min_effective_offload_ratio))),
        "mean_offload_ratio": float(np.mean(offload)),
    }


def ensure_action_diagnostic_fields(method: Dict) -> Dict:
    normalized = normalize_load_balance_metrics(method)
    normalized.pop("mec_activity_score", None)
    normalized.pop("mec_load_mean", None)
    normalized.pop("service_downtime_rate", None)
    for key in ACTION_DIAGNOSTIC_KEYS:
        try:
            normalized[key] = float(normalized.get(key, 0.0))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def selection_score(method: Dict, metric_name: str) -> float:
    return float(compute_model_selection_score(method, metric_name))


def comparison_metric_value(method: Dict, metric_key: str) -> float:
    if (
        metric_key in SUCCESS_DEPENDENT_METRICS
        and float(method.get("completed_tasks", 0.0) or 0.0) <= 0.0
    ):
        return float("nan")
    try:
        return float(method.get(metric_key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def annotate_priority_metrics(methods: Sequence[Dict], metric_name: str) -> List[Dict]:
    annotated = [ensure_action_diagnostic_fields(method) for method in methods]
    for method in annotated:
        method["selection_metric"] = metric_name
        method["selection_score"] = selection_score(method, metric_name)
        method["energy_per_resolved_task"] = float(energy_per_resolved_task(method))
        method["energy_per_successful_task"] = float(energy_per_successful_task(method))
        method["primary_metric_wins"] = []

    for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
        values = [
            comparison_metric_value(method, metric_key)
            for method in annotated
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        if not finite_values:
            continue
        best_value = (
            max(finite_values)
            if HIGHER_IS_BETTER.get(metric_key, True)
            else min(finite_values)
        )
        for method, value in zip(annotated, values):
            if np.isfinite(value) and np.isclose(
                value,
                best_value,
                rtol=1e-9,
                atol=1e-9,
            ):
                method["primary_metric_wins"].append(metric_label)

    for method in annotated:
        method["primary_metric_win_count"] = len(method["primary_metric_wins"])
        method["primary_metric_wins_text"] = " | ".join(method["primary_metric_wins"])

    return annotated


def primary_metric_leaders(methods: Sequence[Dict]) -> Dict[str, List[str]]:
    leaders: Dict[str, List[str]] = {}
    for metric_key, metric_label in PRIMARY_COMPARE_METRICS:
        values = [
            comparison_metric_value(method, metric_key)
            for method in methods
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        if not finite_values:
            continue
        best_value = (
            max(finite_values)
            if HIGHER_IS_BETTER.get(metric_key, True)
            else min(finite_values)
        )
        leaders[metric_label] = [
            method.get("display_name", method.get("method", ""))
            for method, value in zip(methods, values)
            if np.isfinite(value)
            and np.isclose(value, best_value, rtol=1e-9, atol=1e-9)
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
    array = np.asarray(values, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(array))
    if len(valid_indices) == 0:
        return -1
    fn = np.argmax if higher_is_better else np.argmin
    local_index = int(fn(array[valid_indices]))
    return int(valid_indices[local_index])


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
        legal_mask = env.get_handover_action_mask()[user.user_id]
        legal_indices = [
            index
            for index in range(1, min(len(visible_sats), env.max_visible_sats) + 1)
            if legal_mask[index]
        ]
        if not legal_indices:
            return 0
        if self.strategy == "random":
            return int(env.rng.choice([0, *legal_indices]))
        if self.strategy == "min_distance":
            target_action = min(
                legal_indices,
                key=lambda action: visible_sats[action - 1].distance_km,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if user.serving_satellite >= 0:
            current = env._get_satellite_visibility(user, user.serving_satellite)
            if (
                current is not None
                and current.is_visible
                and current.distance_km <= visible_sats[target_action - 1].distance_km
            ):
                return 0
        return target_action

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        actions = np.zeros((env.num_users, 2), dtype=np.float32)
        for user_id, user in enumerate(env.user_manager.users):
            visible_sats = env._get_handover_candidates(user)
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

        link_feasible, _ = env._check_handover_link_feasibility(target_vis)
        migration_feasible = (
            predicted_queue_len + migration_load
            <= target_server.config.max_queue_size
        )
        if not link_feasible or not migration_feasible:
            return -float(env.config.reward_failed_handover_penalty)

        interruption_ratio = min(
            float(env.config.handover_delay_sec)
            / max(float(env.config.time_step_sec), 1e-6),
            1.0,
        )
        return -float(env.config.reward_interruption_weight) * interruption_ratio

    def _task_score(self, env: LEOSatelliteEnv, total_delay: float, total_energy: float, max_delay: float) -> float:
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
            score -= float(env.config.reward_interruption_weight)

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
            visible_sats = env._get_handover_candidates(user)
            legal_mask = env.get_handover_action_mask()[user_id]
            candidate_actions = [
                action
                for action in range(min(len(visible_sats), env.max_visible_sats) + 1)
                if legal_mask[action]
            ]

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
    if hasattr(env, "get_handover_action_mask"):
        handover_masks = env.get_handover_action_mask(env.max_visible_sats)
    else:
        # 仅用于轻量测试替身/旧外部环境；正式 LEOSatelliteEnv 始终走
        # 上面的完整可行性掩码。
        handover_masks = np.zeros(
            (env.num_users, env.max_visible_sats + 1),
            dtype=bool,
        )
        handover_masks[:, 0] = True
        for user_id, user in enumerate(env.user_manager.users):
            valid_count = min(
                len(env._get_handover_candidates(user)),
                env.max_visible_sats,
            )
            handover_masks[user_id, 1:valid_count + 1] = True
        pre_handover = np.asarray(env.get_pre_handover_mask(), dtype=bool)
        handover_masks[~pre_handover, 1:] = False
    for user_id in range(env.num_users):
        for handover_action in np.flatnonzero(handover_masks[user_id]):
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


def _evaluate_action_selector(
    method_name: str,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
    select_actions,
) -> Dict:
    """Evaluate any raw-observation policy under the shared scenario protocol."""
    env = build_env_for_objective(objective, config, seed=seed, max_steps=max_steps)
    rewards: List[float] = []
    summaries: List[Dict] = []
    action_batches: List[np.ndarray] = []

    for episode_idx in range(episodes):
        observations, _ = env.reset(
            seed=seed + EVALUATION_SEED_OFFSET + episode_idx
        )
        done = False
        episode_reward = 0.0
        while not done:
            env_actions = select_actions(env, observations)
            action_batches.append(np.asarray(env_actions, dtype=np.float32).copy())
            observations, reward, terminated, truncated, _ = env.step(env_actions)
            episode_reward += scalar_reward_value(reward)
            done = terminated or truncated
        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())

    extra = action_diagnostics(
        action_batches,
        float(config.get("min_effective_offload_ratio", EnvConfig.min_effective_offload_ratio)),
    )
    return summarize_results(method_name, rewards, summaries, extra=extra, is_system=False)


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
    q_net.eval()
    rng = np.random.default_rng(seed)

    def select_actions(env, observations):
        masks = dqn_action_mask(env, offload_bins)
        indices = select_dqn_indices(
            q_net, observations, masks, 0.0, rng, device
        )
        return dqn_indices_to_env_actions(indices, offload_bins)

    return _evaluate_action_selector(
        "dqn",
        objective,
        config,
        episodes,
        seed,
        max_steps,
        select_actions,
    )


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
    checkpoint_dir = output_dir / "learned_baselines" / "dqn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "training_history.json"
    checkpoint_path = checkpoint_dir / "dqn_model.pt"
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
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
    training_evaluation_records: List[Dict] = []
    validation_episodes = max(
        int(config.get("eval_episodes", TrainConfig.eval_episodes)),
        1,
    )
    validation_interval = max(
        int(config.get("save_interval", TrainConfig.save_interval)),
        1,
    )
    best_validation_reward = -float("inf")
    best_q_state = clone_state_dict(q_net)
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
        user_rewards = per_user_training_rewards(env, reward)

        for user_id in range(env.num_users):
            replay.append(
                (
                    observations[user_id].astype(np.float32, copy=True),
                    int(action_indices[user_id]),
                    float(user_rewards[user_id]),
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

        completed_steps = step_idx + 1
        if (
            completed_steps % validation_interval == 0
            or completed_steps == int(total_timesteps)
        ):
            validation_result = evaluate_dqn_policy(
                q_net=q_net,
                objective=objective,
                config=config,
                episodes=validation_episodes,
                seed=seed,
                max_steps=max_steps,
                offload_bins=clean_bins,
                device=device,
            )
            validation_reward = float(validation_result.get("mean_reward", 0.0))
            training_evaluation_records.append(
                {
                    "total_steps": completed_steps,
                    "eval_mean_reward": validation_reward,
                    "eval_std_reward": float(
                        validation_result.get("std_reward", 0.0)
                    ),
                    "eval_episodes": validation_episodes,
                    "best_model_metric": "reward",
                    "best_model_score": validation_reward,
                }
            )
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                best_q_state = clone_state_dict(q_net)
            q_net.train()

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

    if not training_evaluation_records:
        validation_result = evaluate_dqn_policy(
            q_net=q_net,
            objective=objective,
            config=config,
            episodes=validation_episodes,
            seed=seed,
            max_steps=max_steps,
            offload_bins=clean_bins,
            device=device,
        )
        best_validation_reward = float(validation_result.get("mean_reward", 0.0))
        best_q_state = clone_state_dict(q_net)
        training_evaluation_records.append(
            {
                "total_steps": int(total_timesteps),
                "eval_mean_reward": best_validation_reward,
                "eval_std_reward": float(validation_result.get("std_reward", 0.0)),
                "eval_episodes": validation_episodes,
                "best_model_metric": "reward",
                "best_model_score": best_validation_reward,
            }
        )

    final_q_state = clone_state_dict(q_net)
    q_net.load_state_dict(best_q_state)
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
    torch.save(
        {
            "q_state_dict": final_q_state,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "offload_bins": clean_bins,
            "trained_timesteps": int(total_timesteps),
            "training_history": str(history_path),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "q_state_dict": best_q_state,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "offload_bins": clean_bins,
            "trained_timesteps": int(total_timesteps),
            "best_model_metric": "reward",
            "best_model_score": best_validation_reward,
            "training_history": str(history_path),
        },
        best_checkpoint_path,
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
                "training_evaluation": training_evaluation_records,
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
    result["checkpoint"] = str(best_checkpoint_path)
    result["training_history"] = str(history_path)
    return result


def maddpg_action_mask(env: LEOSatelliteEnv) -> np.ndarray:
    masks = np.zeros((env.num_users, env.max_visible_sats + 1), dtype=bool)
    masks[:, 0] = True
    for user_id, user in enumerate(env.user_manager.users):
        visible_sats = env._get_handover_candidates(user)
        valid_count = min(len(visible_sats), env.max_visible_sats)
        if valid_count > 0:
            masks[user_id, 1:valid_count + 1] = True
    pre_handover_mask = np.asarray(env.get_pre_handover_mask(), dtype=bool)
    masks[~pre_handover_mask, 1:] = False
    masks[:, 0] = True
    return masks


def scalar_reward_value(reward) -> float:
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        return float(np.mean(list(reward.values()))) if reward else 0.0
    return float(np.mean(np.asarray(reward, dtype=float)))


def per_user_training_rewards(env: LEOSatelliteEnv, reward) -> np.ndarray:
    """Return the same local-plus-cooperative reward vector for every learner."""
    local_rewards = getattr(env, "last_user_rewards", None)
    if local_rewards is not None:
        reward_array = np.asarray(local_rewards, dtype=np.float32)
        if reward_array.shape == (env.num_users,):
            return reward_array.copy()
    return np.full(
        env.num_users,
        scalar_reward_value(reward),
        dtype=np.float32,
    )


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a module state for in-memory best-checkpoint tracking."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def evaluate_maddpg_policy(
    algorithm: MADDPGAlgorithm,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
) -> Dict:
    def select_actions(env, observations):
        actions, _, _ = algorithm.act(
            observations,
            maddpg_action_mask(env),
            deterministic=True,
        )
        return actions

    return _evaluate_action_selector(
        "maddpg",
        objective,
        config,
        episodes,
        seed,
        max_steps,
        select_actions,
    )


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

    maddpg_config = MADDPGConfig(
        num_agents=num_agents,
        obs_dim=obs_dim,
        max_candidates=handover_dim - 1,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(512, 256, 128),
        actor_lr=5e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        noise_start=0.35,
        noise_final=0.05,
        noise_decay_steps=max(int(total_timesteps * 0.7), 1),
        batch_size=128,
        replay_size=50_000,
        warmup_steps=min(1_000, max(64, total_timesteps // 20)),
        grad_clip_norm=1.0,
        seed=int(seed),
        device=str(device),
    )
    algo = MADDPGAlgorithm(maddpg_config)
    replay = MultiAgentReplayBuffer(
        capacity=algo.config.replay_size,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_feature_dim=algo.action_feature_dim,
        mask_dim=algo.handover_dim,
        device=str(device),
    )
    random.seed(seed)
    torch.manual_seed(seed)
    training_records: List[Dict] = []
    evaluation_records: List[Dict] = []
    recent_episode_rewards: deque = deque(maxlen=10)
    recent_actor_losses: deque = deque(maxlen=100)
    recent_critic_losses: deque = deque(maxlen=100)
    eval_interval_episodes = 10
    train_eval_episodes = max(
        int(config.get("eval_episodes", TrainConfig.eval_episodes)),
        1,
    )
    best_eval_reward = -float("inf")
    best_actor_state = clone_state_dict(algo.actor)
    best_critic_state = clone_state_dict(algo.critic)
    best_target_actor_state = clone_state_dict(algo.target_actor)
    best_target_critic_state = clone_state_dict(algo.target_critic)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    observations, _ = env.reset(seed=seed)
    for step_idx in range(max(int(total_timesteps), 0)):
        noise_std = algo._noise_std()
        masks = maddpg_action_mask(env)
        if step_idx < algo.config.warmup_steps:
            env_actions, action_features, _ = algo.random_actions(masks)
        else:
            env_actions, action_features, _ = algo.act(observations, masks, deterministic=False)

        next_observations, reward, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)
        reward_value = scalar_reward_value(reward)
        next_masks = maddpg_action_mask(env) if not done else np.zeros_like(masks)
        replay.add(
            observations.astype(np.float32, copy=True),
            action_features.astype(np.float32, copy=True),
            per_user_training_rewards(env, reward),
            next_observations.astype(np.float32, copy=True),
            bool(done),
            masks.astype(bool, copy=True),
            next_masks.astype(bool, copy=True),
        )

        observations = next_observations
        episode_reward += reward_value
        episode_length += 1

        if len(replay) >= max(algo.config.batch_size, algo.config.warmup_steps):
            stats = algo.update(replay)
            if stats:
                recent_actor_losses.append(float(stats.get("actor_loss", 0.0)))
                recent_critic_losses.append(float(stats.get("critic_loss", 0.0)))

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
                    algorithm=algo,
                    objective=objective,
                    config=config,
                    episodes=train_eval_episodes,
                    seed=seed,
                    max_steps=max_steps,
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
                    best_actor_state = clone_state_dict(algo.actor)
                    best_critic_state = clone_state_dict(algo.critic)
                    best_target_actor_state = clone_state_dict(algo.target_actor)
                    best_target_critic_state = clone_state_dict(algo.target_critic)
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
                "exploration_noise": float(algo.config.noise_final),
                "actor_loss": float(np.mean(recent_actor_losses)) if recent_actor_losses else 0.0,
                "critic_loss": float(np.mean(recent_critic_losses)) if recent_critic_losses else 0.0,
                "partial_episode": True,
            }
        )

    final_eval = evaluate_maddpg_policy(
        algorithm=algo,
        objective=objective,
        config=config,
        episodes=train_eval_episodes,
        seed=seed,
        max_steps=max_steps,
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
        best_actor_state = clone_state_dict(algo.actor)
        best_critic_state = clone_state_dict(algo.critic)
        best_target_actor_state = clone_state_dict(algo.target_actor)
        best_target_critic_state = clone_state_dict(algo.target_critic)

    final_actor_state = clone_state_dict(algo.actor)
    final_critic_state = clone_state_dict(algo.critic)
    final_target_actor_state = clone_state_dict(algo.target_actor)
    final_target_critic_state = clone_state_dict(algo.target_critic)
    algo.actor.load_state_dict(best_actor_state)
    algo.critic.load_state_dict(best_critic_state)
    algo.target_actor.load_state_dict(best_target_actor_state)
    algo.target_critic.load_state_dict(best_target_critic_state)

    result = evaluate_maddpg_policy(
        algorithm=algo,
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    result["trained_timesteps"] = int(total_timesteps)
    result["best_training_eval_reward"] = float(best_eval_reward)
    checkpoint_dir = output_dir / "learned_baselines" / "maddpg"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "training_history.json"
    checkpoint_path = checkpoint_dir / "maddpg_model.pt"
    final_checkpoint_path = checkpoint_dir / "maddpg_final_model.pt"

    def save_maddpg_checkpoint(
        path: Path,
        actor_state: Dict[str, torch.Tensor],
        critic_state: Dict[str, torch.Tensor],
        target_actor_state: Dict[str, torch.Tensor],
        target_critic_state: Dict[str, torch.Tensor],
    ) -> None:
        torch.save(
            {
                "config": asdict(algo.config),
                "actor_state_dict": actor_state,
                "critic_state_dict": critic_state,
                "target_actor_state_dict": target_actor_state,
                "target_critic_state_dict": target_critic_state,
                "actor_optimizer_state_dict": algo.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": algo.critic_optimizer.state_dict(),
                "train_step": int(algo.train_step),
                "obs_dim": obs_dim,
                "num_agents": num_agents,
                "handover_dim": handover_dim,
                "action_feature_dim": algo.action_feature_dim,
                "trained_timesteps": int(total_timesteps),
                "best_training_eval_reward": float(best_eval_reward),
                "training_history": str(history_path),
            },
            path,
        )

    save_maddpg_checkpoint(
        checkpoint_path,
        best_actor_state,
        best_critic_state,
        best_target_actor_state,
        best_target_critic_state,
    )
    save_maddpg_checkpoint(
        final_checkpoint_path,
        final_actor_state,
        final_critic_state,
        final_target_actor_state,
        final_target_critic_state,
    )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "method": "maddpg",
                    "objective": objective,
                    "total_timesteps": int(total_timesteps),
                    "seed": algo.config.seed,
                    "max_steps": int(max_steps) if max_steps is not None else None,
                    "actor_lr": algo.config.actor_lr,
                    "critic_lr": algo.config.critic_lr,
                    "gamma": algo.config.gamma,
                    "tau": algo.config.tau,
                    "noise_start": algo.config.noise_start,
                    "noise_final": algo.config.noise_final,
                    "noise_decay_steps": algo.config.noise_decay_steps,
                    "warmup_steps": algo.config.warmup_steps,
                    "batch_size": algo.config.batch_size,
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
        visible_sats = env._get_handover_candidates(user)
        valid_count = min(len(visible_sats), env.max_visible_sats)
        if valid_count > 0:
            masks[user_id, 1:valid_count + 1] = True
    pre_handover_mask = np.asarray(env.get_pre_handover_mask(), dtype=bool)
    masks[~pre_handover_mask, 1:] = False
    masks[:, 0] = True
    return masks


def pdqn_action_features_from_env_actions(
    env_actions: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks_np = np.asarray(masks, dtype=bool)
    num_agents, handover_dim = masks_np.shape
    actions = np.asarray(env_actions, dtype=np.float32).reshape(num_agents, 2).copy()
    handover = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, handover_dim - 1)
    for agent_id, action_id in enumerate(handover):
        if not masks_np[agent_id, action_id]:
            valid = np.flatnonzero(masks_np[agent_id])
            handover[agent_id] = int(valid[0]) if len(valid) else 0
    offload = np.clip(actions[:, 1], 0.0, 1.0).astype(np.float32)
    features = np.zeros((num_agents, handover_dim + 1), dtype=np.float32)
    features[np.arange(num_agents), handover] = 1.0
    features[:, -1] = offload
    env_actions = np.column_stack([handover, offload]).astype(np.float32)
    return env_actions, features, handover


def pdqn_safe_heuristic_actions(
    env: LEOSatelliteEnv,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks_np = np.asarray(masks, dtype=bool)
    actions = np.zeros((env.num_users, 2), dtype=np.float32)
    rvt_threshold = float(getattr(env.config, "rvt_threshold_sec", 60.0))

    for user_id, user in enumerate(env.user_manager.users):
        visible_sats = list(env._get_handover_candidates(user))[: env.max_visible_sats]
        keep_current = False
        if user.serving_satellite >= 0:
            vis = env._get_satellite_visibility(user, user.serving_satellite)
            keep_current = bool(vis is not None and vis.is_visible and vis.rvt_seconds >= rvt_threshold)

        handover = 0
        if not keep_current and visible_sats:
            target_idx = int(np.argmax([sat.elevation_deg for sat in visible_sats]))
            candidate = target_idx + 1
            if candidate < masks_np.shape[1] and masks_np[user_id, candidate]:
                handover = candidate

        task = env.user_tasks.get(user_id)
        actions[user_id, 0] = float(handover)
        actions[user_id, 1] = 0.5 if task is not None and (keep_current or handover > 0) else 0.0

    return pdqn_action_features_from_env_actions(actions, masks_np)


def pdqn_mixed_safe_random_actions(
    env: LEOSatelliteEnv,
    algorithm: PDQNAlgorithm,
    masks: np.ndarray,
    safe_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_actions, safe_features, safe_handover = pdqn_safe_heuristic_actions(env, masks)
    random_actions, random_features, random_handover = algorithm.random_actions(masks)
    use_safe = algorithm.rng.random(env.num_users) < float(safe_probability)
    env_actions = np.where(use_safe[:, None], safe_actions, random_actions).astype(np.float32)
    action_features = np.where(use_safe[:, None], safe_features, random_features).astype(np.float32)
    handover = np.where(use_safe, safe_handover, random_handover).astype(np.int64)
    return env_actions, action_features, handover


def evaluate_pdqn_policy(
    algorithm: PDQNAlgorithm,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
) -> Dict:
    def select_actions(env, observations):
        actions, _, _ = algorithm.act(
            observations,
            pdqn_action_mask(env),
            epsilon=0.0,
        )
        return actions

    return _evaluate_action_selector(
        "pdqn",
        objective,
        config,
        episodes,
        seed,
        max_steps,
        select_actions,
    )


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
    save_dir = output_dir / "learned_baselines" / "pdqn"
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "training_history.json"
    checkpoint_path = save_dir / "pdqn_model.pt"
    best_checkpoint_path = save_dir / "best_model.pt"
    log_path = save_dir / "pdqn_training.log"
    logger = logging.getLogger(
        f"{__name__}.raw_pdqn.{seed}.{id(output_dir)}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing_handler in list(logger.handlers):
        existing_handler.close()
        logger.removeHandler(existing_handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

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
            target_update_interval=int(config.get("target_update_interval", 500)),
            epsilon_start=float(config.get("epsilon_start", 1.0)),
            epsilon_final=float(config.get("epsilon_final", 0.02)),
            epsilon_decay_steps=max(
                int(total_timesteps * float(config.get("epsilon_decay_fraction", 0.25))),
                1_001,
                1,
            ),
            bc_loss_coef=float(config.get("bc_loss_coef", 0.0)),
            seed=int(seed),
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
    training_evaluation_records: List[Dict] = []
    recent_episode_rewards: deque = deque(maxlen=10)
    recent_q_losses: deque = deque(maxlen=100)
    recent_param_losses: deque = deque(maxlen=100)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    best_validation_reward = -float("inf")
    validation_episodes = max(
        int(config.get("eval_episodes", TrainConfig.eval_episodes)),
        1,
    )
    save_interval = max(
        int(config.get("save_interval", TrainConfig.save_interval)),
        1,
    )
    episode_step_budget = max(
        int(
            max_steps
            if max_steps is not None
            else getattr(env.config, "max_steps", EnvConfig.max_steps)
        ),
        1,
    )
    log_interval_steps = max(
        int(config.get("log_interval", TrainConfig.log_interval))
        * episode_step_budget,
        1,
    )
    history_config = {
        **config,
        "algorithm": "pdqn",
        "total_timesteps": int(total_timesteps),
        "batch_size": algo.config.batch_size,
        "warmup_steps": algo.config.warmup_steps,
        "replay_size": algo.config.replay_size,
        "target_update_interval": algo.config.target_update_interval,
        "epsilon_start": algo.config.epsilon_start,
        "epsilon_final": algo.config.epsilon_final,
        "epsilon_decay_steps": algo.config.epsilon_decay_steps,
        "epsilon_decay_fraction": float(
            config.get("epsilon_decay_fraction", 0.25)
        ),
        "bc_loss_coef": algo.config.bc_loss_coef,
        "safe_exploration_probability": float(
            config.get("safe_exploration_probability", 0.0)
        ),
        "seed": algo.config.seed,
        "device": device,
        "save_interval": save_interval,
        "log_interval_steps": log_interval_steps,
    }

    def write_training_history(
        *,
        evaluation_records: Optional[Sequence[Dict]] = None,
        summary: Optional[Dict] = None,
        completed_steps: int = 0,
    ) -> None:
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": history_config,
                    "training": training_records,
                    "training_evaluation": training_evaluation_records,
                    "evaluation": list(evaluation_records or []),
                    "summary": {
                        "total_steps": int(completed_steps),
                        "episodes": int(episode_count),
                        **dict(summary or {}),
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    logger.info("=" * 60)
    logger.info("开始训练纯 PDQN 基线")
    logger.info(
        "总步数: %d | warmup: %d | batch: %d | "
        "日志间隔: %d steps | 保存间隔: %d steps | device: %s",
        int(total_timesteps),
        int(algo.config.warmup_steps),
        int(algo.config.batch_size),
        log_interval_steps,
        save_interval,
        device,
    )
    logger.info("=" * 60)

    observations, _ = env.reset(seed=seed)
    for step_idx in range(max(int(total_timesteps), 0)):
        masks = pdqn_action_mask(env)
        if step_idx < algo.config.warmup_steps:
            env_actions, action_features, _ = pdqn_mixed_safe_random_actions(
                env,
                algo,
                masks,
                safe_probability=float(
                    config.get("safe_exploration_probability", 0.0)
                ),
            )
        else:
            exploration_actions, _, _ = pdqn_mixed_safe_random_actions(
                env,
                algo,
                masks,
                safe_probability=float(
                    config.get("safe_exploration_probability", 0.0)
                ),
            )
            env_actions, action_features, _ = algo.act(
                observations,
                masks,
                exploration_actions=exploration_actions,
            )

        next_observations, reward, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)
        reward_value = scalar_reward_value(reward)
        next_masks = pdqn_action_mask(env) if not done else np.zeros_like(masks)
        replay_reward = per_user_training_rewards(env, reward)
        replay.add(observations, action_features, replay_reward, next_observations, done, masks, next_masks)

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
            summary = summarize_env_stats_with_load_balance(env_stats)
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
                    "avg_success_delay": summary.get("avg_success_delay", 0.0),
                    "p95_success_delay": summary.get("p95_success_delay", 0.0),
                    "handover_frequency": summary.get("handover_frequency", 0.0),
                    "handovers_per_user_minute": summary.get("handovers_per_user_minute", 0.0),
                    "blocked_time_ratio": summary.get("blocked_time_ratio", 0.0),
                    "service_continuity_rate": summary.get("service_continuity_rate", 0.0),
                    "task_completion_rate": summary.get("task_completion_rate", 0.0),
                    "task_success_rate": summary.get("task_success_rate", 0.0),
                    "mec_load_fairness": summary.get("mec_load_fairness", summary.get("active_load_balance_score", summary.get("avg_load_balance_score", 0.0))),
                    "jain_mec_load_fairness": summary.get("jain_mec_load_fairness", 0.0),
                    "avg_load_balance_score": summary.get("mec_load_fairness", summary.get("active_load_balance_score", summary.get("avg_load_balance_score", 0.0))),
                    "active_load_balance_score": summary.get("mec_load_fairness", summary.get("active_load_balance_score", summary.get("avg_load_balance_score", 0.0))),
                    "total_energy": env_stats.get("total_energy", 0.0),
                    "energy_per_resolved_task": energy_per_resolved_task(
                        {
                            "total_energy": env_stats.get("total_energy", 0.0),
                            "resolved_tasks": summary.get("resolved_tasks", 0.0),
                        }
                    ),
                    "energy_per_successful_task": energy_per_successful_task(
                        {
                            "total_energy": env_stats.get("total_energy", 0.0),
                            "completed_tasks": env_stats.get("completed_tasks", 0.0),
                        }
                    ),
                    "load_balance_variance": summary.get("load_balance_variance", 0.0),
                    "load_balance_coefficient": summary.get("load_balance_coefficient", summary.get("mec_load_fairness", 0.0)),
                    "load_variance_sample_count": summary.get("load_variance_sample_count", 0),
                }
            )
            observations, _ = env.reset(seed=seed + step_idx + 1)
            episode_reward = 0.0
            episode_length = 0

        completed_steps = step_idx + 1
        if (
            completed_steps % log_interval_steps == 0
            or completed_steps == int(total_timesteps)
        ):
            logger.info(
                "Steps: %d/%d | Episodes: %d | CurrentEpReward: %.3f | "
                "RecentReward: %.3f | QLoss: %.6f | ParamLoss: %.6f | "
                "Epsilon: %.4f | Replay: %d",
                completed_steps,
                int(total_timesteps),
                episode_count,
                episode_reward,
                (
                    float(np.mean(recent_episode_rewards))
                    if recent_episode_rewards
                    else 0.0
                ),
                float(np.mean(recent_q_losses)) if recent_q_losses else 0.0,
                (
                    float(np.mean(recent_param_losses))
                    if recent_param_losses
                    else 0.0
                ),
                float(algo.current_epsilon()),
                len(replay),
            )

        if completed_steps % save_interval == 0:
            periodic_path = save_dir / f"checkpoint_{completed_steps}.pt"
            algo.save(periodic_path)
            validation_result = evaluate_pdqn_policy(
                algorithm=algo,
                objective=objective,
                config=config,
                episodes=validation_episodes,
                seed=seed,
                max_steps=max_steps,
            )
            validation_reward = float(
                validation_result.get("mean_reward", -float("inf"))
            )
            training_evaluation_records.append(
                {
                    "total_steps": completed_steps,
                    "eval_mean_reward": validation_reward,
                    "eval_std_reward": float(
                        validation_result.get("std_reward", 0.0)
                    ),
                    "eval_episodes": validation_episodes,
                    "best_model_metric": "reward",
                    "best_model_score": validation_reward,
                }
            )
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                algo.save(best_checkpoint_path)
                logger.info(
                    "新的最佳模型 (reward): %.3f，保存至 %s",
                    validation_reward,
                    best_checkpoint_path,
                )
            write_training_history(completed_steps=completed_steps)
            logger.info("周期权重已保存: %s", periodic_path)

    algo.save(checkpoint_path)
    if (
        not training_evaluation_records
        or int(training_evaluation_records[-1].get("total_steps", -1))
        != int(total_timesteps)
    ):
        validation_result = evaluate_pdqn_policy(
            algorithm=algo,
            objective=objective,
            config=config,
            episodes=validation_episodes,
            seed=seed,
            max_steps=max_steps,
        )
        final_validation_reward = float(
            validation_result.get("mean_reward", 0.0)
        )
        training_evaluation_records.append(
            {
                "total_steps": int(total_timesteps),
                "eval_mean_reward": final_validation_reward,
                "eval_std_reward": float(validation_result.get("std_reward", 0.0)),
                "eval_episodes": validation_episodes,
                "best_model_metric": "reward",
                "best_model_score": final_validation_reward,
            }
        )
        if final_validation_reward > best_validation_reward:
            best_validation_reward = final_validation_reward
            algo.save(best_checkpoint_path)
    if not best_checkpoint_path.exists():
        algo.save(best_checkpoint_path)
    write_training_history(completed_steps=int(total_timesteps))
    logger.info("最终权重已保存: %s", checkpoint_path)
    algo.load(best_checkpoint_path)

    result = evaluate_pdqn_policy(
        algorithm=algo,
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    write_training_history(
        evaluation_records=result.get("episode_metrics", []),
        summary={
            "mean_reward": float(result.get("mean_reward", 0.0)),
            "std_reward": float(result.get("std_reward", 0.0)),
        },
        completed_steps=int(total_timesteps),
    )
    logger.info(
        "训练与评估完成 | EvalReward: %.3f ± %.3f",
        float(result.get("mean_reward", 0.0)),
        float(result.get("std_reward", 0.0)),
    )
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    result["trained_timesteps"] = int(total_timesteps)
    result["checkpoint"] = str(best_checkpoint_path)
    result["training_history"] = str(history_path)
    result["training_log"] = str(log_path)
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
    seed_offset: int = EVALUATION_SEED_OFFSET,
    collect_task_trace: bool = False,
) -> Dict:
    evaluation_config = dict(config)
    if collect_task_trace:
        evaluation_config["enable_task_trace"] = True
    env = build_env_for_objective(
        objective,
        evaluation_config,
        seed=seed,
        max_steps=max_steps,
    )
    rewards: List[float] = []
    summaries: List[Dict] = []
    action_batches: List[np.ndarray] = []
    task_trace: List[Dict] = []

    for episode_idx in range(episodes):
        policy.begin_episode(env)
        env.reset(seed=seed + seed_offset + episode_idx)
        done = False
        episode_reward = 0.0

        while not done:
            actions = policy.select_actions(env)
            action_batches.append(np.asarray(actions, dtype=np.float32).copy())
            _, reward, terminated, truncated, _ = env.step(
                actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += float(reward)
            done = terminated or truncated

        rewards.append(episode_reward)
        summaries.append(env.get_stats_summary())
        if collect_task_trace:
            episode_seed = seed + seed_offset + episode_idx
            for record in env.get_task_trace_records(include_pending=True):
                task_trace.append({
                    "episode": episode_idx + 1,
                    "episode_seed": episode_seed,
                    **record,
                })

    extra = {}
    if isinstance(policy, SimpleHeuristicPolicy):
        extra["selected_offload"] = policy.offload_ratio
    extra.update(
        action_diagnostics(
            action_batches,
            float(config.get("min_effective_offload_ratio", EnvConfig.min_effective_offload_ratio)),
        )
    )
    result = summarize_results(
        policy.name,
        rewards,
        summaries,
        extra=extra,
        is_system=False,
    )
    if collect_task_trace:
        result["task_trace"] = task_trace
    return result


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
            seed_offset=HEURISTIC_TUNING_SEED_OFFSET,
        )
        candidates.append(result)

    best_index = choose_best_index(
        [selection_score(candidate, selection_metric_name) for candidate in candidates],
        higher_is_better=True,
    )
    selected_offload = float(
        candidates[best_index].get("selected_offload", 0.0)
    )
    best = evaluate_policy(
        policy=SimpleHeuristicPolicy(strategy, selected_offload),
        objective=objective,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        seed_offset=EVALUATION_SEED_OFFSET,
    )
    best["method"] = strategy
    best["display_name"] = pretty_method_name(strategy, is_system=False)
    best["selected_offload"] = selected_offload
    best["method_variant"] = best["method"] + f"(best_offload={selected_offload:.2f})"
    return best


def trainer_class_for_objective(objective: str):
    return HANMAPPOTrainer


def system_trainer_class_for_config(objective: str, config_data: Dict):
    algorithm = str(config_data.get("algorithm", "mappo"))
    if algorithm == "attn_mappo":
        return AttentionMAPPOTrainer
    if algorithm == "han_attn":
        return HANCandidateAttentionMAPPOTrainer
    return trainer_class_for_objective(objective)


def baseline_config_data(config_data: Dict, algorithm: str) -> Dict:
    """Copy run config and set the algorithm owned by a baseline."""
    sanitized = dict(config_data)
    sanitized["algorithm"] = algorithm
    return sanitized


class NoHANTrainerMixin:
    """Mixin for MAPPO ablations that bypass the HAN graph encoder."""

    def _init_environment(self):
        super()._init_environment()
        # 无 HAN 消融仍需保留候选卫星的可区分原始上下文，否则 HybridActor
        # 会收到全零候选 token，使所有候选切换 logit 完全相同。
        self.han_out_dim = SATELLITE_CONTEXT_FEATURE_DIM
        self.obs_dim = self.raw_obs_dim
        self.global_state_dim = self.num_agents * self.obs_dim
        self.logger.info(
            "  - HAN ablation: raw user observations + raw candidate satellite context"
        )
        self.logger.info(f"  - No-HAN observation dim: {self.obs_dim}")

    def _init_han_encoder(self):
        self.han_encoder = nn.Identity().to(self.device)

    def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observations = self.env._get_observation().astype(np.float32, copy=False)
        available_actions = self.env.get_handover_action_mask(
            self.max_candidates,
            apply_pre_handover_gate=False,
        )
        candidate_sat_ids = np.full(
            (self.num_agents, self.max_candidates),
            -1,
            dtype=np.int64,
        )
        for uid, user in enumerate(self.env.user_manager.users):
            visible_sats = self.env._get_handover_candidates(user)
            valid_count = min(len(visible_sats), self.max_candidates)
            if valid_count > 0:
                candidate_sat_ids[uid, :valid_count] = [
                    sat_info.sat_id for sat_info in visible_sats[:valid_count]
                ]

        satellite_embeddings = build_satellite_context_features(
            self.env,
            self.num_agents,
        )
        return observations, satellite_embeddings, available_actions, candidate_sat_ids


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
        config.log_path = str(PROJECT_ROOT / "results" / "logs")
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
    trainer_cls = system_trainer_class_for_config(objective, config_data)
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
    method_name = str(config_data.get("exp_name", checkpoint.parent.name or "system"))
    return evaluate_mappo_checkpoint_with_trainer(
        checkpoint=checkpoint,
        config_data=config_data,
        episodes=episodes,
        device=device,
        max_steps=max_steps,
        trainer_cls=system_trainer_class_for_config(objective, config_data),
        method_name=method_name,
        is_system=True,
    )


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
    schema_contract = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
    }
    for key, required_version in schema_contract.items():
        actual_version = int(checkpoint_payload.get(key, 0))
        if actual_version != int(required_version):
            raise ValueError(
                f"Checkpoint {checkpoint} 的 {key}={actual_version}，"
                f"当前正式评估要求 {required_version}；禁止混用不兼容结果。"
            )
    trainer.total_steps = checkpoint_payload.get("total_steps", trainer.total_steps)
    trainer.episodes = checkpoint_payload.get("episodes", trainer.episodes)
    trainer.best_reward = checkpoint_payload.get("best_reward", trainer.best_reward)
    if "best_model_metric" in checkpoint_payload:
        trainer.config.best_model_metric = checkpoint_payload["best_model_metric"]
    trainer.mappo.actor.load_state_dict(checkpoint_payload["actor_state_dict"])
    trainer.mappo.critic.load_state_dict(checkpoint_payload["critic_state_dict"])
    if "han_state_dict" in checkpoint_payload and hasattr(trainer.han_encoder, "load_state_dict"):
        trainer.han_encoder.load_state_dict(checkpoint_payload["han_state_dict"], strict=False)
    trainer.mappo.actor.eval()
    trainer.mappo.critic.eval()
    if hasattr(trainer.han_encoder, "eval"):
        trainer.han_encoder.eval()

    rewards: List[float] = []
    summaries: List[Dict] = []
    action_batches: List[np.ndarray] = []

    for episode_idx in range(episodes):
        trainer.env.reset(
            seed=(
                int(config.seed) + EVALUATION_SEED_OFFSET + episode_idx
                if config.seed is not None
                else None
            )
        )
        observations, satellite_embeddings, available_actions, candidate_sat_ids = trainer._encode_graph_state()
        available_actions = trainer._apply_pre_handover_action_mask(
            available_actions,
            trainer.env.get_pre_handover_mask(),
        )
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                actions, _, _ = trainer.mappo.act(
                    observations,
                    available_actions,
                    satellite_embeddings=satellite_embeddings,
                    candidate_sat_ids=candidate_sat_ids,
                    deterministic=True,
                )

            env_actions = trainer._process_actions(actions)
            action_batches.append(np.asarray(env_actions, dtype=np.float32).copy())
            _, reward, terminated, truncated, _ = trainer.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += float(reward)
            done = terminated or truncated

            if not done:
                observations, satellite_embeddings, available_actions, candidate_sat_ids = trainer._encode_graph_state()
                available_actions = trainer._apply_pre_handover_action_mask(
                    available_actions,
                    trainer.env.get_pre_handover_mask(),
                )

        rewards.append(episode_reward)
        summaries.append(trainer.env.get_stats_summary())

    extra = action_diagnostics(
        action_batches,
        float(config_data.get("min_effective_offload_ratio", EnvConfig.min_effective_offload_ratio)),
    )
    return summarize_results(method_name, rewards, summaries, extra=extra, is_system=is_system)


def _train_and_evaluate_mappo_variant(
    *,
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    reuse_checkpoint_if_available: bool,
    method_name: str,
    algorithm_name: str,
    trainer_cls,
) -> Dict:
    """Shared lifecycle for MAPPO baselines and ablations."""
    variant_config = baseline_config_data(config_data, algorithm_name)
    save_dir = output_dir / "learned_baselines" / method_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "training_history.json"

    checkpoint = (
        find_existing_checkpoint(save_dir)
        if reuse_checkpoint_if_available
        else None
    )
    if checkpoint is not None:
        result = evaluate_mappo_checkpoint_with_trainer(
            checkpoint=checkpoint,
            config_data=variant_config,
            episodes=episodes,
            device=resolve_device(device),
            max_steps=max_steps,
            trainer_cls=trainer_cls,
            method_name=method_name,
            is_system=False,
        )
        result["source"] = f"{method_name}_checkpoint_eval"
    else:
        config = train_config_from_dict(
            variant_config,
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
        checkpoint = find_existing_checkpoint(save_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"{method_name} training did not produce a checkpoint in {save_dir}"
            )
        result = evaluate_mappo_checkpoint_with_trainer(
            checkpoint=checkpoint,
            config_data=asdict(config),
            episodes=episodes,
            device=resolve_device(device),
            max_steps=max_steps,
            trainer_cls=trainer_cls,
            method_name=method_name,
            is_system=False,
        )
        result["trained_timesteps"] = int(total_timesteps)

    result["checkpoint"] = str(checkpoint)
    if history_path.exists():
        result["training_history"] = str(history_path)
    return result


def train_and_evaluate_no_han_mappo(
    config_data: Dict,
    objective: str,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    reuse_checkpoint_if_available: bool = False,
) -> Dict:
    return _train_and_evaluate_mappo_variant(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        reuse_checkpoint_if_available=reuse_checkpoint_if_available,
        method_name="mappo_no_han",
        algorithm_name="mappo",
        trainer_cls=no_han_trainer_class_for_objective(objective),
    )


def train_and_evaluate_han_mappo(
    config_data: Dict,
    objective: str,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    reuse_checkpoint_if_available: bool = False,
) -> Dict:
    return _train_and_evaluate_mappo_variant(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        reuse_checkpoint_if_available=reuse_checkpoint_if_available,
        method_name="han_mappo",
        algorithm_name="mappo",
        trainer_cls=trainer_class_for_objective(objective),
    )


def train_and_evaluate_attention_mappo(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    reuse_checkpoint_if_available: bool = False,
) -> Dict:
    return _train_and_evaluate_mappo_variant(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        reuse_checkpoint_if_available=reuse_checkpoint_if_available,
        method_name="attn_mappo",
        algorithm_name="attn_mappo",
        trainer_cls=AttentionMAPPOTrainer,
    )


def train_and_evaluate_han_attn_mappo(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    reuse_checkpoint_if_available: bool = False,
) -> Dict:
    return _train_and_evaluate_mappo_variant(
        config_data=config_data,
        output_dir=output_dir,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        early_stop_patience=early_stop_patience,
        reuse_checkpoint_if_available=reuse_checkpoint_if_available,
        method_name="han_attn",
        algorithm_name="han_attn",
        trainer_cls=HANCandidateAttentionMAPPOTrainer,
    )


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
    action_batches: List[np.ndarray] = []

    for episode_idx in range(episodes):
        observations, _, masks = trainer._reset_encoded_env(
            seed=int(config.seed) + EVALUATION_SEED_OFFSET + episode_idx
        )
        done = False
        episode_reward = 0.0
        while not done:
            env_actions, _, _ = trainer._select_eval_action(observations, masks)
            action_batches.append(np.asarray(env_actions, dtype=np.float32).copy())
            _, reward, terminated, truncated, _ = trainer.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            episode_reward += trainer._scalar_reward(reward)
            done = terminated or truncated
            if not done:
                observations, _, masks, *_ = trainer._encode_graph_state()
                masks = trainer._apply_pre_handover_action_mask(
                    masks,
                    trainer.env.get_pre_handover_mask(),
                )
        rewards.append(episode_reward)
        summaries.append(trainer.env.get_stats_summary())

    extra = action_diagnostics(
        action_batches,
        float(config_data.get("min_effective_offload_ratio", EnvConfig.min_effective_offload_ratio)),
    )
    return summarize_results(method_name, rewards, summaries, extra=extra, is_system=False)


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
    pretrained_han_path: Optional[Path],
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
    if pretrained_han_path is None:
        raise ValueError(
            f"{method_name} requires a trained system checkpoint for its HAN encoder"
        )
    config.pretrained_han_path = str(pretrained_han_path)
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
    pretrained_han_path: Optional[Path] = None,
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
        pretrained_han_path=pretrained_han_path,
    )


def train_and_evaluate_han_pdqn_baseline(
    config_data: Dict,
    output_dir: Path,
    device: str,
    episodes: int,
    max_steps: Optional[int],
    total_timesteps: int,
    early_stop_patience: int,
    pretrained_han_path: Optional[Path] = None,
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
        pretrained_han_path=pretrained_han_path,
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
    best_record = derive_paper_metrics(normalize_load_balance_metrics(best_record))
    handover_success_rate = float(best_record.get("handover_success_rate", 0.0))
    service_continuity_rate = float(best_record.get("service_continuity_rate", 0.0))
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
        "avg_success_delay": float(best_record.get("avg_success_delay", 0.0)),
        "p95_success_delay": float(best_record.get("p95_success_delay", 0.0)),
        "total_energy": float(best_record.get("total_energy", 0.0)),
        "handover_success_rate": handover_success_rate,
        "handover_failure_rate": float(best_record.get("handover_failure_rate", max(0.0, 1.0 - handover_success_rate))),
        "forced_termination_rate": float(best_record.get("forced_termination_rate", max(0.0, 1.0 - service_continuity_rate))),
        "total_user_seconds": float(best_record.get("total_user_seconds", 0.0)),
        "blocked_user_seconds": float(best_record.get("blocked_user_seconds", 0.0)),
        "handover_interruption_seconds": float(best_record.get("handover_interruption_seconds", 0.0)),
        "service_interruption_seconds": float(best_record.get("service_interruption_seconds", 0.0)),
        "blocked_time_ratio": float(best_record.get("blocked_time_ratio", 0.0)),
        "total_handovers": float(best_record.get("total_handovers", 0.0)),
        "handover_attempts": float(best_record.get("handover_attempts", best_record.get("total_handovers", 0.0))),
        "handover_committed": float(best_record.get("handover_committed", best_record.get("successful_handovers", 0.0))),
        "handover_aborted": float(best_record.get("handover_aborted", best_record.get("failed_handovers", 0.0))),
        "handover_radio_failures": float(best_record.get("handover_radio_failures", 0.0)),
        "migration_rejections": float(best_record.get("migration_rejections", 0.0)),
        "reconnection_attempts": float(best_record.get("reconnection_attempts", 0.0)),
        "reconnections": float(best_record.get("reconnections", 0.0)),
        "failed_handovers": float(best_record.get("failed_handovers", 0.0)),
        "service_continuity_rate": service_continuity_rate,
        "service_availability_rate": float(best_record.get("service_availability_rate", service_continuity_rate)),
        "task_completion_rate": float(best_record.get("task_completion_rate", 0.0)),
        "task_success_rate": float(best_record.get("task_success_rate", best_record.get("task_completion_rate", 0.0))),
        "task_failure_rate": float(best_record.get("task_failure_rate", best_record.get("deadline_violation_rate", 0.0))),
        "task_settlement_rate": float(best_record.get("task_settlement_rate", best_record.get("task_resolution_rate", 0.0))),
        "task_resolution_rate": float(best_record.get("task_resolution_rate", 0.0)),
        "pending_task_rate": float(best_record.get("pending_task_rate", 0.0)),
        "handover_frequency": float(best_record.get("handover_frequency", compute_handover_frequency(best_record))),
        "handovers_per_user_minute": float(best_record.get("handovers_per_user_minute", 0.0)),
        "load_balance_variance": float(best_record.get("load_balance_variance", 0.0)),
        "load_balance_coefficient": float(best_record.get("load_balance_coefficient", best_record.get("mec_load_fairness", 0.0))),
        "load_variance_sample_count": float(best_record.get("load_variance_sample_count", 0.0)),
        "mec_load_fairness": float(best_record.get("mec_load_fairness", best_record.get("active_load_balance_score", best_record.get("avg_load_balance_score", 0.0)))),
        "jain_mec_load_fairness": float(best_record.get("jain_mec_load_fairness", 0.0)),
        "active_mec_load_fairness": float(best_record.get("active_mec_load_fairness", best_record.get("mec_load_fairness", 0.0))),
        "reachable_jain_mec_load_fairness": float(best_record.get("reachable_jain_mec_load_fairness", best_record.get("jain_mec_load_fairness", 0.0))),
        "active_load_balance_score": float(best_record.get("mec_load_fairness", best_record.get("active_load_balance_score", best_record.get("avg_load_balance_score", 0.0)))),
        "avg_load_balance_score": float(best_record.get("mec_load_fairness", best_record.get("active_load_balance_score", best_record.get("avg_load_balance_score", 0.0)))),
        "resolved_tasks": float(best_record.get("resolved_tasks", 0.0)),
        "pending_tasks": float(best_record.get("pending_tasks", 0.0)),
        "total_tasks": float(best_record.get("total_tasks", 0.0)),
        "completed_tasks": float(best_record.get("completed_tasks", 0.0)),
        "deadline_violations": float(best_record.get("deadline_violations", 0.0)),
        "failed_tasks": float(best_record.get("failed_tasks", 0.0)),
        "deadline_violation_rate": compute_deadline_violation_rate(best_record),
        "energy_per_successful_task": float(energy_per_successful_task(best_record)),
        "energy_per_resolved_task": float(energy_per_resolved_task(best_record)),
        "episode_metrics": [],
        "training_history": str(history_path),
        "source": f"training_history_best_{selection_metric_name}",
    }
    return config_data, derive_paper_metrics(result)


def save_results_json(output_dir: Path, payload: Dict) -> Path:
    path = output_dir / "comparison_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        payload_to_save = dict(payload)
        if "methods" in payload_to_save:
            payload_to_save["methods"] = [
                ensure_action_diagnostic_fields(method)
                for method in payload_to_save.get("methods", [])
            ]
        json.dump(payload_to_save, handle, ensure_ascii=False, indent=2)
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
        "avg_success_delay",
        "p95_success_delay",
        "total_energy",
        "handover_success_rate",
        "handover_failure_rate",
        "forced_termination_rate",
        "total_user_seconds",
        "blocked_user_seconds",
        "blocked_time_ratio",
        "handover_interruption_seconds",
        "service_interruption_seconds",
        "total_handovers",
        "handover_attempts",
        "handover_committed",
        "handover_aborted",
        "handover_radio_failures",
        "migration_rejections",
        "reconnection_attempts",
        "reconnections",
        "failed_handovers",
        "service_continuity_rate",
        "service_availability_rate",
        "task_completion_rate",
        "task_success_rate",
        "task_failure_rate",
        "task_settlement_rate",
        "task_resolution_rate",
        "pending_task_rate",
        "deadline_violation_rate",
        "failed_tasks",
        "handover_frequency",
        "handovers_per_user_minute",
        "load_balance_variance",
        "load_balance_coefficient",
        "load_variance_sample_count",
        "mec_load_fairness",
        "jain_mec_load_fairness",
        "active_mec_load_fairness",
        "reachable_jain_mec_load_fairness",
        "active_load_balance_score",
        "avg_load_balance_score",
        "resolved_tasks",
        "pending_tasks",
        "total_tasks",
        "completed_tasks",
        "deadline_violations",
        "energy_per_resolved_task",
        "energy_per_successful_task",
        *ACTION_DIAGNOSTIC_KEYS,
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
                "avg_success_delay": float(record.get("avg_success_delay", 0.0)),
                "p95_success_delay": float(record.get("p95_success_delay", 0.0)),
                "total_energy": float(record.get("total_energy", 0.0)),
                "handover_success_rate": float(record.get("handover_success_rate", 0.0)),
                "handover_failure_rate": float(record.get("handover_failure_rate", 0.0)),
                "forced_termination_rate": float(record.get("forced_termination_rate", 0.0)),
                "total_user_seconds": float(record.get("total_user_seconds", 0.0)),
                "blocked_user_seconds": float(record.get("blocked_user_seconds", 0.0)),
                "blocked_time_ratio": float(record.get("blocked_time_ratio", 0.0)),
                "handover_interruption_seconds": float(record.get("handover_interruption_seconds", 0.0)),
                "service_interruption_seconds": float(record.get("service_interruption_seconds", 0.0)),
                "total_handovers": float(record.get("total_handovers", 0.0)),
                "failed_handovers": float(record.get("failed_handovers", 0.0)),
                "service_continuity_rate": float(record.get("service_continuity_rate", 0.0)),
                "service_availability_rate": float(record.get("service_availability_rate", 0.0)),
                "task_completion_rate": float(record.get("task_completion_rate", 0.0)),
                "task_success_rate": float(record.get("task_success_rate", record.get("task_completion_rate", 0.0))),
                "task_failure_rate": float(record.get("task_failure_rate", record.get("deadline_violation_rate", 0.0))),
                "task_settlement_rate": float(record.get("task_settlement_rate", record.get("task_resolution_rate", 0.0))),
                "task_resolution_rate": float(record.get("task_resolution_rate", 0.0)),
                "pending_task_rate": float(record.get("pending_task_rate", 0.0)),
                "handover_frequency": float(record.get("handover_frequency", compute_handover_frequency(record))),
                "handovers_per_user_minute": float(record.get("handovers_per_user_minute", 0.0)),
                "load_balance_variance": float(record.get("load_balance_variance", 0.0)),
                "load_balance_coefficient": float(record.get("load_balance_coefficient", record.get("mec_load_fairness", 0.0))),
                "load_variance_sample_count": float(record.get("load_variance_sample_count", 0.0)),
                "mec_load_fairness": float(record.get("mec_load_fairness", record.get("active_load_balance_score", record.get("avg_load_balance_score", 0.0)))),
                "jain_mec_load_fairness": float(record.get("jain_mec_load_fairness", 0.0)),
                "active_load_balance_score": float(record.get("mec_load_fairness", record.get("active_load_balance_score", record.get("avg_load_balance_score", 0.0)))),
                "avg_load_balance_score": float(record.get("mec_load_fairness", record.get("active_load_balance_score", record.get("avg_load_balance_score", 0.0)))),
                "deadline_violation_rate": float(record.get("deadline_violation_rate", 0.0)),
                "resolved_tasks": float(record.get("resolved_tasks", 0.0)),
                "completed_tasks": float(record.get("completed_tasks", 0.0)),
                "deadline_violations": float(record.get("deadline_violations", 0.0)),
                "energy_per_successful_task": float(record.get("energy_per_successful_task", energy_per_successful_task(record))),
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
    if metric_key.endswith("_rate") or metric_key == "blocked_time_ratio":
        return 100.0
    return 1.0


def metric_display_value(value: float, metric_key: str) -> str:
    if metric_key.endswith("_rate"):
        return f"{value:.1f}%"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def paper_metric_scale(metric_key: str) -> float:
    return unified_metric_scale(metric_key)


def paper_metric_value(method: Dict, metric_key: str) -> float:
    if (
        metric_key in SUCCESS_DEPENDENT_METRICS
        and float(method.get("completed_tasks", 0.0) or 0.0) <= 0.0
    ):
        return float("nan")
    try:
        value = float(method.get(metric_key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value * paper_metric_scale(metric_key)


def paper_metric_samples(method: Dict, metric_key: str) -> np.ndarray:
    records = method.get("seed_metrics", [])
    if not records:
        return np.array([], dtype=float)
    scale = paper_metric_scale(metric_key)
    values: List[float] = []
    for record in records:
        if (
            metric_key in SUCCESS_DEPENDENT_METRICS
            and float(record.get("completed_tasks", 0.0) or 0.0) <= 0.0
        ):
            continue
        try:
            value = float(record.get(metric_key, float("nan"))) * scale
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def method_tick_label(method: Dict) -> str:
    label = str(method.get("display_name", method.get("method", "")))
    replacements = {
        "HAN+MAPPO": "HAN+\nMAPPO",
        "Joint Greedy": "Joint\nGreedy",
        "Min-Distance": "Min-\nDistance",
        "Full-Local": "Full-\nLocal",
        "MAPPO": "MAPPO",
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
    values = []
    lower_errors = []
    upper_errors = []
    for method in ordered:
        samples = paper_metric_samples(method, metric_key)
        if len(samples) > 0:
            mean, low, high = bootstrap_mean_ci(samples)
        else:
            mean = paper_metric_value(method, metric_key)
            low = high = mean
        values.append(mean)
        if np.isfinite(mean):
            lower_errors.append(max(mean - low, 0.0))
            upper_errors.append(max(high - mean, 0.0))
        else:
            lower_errors.append(0.0)
            upper_errors.append(0.0)

    positions = np.arange(len(ordered), dtype=float)
    plot_values = [value if np.isfinite(value) else 0.0 for value in values]
    colors = [
        (
            styles[str(method.get("method", ""))].get("color", PAPER_COLORS["muted"])
            if np.isfinite(value)
            else "#D9D9D9"
        )
        for method, value in zip(ordered, values)
    ]
    value_top = max(
        [
            value + error
            for value, error in zip(values, upper_errors)
            if np.isfinite(value)
        ]
        + [0.0]
    )
    y_top = value_top * 1.24 if value_top > 0.0 else 1.0
    label_offset = y_top * 0.025
    bars = ax.bar(
        positions,
        plot_values,
        width=0.72,
        color=colors,
        edgecolor=PAPER_COLORS["dark"],
        linewidth=1.0,
        alpha=0.92,
    )
    cap_half_width = 0.08
    for position, value, lower_error, upper_error in zip(
        positions,
        values,
        lower_errors,
        upper_errors,
    ):
        if lower_error <= 0.0 and upper_error <= 0.0:
            continue
        low = value - lower_error
        high = value + upper_error
        ax.vlines(position, low, high, color=PAPER_COLORS["dark"], linewidth=1.1)
        ax.hlines(
            [low, high],
            position - cap_half_width,
            position + cap_half_width,
            color=PAPER_COLORS["dark"],
            linewidth=1.1,
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
        if not np.isfinite(value):
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_offset,
                "N/A",
                va="bottom",
                ha="center",
                fontsize=10 if compact else 10.5,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + upper_errors[index] + label_offset, y_top * 0.98),
            metric_display_value(value, metric_key),
            va="bottom",
            ha="center",
            fontsize=10 if compact else 10.5,
            rotation=90 if compact else 0,
        )

    ax.set_xticks(positions, labels=labels)
    ax.tick_params(axis="x", rotation=25 if compact else 28)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#BDBDBD")
    if values:
        ax.set_ylim(0.0, y_top + 1e-12)
    style_axes_frame(ax)


def load_variance_samples_for_plot(method: Dict) -> List[float]:
    raw_samples = method.get("load_variance_samples")
    if isinstance(raw_samples, list) and raw_samples:
        samples = []
        for value in raw_samples:
            try:
                sample = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(sample):
                samples.append(float(np.clip(sample, 0.0, 0.25)))
        if samples:
            return samples

    cdf_points = method.get("load_variance_cdf")
    if isinstance(cdf_points, list) and cdf_points:
        samples = []
        for point in cdf_points:
            if isinstance(point, dict) and "x" in point:
                try:
                    sample = float(point["x"])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(sample):
                    samples.append(float(np.clip(sample, 0.0, 0.25)))
        if samples:
            return samples
    return []


def draw_load_variance_cdf_panel(ax, methods: Sequence[Dict], compact: bool = False) -> None:
    ordered = order_methods(methods)
    styles = build_method_styles(ordered)
    plotted = False
    for method in ordered:
        samples = load_variance_samples_for_plot(method)
        if not samples:
            continue
        points = empirical_cdf(samples)
        if not points:
            continue
        x_values = [points[0]["x"], *[point["x"] for point in points]]
        y_values = [0.0, *[point["cdf"] for point in points]]
        style = styles[str(method.get("method", ""))]
        ax.step(
            x_values,
            y_values,
            where="post",
            color=style["color"],
            linestyle=style.get("linestyle", "-"),
            linewidth=2.0 if compact else 2.3,
            label=method.get("display_name", method.get("method", "")),
        )
        plotted = True

    ax.set_title("Load Variance CDF")
    ax.set_xlabel("Load Variance")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(left=0.0)
    if plotted:
        ax.legend(fontsize=8.5 if compact else 9.5, ncol=1 if compact else 2)
    else:
        ax.text(
            0.5,
            0.5,
            "No load variance data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=PAPER_COLORS["muted"],
        )
    style_axes_frame(ax)


def draw_reward_curve_panel(
    ax,
    history_path: Optional[Path],
    methods: Sequence[Dict],
    window: int,
    compact: bool = False,
    output_dir: Optional[Path] = None,
) -> bool:
    ordered = order_methods(methods)
    styles = build_method_styles(ordered)
    plotted = False
    for method in ordered:
        paths = method_training_history_paths(
            method,
            output_dir=output_dir,
            primary_history_path=history_path,
        )
        steps, mean_reward, lower, upper = aggregate_reward_curves(paths, window)
        if len(steps) == 0:
            continue
        plotted = True
        style = styles[str(method.get("method", ""))]
        color = style.get("color", PAPER_COLORS["muted"])
        linewidth = 3.0 if method.get("is_system") else (2.2 if compact else 2.4)
        ax.plot(
            steps,
            mean_reward,
            color=color,
            linestyle="-",
            linewidth=linewidth,
            alpha=0.98,
            zorder=3,
            label=method.get("display_name", method.get("method", "")),
        )
        if len(paths) > 1:
            ax.fill_between(
                steps,
                lower,
                upper,
                color=color,
                alpha=0.16 if method.get("is_system") else 0.10,
                linewidth=0,
                zorder=2,
            )

    if not plotted:
        return False

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Learning Algorithm Reward Convergence")
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(
        loc="lower right" if compact else "best",
        fontsize=10.5 if compact else 11.5,
        ncol=1 if compact else 2,
    )
    style_axes_frame(ax)
    return True


def plot_method_comparison(
    methods: Sequence[Dict],
    output_dir: Path,
    output_suffix: str = "",
) -> Optional[Path]:
    if not methods:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=220)
    for axis, (metric_key, title, ylabel) in zip(axes.flatten(), CORE_BAR_METRICS):
        draw_metric_bar_panel(axis, methods, metric_key=metric_key, title=title, ylabel=ylabel)
    if len(CORE_BAR_METRICS) < len(axes.flatten()):
        draw_load_variance_cdf_panel(axes.flatten()[len(CORE_BAR_METRICS)], methods)

    fig.suptitle(
        f"{system_display_name(methods)} vs. Baselines: Core Metrics",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(
        fig,
        figure_output_path(output_dir, "method_comparison.png", output_suffix),
    )


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
    usable_records = [
        record
        for record in training
        if (
            "total_steps" in record
            and metric_key in record
            and not bool(record.get("partial_episode", False))
        )
    ]
    totals = np.array(
        [
            max(float(record.get("total_tasks", 0.0)), 1.0)
            for record in usable_records
        ],
        dtype=float,
    )
    if len(totals) != len(values):
        return steps, values
    config_data = history.get("config", {}) if isinstance(history, dict) else {}
    num_users = max(float(config_data.get("num_users", 1.0)), 1.0)
    # reward breakdown 已经按用户均值累计；恢复系统总贡献后再除任务数。
    return steps, values * num_users / totals


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

    metric_specs = [spec for spec in reward_component_step_metrics_for_history(history_path) if spec[0] != "mean_reward"]
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
    return save_figure(fig, output_dir / "reward_components_per_task_vs_steps.png")


def plot_step_metric_curves(history_path: Optional[Path], output_dir: Path, window: int) -> List[Path]:
    paths: List[Path] = []
    qos_path = plot_step_metric_group(
        history_path,
        output_dir,
        window,
        TRAINING_QOS_STEP_METRICS,
        "training_qos_metrics_vs_steps.png",
        "Training QoS Metrics vs. Steps",
    )
    if qos_path is not None:
        paths.append(qos_path)

    reward_component_path = plot_step_metric_group(
        history_path,
        output_dir,
        window,
        reward_component_step_metrics_for_history(history_path),
        "reward_components_vs_steps.png",
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


def learned_methods_with_episode_metrics(methods: Sequence[Dict]) -> List[Dict]:
    """Exclude heuristic outliers from learned-policy uncertainty figures."""
    return [
        method
        for method in methods_with_episode_metrics(methods)
        if method.get("is_system")
        or str(method.get("method", "")) in {
            "dqn",
            "maddpg",
            "pdqn",
            "han_mappo",
            "mappo_no_han",
            "attn_mappo",
            "han_attn",
            "han_maddpg",
            "han_pdqn",
        }
    ]


def plot_additional_metric_curves(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    """Plot scenario aggregates instead of connecting independent episodes."""
    plottable = learned_methods_with_episode_metrics(methods)
    if not plottable:
        return None

    styles = build_method_styles(plottable)
    ncols = 2
    nrows = int(np.ceil(len(ADDITIONAL_EPISODE_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), dpi=220)
    axes = np.asarray(axes).flatten()

    for axis, (metric_key, title) in zip(axes, ADDITIONAL_EPISODE_METRICS):
        labels = []
        means = []
        lower_errors = []
        upper_errors = []
        colors = []
        for method_index, method in enumerate(plottable):
            records = method.get("episode_metrics", [])
            values = np.asarray(
                [
                    float(record.get(metric_key, 0.0)) * metric_scale(metric_key)
                    for record in records
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            mean, low, high = bootstrap_mean_ci(
                values,
                seed=20260731 + method_index,
            )
            style = styles[str(method.get("method", ""))]
            labels.append(method.get("display_name", method.get("method", "")))
            means.append(mean)
            lower_errors.append(max(mean - low, 0.0))
            upper_errors.append(max(high - mean, 0.0))
            colors.append(style["color"])

        positions = np.arange(len(labels), dtype=float)
        axis.errorbar(
            means,
            positions,
            xerr=np.asarray([lower_errors, upper_errors]),
            fmt="none",
            ecolor=PAPER_COLORS["dark"],
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )
        axis.scatter(
            means,
            positions,
            c=colors,
            s=58,
            edgecolors=PAPER_COLORS["dark"],
            linewidths=0.7,
            zorder=3,
        )
        axis.set_yticks(positions, labels=labels)
        axis.invert_yaxis()
        axis.set_title(title)
        if metric_key.endswith("_rate") or metric_key == "blocked_time_ratio":
            xlabel = "Rate (%)"
        elif metric_key == "handovers_per_user_minute":
            xlabel = "Handovers / User-Minute"
        else:
            xlabel = "Jain Index"
        axis.set_xlabel(f"{xlabel} (mean and 95% scenario CI)")
        style_axes_frame(axis)

    for axis in axes[len(ADDITIONAL_EPISODE_METRICS):]:
        axis.axis("off")

    fig.suptitle(
        "Learned Algorithms: Aggregate Results over Independent Evaluation Scenarios",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output_dir / "additional_metrics_episode_comparison.png")


def plot_paired_advantage_over_baselines(
    methods: Sequence[Dict],
    output_dir: Path,
) -> Optional[Path]:
    """Show paired HAN+MAPPO advantage under identical evaluation scenarios."""
    plottable = learned_methods_with_episode_metrics(methods)
    reference = next((method for method in plottable if method.get("is_system")), None)
    comparisons = [method for method in plottable if method is not reference]
    if reference is None or not comparisons:
        return None
    reference_name = str(
        reference.get("display_name", reference.get("method", "System"))
    )

    reference_by_episode = {
        int(record.get("episode", index + 1)): record
        for index, record in enumerate(reference.get("episode_metrics", []))
    }
    styles = build_method_styles(comparisons)
    ncols = 2
    nrows = int(np.ceil(len(ADDITIONAL_EPISODE_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), dpi=220)
    axes = np.asarray(axes).flatten()

    for metric_index, (axis, (metric_key, title)) in enumerate(
        zip(axes, ADDITIONAL_EPISODE_METRICS)
    ):
        labels = []
        means = []
        lows = []
        highs = []
        colors = []
        scale = metric_scale(metric_key)
        higher_is_better = HIGHER_IS_BETTER.get(metric_key, True)
        for method_index, method in enumerate(comparisons):
            advantages = []
            for index, record in enumerate(method.get("episode_metrics", [])):
                episode = int(record.get("episode", index + 1))
                reference_record = reference_by_episode.get(episode)
                if reference_record is None:
                    continue
                reference_value = float(reference_record.get(metric_key, 0.0)) * scale
                method_value = float(record.get(metric_key, 0.0)) * scale
                advantage = (
                    reference_value - method_value
                    if higher_is_better
                    else method_value - reference_value
                )
                if np.isfinite(advantage):
                    advantages.append(advantage)
            if not advantages:
                continue
            mean, low, high = bootstrap_mean_ci(
                advantages,
                seed=20260831 + metric_index * 100 + method_index,
            )
            labels.append(method.get("display_name", method.get("method", "")))
            means.append(mean)
            lows.append(max(mean - low, 0.0))
            highs.append(max(high - mean, 0.0))
            colors.append(styles[str(method.get("method", ""))]["color"])

        positions = np.arange(len(labels), dtype=float)
        axis.axvline(0.0, color=PAPER_COLORS["dark"], linestyle="--", linewidth=1.0)
        axis.errorbar(
            means,
            positions,
            xerr=np.asarray([lows, highs]),
            fmt="none",
            ecolor=PAPER_COLORS["dark"],
            elinewidth=1.2,
            capsize=3,
        )
        axis.scatter(
            means,
            positions,
            c=colors,
            s=58,
            edgecolors=PAPER_COLORS["dark"],
            linewidths=0.7,
            zorder=3,
        )
        axis.set_yticks(positions, labels=labels)
        axis.invert_yaxis()
        axis.set_title(title)
        unit = "percentage points" if metric_key.endswith("_rate") or metric_key == "blocked_time_ratio" else "metric units"
        axis.set_xlabel(f"{reference_name} paired advantage ({unit})")
        style_axes_frame(axis)

    fig.suptitle(
        f"Paired {reference_name} Advantage "
        f"(positive values favor {reference_name})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output_dir / "additional_metrics_paired_advantage.png")


def plot_delay_energy_tradeoff(
    methods: Sequence[Dict],
    output_dir: Path,
    output_suffix: str = "",
) -> Optional[Path]:
    ordered = order_methods(methods)
    points = [
        (
            method,
            paper_metric_value(method, "avg_success_delay"),
            paper_metric_value(method, "energy_per_successful_task"),
        )
        for method in ordered
    ]
    points = [
        (method, delay, energy)
        for method, delay, energy in points
        if np.isfinite(delay) and np.isfinite(energy)
    ]
    if not points:
        return None

    ordered = [method for method, _delay, _energy in points]
    styles = build_method_styles(ordered)
    x_values = np.asarray([delay for _method, delay, _energy in points], dtype=float)
    y_values = np.asarray([energy for _method, _delay, energy in points], dtype=float)

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
    ax.set_ylabel("Energy per Successful Task")
    ax.set_title("Delay-Energy Trade-off Across System and Heuristic Methods")
    style_axes_frame(ax)
    fig.tight_layout()
    return save_figure(
        fig,
        figure_output_path(output_dir, "delay_energy_tradeoff.png", output_suffix),
    )


def plot_success_continuity_scatter(
    methods: Sequence[Dict],
    output_dir: Path,
    output_suffix: str = "",
) -> Optional[Path]:
    ordered = order_methods(methods)
    if not ordered:
        return None

    styles = build_method_styles(ordered)
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    for method in ordered:
        style = styles[str(method.get("method", ""))]
        x_value = float(method.get("task_success_rate", method.get("task_completion_rate", 0.0))) * 100.0
        y_value = float(method.get("service_continuity_rate", 0.0)) * 100.0
        load_balance = float(
            method.get(
                "jain_mec_load_fairness",
                method.get("mec_load_fairness", 0.0),
            )
        )
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
    ax.set_xlabel("Task Success Rate (%)")
    ax.set_ylabel("Service Continuity Rate (%)")
    ax.set_title("Success-Continuity Trade-off (marker size: Jain fairness)")
    ax.set_xlim(left=0.0, right=max(100.0, ax.get_xlim()[1]))
    ax.set_ylim(bottom=0.0, top=max(100.0, ax.get_ylim()[1]))
    style_axes_frame(ax)
    fig.tight_layout()
    return save_figure(
        fig,
        figure_output_path(
            output_dir,
            "success_continuity_tradeoff.png",
            output_suffix,
        ),
    )


def normalized_metric_values(methods: Sequence[Dict], metric_key: str, higher_is_better: bool) -> np.ndarray:
    values = np.array([float(method.get(metric_key, 0.0)) for method in methods], dtype=float)
    if len(values) == 0:
        return values
    if metric_key.endswith("_rate") or metric_key in {
        "blocked_time_ratio",
        "jain_mec_load_fairness",
        "mec_load_fairness",
        "active_load_balance_score",
        "avg_load_balance_score",
    }:
        bounded = np.clip(values, 0.0, 1.0)
        return bounded if higher_is_better else 1.0 - bounded

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        return np.ones_like(values)
    normalized = (values - min_value) / (max_value - min_value)
    return normalized if higher_is_better else 1.0 - normalized


def plot_performance_radar(
    methods: Sequence[Dict],
    output_dir: Path,
    output_suffix: str = "",
) -> Optional[Path]:
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
    return save_figure(
        fig,
        figure_output_path(output_dir, "performance_radar.png", output_suffix),
    )


def plot_paper_dashboard(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
    window: int,
    output_suffix: str = "",
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
        metric_key="task_success_rate",
        title="Task Success Rate",
        ylabel="Task Success Rate (%)",
        compact=True,
    )
    add_panel_label(ax_delay, "(b)")

    ax_energy = fig.add_subplot(grid[1, 1])
    draw_metric_bar_panel(
        ax_energy,
        methods,
        metric_key="service_continuity_rate",
        title="Service Continuity Rate",
        ylabel="Service Continuity Rate (%)",
        compact=True,
    )
    add_panel_label(ax_energy, "(c)")

    ax_load_cdf = fig.add_subplot(grid[1, 2])
    draw_metric_bar_panel(
        ax_load_cdf,
        methods,
        metric_key="energy_per_successful_task",
        title="Energy per Successful Task",
        ylabel="Energy / Successful Task",
        compact=True,
    )
    add_panel_label(ax_load_cdf, "(d)")

    fig.suptitle("Publication-Style Baseline Comparison", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.92, wspace=0.26, hspace=0.32)
    return save_figure(
        fig,
        figure_output_path(
            output_dir,
            "paper_baseline_dashboard.png",
            output_suffix,
        ),
    )


def plot_training_curve_vs_baselines(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
    window: int,
    output_suffix: str = "",
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=220)
    has_curve = draw_reward_curve_panel(ax, history_path, methods, window=window, output_dir=output_dir)
    if not has_curve:
        plt.close(fig)
        return None
    fig.tight_layout()
    return save_figure(
        fig,
        figure_output_path(
            output_dir,
            "reward_curve_vs_baselines.png",
            output_suffix,
        ),
    )


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
    return save_figure(fig, output_dir / "reward_distribution.png")


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
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Number of evaluation episodes for each method.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override episode length for training/evaluation/baselines.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help=f"Total system training steps. Default is {DEFAULT_TOTAL_TIMESTEPS:,} steps.")
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
                        choices=["multi_objective"],
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
    parser.add_argument("--attn-mappo-timesteps", type=int, default=None,
                        help="Training steps for the Attn+MAPPO baseline. Defaults to --total-timesteps.")
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
    parser.add_argument("--reuse-learned-checkpoints", action="store_true",
                        help="Evaluate existing learned baseline checkpoints in --output-dir instead of retraining them.")
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

    requested_baselines = list(baselines)
    baselines = filter_duplicate_system_baselines(baselines, config_data)
    duplicate_system_baselines = [
        baseline for baseline in requested_baselines if baseline not in baselines
    ]

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
                exclude_methods=[*baselines, *duplicate_system_baselines],
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
                reuse_checkpoint_if_available=args.reuse_learned_checkpoints,
            )
            result.setdefault("source", "mappo_no_han_train_eval")
        elif baseline_name == "han_mappo":
            result = train_and_evaluate_han_mappo(
                config_data=config_data,
                objective=objective,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
                reuse_checkpoint_if_available=args.reuse_learned_checkpoints,
            )
            result.setdefault("source", "han_mappo_train_eval")
        elif baseline_name == "attn_mappo":
            result = train_and_evaluate_attention_mappo(
                config_data=config_data,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.attn_mappo_timesteps or args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
                reuse_checkpoint_if_available=args.reuse_learned_checkpoints,
            )
            result.setdefault("source", "attn_mappo_train_eval")
        elif baseline_name == "han_attn":
            result = train_and_evaluate_han_attn_mappo(
                config_data=config_data,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
                reuse_checkpoint_if_available=args.reuse_learned_checkpoints,
            )
            result.setdefault("source", "han_attn_train_eval")
        elif baseline_name == "han_maddpg":
            result = train_and_evaluate_han_maddpg_baseline(
                config_data=config_data,
                output_dir=output_dir,
                device=args.device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                total_timesteps=args.maddpg_timesteps or args.total_timesteps,
                early_stop_patience=args.early_stop_patience,
                pretrained_han_path=checkpoint,
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
                pretrained_han_path=checkpoint,
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
            "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "metric_schema_version": 2,
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
    paired_advantage_plot = plot_paired_advantage_over_baselines(methods, output_dir)
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
    if paired_advantage_plot is not None:
        print(f"Paired advantage figure: {paired_advantage_plot}")
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
