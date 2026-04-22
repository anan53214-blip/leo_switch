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
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

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

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.user import UserState

try:
    from scripts.train import (
        BEST_MODEL_METRIC_CHOICES,
        HANMAPPOTrainer,
        TrainConfig,
        compute_model_selection_score,
        energy_per_resolved_task,
    )
except ModuleNotFoundError:
    # Compatible with direct execution: python scripts/compare_system_baselines.py
    from train import (
        BEST_MODEL_METRIC_CHOICES,
        HANMAPPOTrainer,
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
    "stay",
    "max_elev",
    "max_rvt",
    "min_distance",
    "threshold_rvt",
    "joint_greedy",
]

DEFAULT_SYSTEM_RUN_DIR = PROJECT_ROOT / "results" / "full_train_latency_priority"
DEFAULT_SYSTEM_EXP_NAME = "han_mappo_latency_priority"
DEFAULT_TOTAL_TIMESTEPS = 1_200_000
DEFAULT_PLOT_WINDOW = 10
DEFAULT_SELECTION_METRIC = "latency_priority_score"

PRIMARY_COMPARE_METRICS = [
    ("avg_delay", "Average Delay"),
    ("service_continuity_rate", "Service Continuity"),
    ("service_availability_rate", "Service Availability"),
    ("task_completion_rate", "Task Completion"),
]

DISPLAY_NAME_MAP = {
    "random": "Random",
    "stay": "Stay",
    "max_elev": "Max-Elev",
    "max_rvt": "Max-RVT",
    "min_distance": "Min-Distance",
    "threshold_rvt": "Threshold-RVT Adaptive",
    "threshold_rvt_adaptive": "Threshold-RVT Adaptive",
    "joint_greedy": "Joint Greedy",
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
    "task_resolution_rate",
    "pending_task_rate",
    "avg_load_balance_score",
    "resolved_tasks",
    "pending_tasks",
    "total_tasks",
    "completed_tasks",
    "deadline_violations",
    "deadline_violation_rate",
]

HIGHER_IS_BETTER = {
    "mean_reward": True,
    "reward": True,
    "handover_success_rate": True,
    "service_continuity_rate": True,
    "service_availability_rate": True,
    "task_completion_rate": True,
    "task_resolution_rate": True,
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
    "color": "#D55E00",
    "linestyle": "-",
    "marker": "*",
    "linewidth": 2.6,
    "markersize": 8,
}

BASELINE_COLORS = [
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#7F7F7F",
]

BASELINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BASELINE_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]

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
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
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
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
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
        reward_delay_weight=float(config.get("reward_delay_weight", EnvConfig.reward_delay_weight)),
        reward_energy_weight=float(config.get("reward_energy_weight", EnvConfig.reward_energy_weight)),
        reward_handover_weight=float(config.get("reward_handover_weight", EnvConfig.reward_handover_weight)),
        reward_load_balance_weight=float(
            config.get("reward_load_balance_weight", EnvConfig.reward_load_balance_weight)
        ),
        reward_qos_weight=float(config.get("reward_qos_weight", EnvConfig.reward_qos_weight)),
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
        config["reward_delay_weight"] = 1.8
        config["reward_energy_weight"] = 0.15
        config["reward_handover_weight"] = 0.45
        config["reward_load_balance_weight"] = 0.25
        config["reward_qos_weight"] = 0.60
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


def compute_deadline_violation_rate(summary: Dict) -> float:
    resolved_tasks = float(summary.get("resolved_tasks", 0.0))
    return float(summary.get("deadline_violations", 0.0)) / max(resolved_tasks, 1.0)


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

        style = {
            "color": BASELINE_COLORS[baseline_index % len(BASELINE_COLORS)],
            "linestyle": BASELINE_LINESTYLES[baseline_index % len(BASELINE_LINESTYLES)],
            "marker": BASELINE_MARKERS[baseline_index % len(BASELINE_MARKERS)],
            "linewidth": 1.8,
            "markersize": 5.5,
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
        if self.strategy == "stay":
            return 0
        if self.strategy == "max_elev":
            target_idx = int(np.argmax([sat.elevation_deg for sat in visible_sats]))
        elif self.strategy == "max_rvt":
            target_idx = int(np.argmax([sat.rvt_seconds for sat in visible_sats]))
        elif self.strategy == "min_distance":
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


class ThresholdRVTPolicy(BasePolicy):
    def __init__(
        self,
        objective: str,
        rvt_threshold_sec: float = 60.0,
        queue_threshold: float = 0.5,
    ):
        self.objective = objective
        self.rvt_threshold_sec = float(rvt_threshold_sec)
        self.queue_threshold = float(queue_threshold)
        self.name = "threshold_rvt_adaptive"

    def _objective_name(self) -> str:
        if self.objective == "delay_only":
            return "delay"
        if self.objective == "energy_only":
            return "energy"
        return "weighted"

    def _select_offload_ratio(self, env: LEOSatelliteEnv, task, server, vis_info: object) -> float:
        if task is None or server is None or vis_info is None:
            return 0.0
        best_ratio, _ = env.offload_calc.find_optimal_offload_ratio(
            data_size_bits=task.data_size,
            computation_cycles=task.computation,
            max_delay=task.max_delay,
            distance_km=vis_info.distance_km,
            elevation_deg=vis_info.elevation_deg,
            satellite_freq_ghz=server.total_capacity_ghz,
            objective=self._objective_name(),
            num_samples=25,
        )
        queue_headroom = 1.0 - (
            server.queue_length / max(server.config.max_queue_size, 1)
        )
        return float(np.clip(best_ratio * max(queue_headroom, 0.25), 0.0, 1.0))

    def select_actions(self, env: LEOSatelliteEnv) -> np.ndarray:
        actions = np.zeros((env.num_users, 2), dtype=np.float32)
        for user_id, user in enumerate(env.user_manager.users):
            visible_sats = env._get_visible_satellites(user)
            current_vis = current_visibility(env, user)
            current_server = env.mec_manager.get_server(user.serving_satellite) if user.serving_satellite >= 0 else None

            should_stay = False
            if current_vis is not None and current_server is not None:
                queue_ratio = current_server.queue_length / max(current_server.config.max_queue_size, 1)
                should_stay = current_vis.rvt_seconds >= self.rvt_threshold_sec and queue_ratio <= self.queue_threshold

            selected_index = 0
            selected_vis = current_vis
            if not should_stay and visible_sats:
                scores = []
                for sat in visible_sats:
                    server = env.mec_manager.get_server(sat.sat_id)
                    queue_headroom = 1.0
                    if server is not None:
                        queue_headroom = 1.0 - (
                            server.queue_length / max(server.config.max_queue_size, 1)
                        )
                    snr_score = np.clip(
                        (env.channel.compute_snr_db(sat.distance_km, sat.elevation_deg) + 5.0) / 30.0,
                        0.0,
                        1.0,
                    )
                    score = (
                        0.45 * np.clip(sat.rvt_seconds / max(self.rvt_threshold_sec, 1.0), 0.0, 2.0)
                        + 0.20 * np.clip(sat.elevation_deg / 90.0, 0.0, 1.0)
                        + 0.20 * snr_score
                        + 0.15 * queue_headroom
                    )
                    scores.append(score)
                selected_index = int(np.argmax(scores)) + 1
                selected_vis = visible_sats[selected_index - 1]
                if selected_vis.sat_id == user.serving_satellite:
                    selected_index = 0

            actions[user_id, 0] = selected_index
            target_sat_id = selected_vis.sat_id if selected_vis is not None else user.serving_satellite
            target_server = env.mec_manager.get_server(target_sat_id) if target_sat_id >= 0 else None
            task = env.user_tasks.get(user_id)
            actions[user_id, 1] = self._select_offload_ratio(env, task, target_server, selected_vis)
        return actions


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


def build_policy(name: str, objective: str, fixed_offload: float, joint_offload_grid: Sequence[float]) -> BasePolicy:
    if name in {"random", "stay", "max_elev", "max_rvt", "min_distance"}:
        return SimpleHeuristicPolicy(name, fixed_offload)
    if name == "threshold_rvt":
        return ThresholdRVTPolicy(objective=objective)
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


def train_config_from_dict(
    config_data: Dict,
    device: str,
    max_steps: Optional[int],
    episodes: int,
    total_timesteps: Optional[int] = None,
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
        "task_resolution_rate": float(best_record.get("task_resolution_rate", 0.0)),
        "pending_task_rate": float(best_record.get("pending_task_rate", 0.0)),
        "avg_load_balance_score": float(best_record.get("avg_load_balance_score", 0.0)),
        "resolved_tasks": float(best_record.get("resolved_tasks", 0.0)),
        "pending_tasks": float(best_record.get("pending_tasks", 0.0)),
        "total_tasks": float(best_record.get("total_tasks", 0.0)),
        "completed_tasks": float(best_record.get("completed_tasks", 0.0)),
        "deadline_violations": float(best_record.get("deadline_violations", 0.0)),
        "deadline_violation_rate": (
            float(best_record.get("deadline_violations", 0.0)) / max(float(best_record.get("resolved_tasks", 0.0)), 1.0)
        ),
        "episode_metrics": [],
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
        "task_resolution_rate",
        "pending_task_rate",
        "deadline_violation_rate",
        "avg_load_balance_score",
        "energy_per_resolved_task",
        "selection_metric",
        "selection_score",
        "primary_metric_win_count",
        "primary_metric_wins_text",
        "selected_offload",
        "method_variant",
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
                "task_resolution_rate": float(record.get("task_resolution_rate", 0.0)),
                "pending_task_rate": float(record.get("pending_task_rate", 0.0)),
                "avg_load_balance_score": float(record.get("avg_load_balance_score", 0.0)),
                "deadline_violation_rate": float(record.get("deadline_violation_rate", 0.0)),
                "resolved_tasks": float(record.get("resolved_tasks", 0.0)),
                "completed_tasks": float(record.get("completed_tasks", 0.0)),
                "deadline_violations": float(record.get("deadline_violations", 0.0)),
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


def draw_metric_bar_panel(ax, methods: Sequence[Dict], metric_key: str, title: str, xlabel: str, compact: bool = False) -> None:
    ordered = order_methods(methods)
    styles = build_method_styles(ordered)
    labels = [method.get("display_name", method.get("method", "")) for method in ordered]
    values = [float(method.get(metric_key, 0.0)) * metric_scale(metric_key) for method in ordered]
    errors = []
    for method in ordered:
        samples = metric_episode_samples(method, metric_key)
        errors.append(float(np.std(samples)) if len(samples) > 1 else 0.0)

    positions = np.arange(len(ordered), dtype=float)
    colors = [styles[str(method.get("method", ""))]["color"] for method in ordered]
    bars = ax.barh(
        positions,
        values,
        xerr=errors if any(error > 0 for error in errors) else None,
        color=colors,
        edgecolor="#1F1F1F",
        linewidth=0.85,
        alpha=0.92,
        error_kw={"elinewidth": 1.1, "capsize": 2.8, "capthick": 1.1},
    )

    best_index = choose_best_index(values, HIGHER_IS_BETTER.get(metric_key, True))
    for index, (bar, value, method) in enumerate(zip(bars, values, ordered)):
        if method.get("is_system"):
            bar.set_hatch("///")
            bar.set_linewidth(1.2)
        if index == best_index:
            bar.set_edgecolor(PAPER_COLORS["dark"])
            bar.set_linewidth(1.8)
        ax.text(
            value + max(max(values) * 0.015, 0.02),
            bar.get_y() + bar.get_height() / 2.0,
            metric_display_value(value, metric_key),
            va="center",
            ha="left",
            fontsize=8.5 if compact else 9.0,
        )

    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    direction = "Higher is better" if HIGHER_IS_BETTER.get(metric_key, True) else "Lower is better"
    ax.set_title(f"{title} ({direction})")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.24)
    if values:
        ax.set_xlim(0.0, max(values) * 1.18 + 1e-9)
    if compact:
        ax.tick_params(axis="y", labelsize=9)


def draw_reward_curve_panel(ax, history_path: Optional[Path], methods: Sequence[Dict], window: int, compact: bool = False) -> bool:
    if history_path is None or not history_path.exists():
        return False

    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training = payload.get("training", [])
    evaluation = payload.get("evaluation", [])
    if not training:
        return False

    steps = np.array([record.get("total_steps", 0) for record in training], dtype=float)
    rewards = np.array([record.get("recent_mean_reward", 0.0) for record in training], dtype=float)

    mean_reward, lower_reward, upper_reward = compute_confidence_band(rewards, window=max(window, 3))
    ax.plot(steps, rewards, color=PAPER_COLORS["primary"], alpha=0.16, linewidth=0.8)
    ax.fill_between(steps, lower_reward, upper_reward, color=PAPER_COLORS["primary"], alpha=PAPER_COLORS["fill_alpha"])
    ax.plot(steps, mean_reward, color=PAPER_COLORS["primary"], linewidth=2.4, label="HAN+MAPPO training")

    if evaluation:
        eval_steps = np.array([record.get("total_steps", 0) for record in evaluation], dtype=float)
        eval_rewards = np.array([record.get("eval_mean_reward", 0.0) for record in evaluation], dtype=float)
        eval_stds = np.array([record.get("eval_std_reward", 0.0) for record in evaluation], dtype=float)
        ax.fill_between(
            eval_steps,
            eval_rewards - eval_stds,
            eval_rewards + eval_stds,
            color=PAPER_COLORS["success"],
            alpha=0.12,
        )
        ax.plot(
            eval_steps,
            eval_rewards,
            color=PAPER_COLORS["success"],
            marker="o",
            markersize=4.0,
            linewidth=1.9,
            label="HAN+MAPPO eval",
        )

    baseline_methods = [method for method in order_methods(methods) if not method.get("is_system")]
    baseline_styles = build_method_styles(baseline_methods)
    for method in baseline_methods:
        style = baseline_styles[str(method.get("method", ""))]
        mean_value = float(method.get("mean_reward", 0.0))
        std_value = float(method.get("std_reward", 0.0))
        if std_value > 0.0:
            ax.axhspan(
                mean_value - std_value,
                mean_value + std_value,
                color=style["color"],
                alpha=0.06,
                linewidth=0,
            )
        ax.axhline(
            y=mean_value,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            alpha=0.92,
            label=method.get("display_name", method.get("method", "")),
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Convergence Against Baseline Levels")
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(
        loc="lower right" if compact else "best",
        fontsize=8.2 if compact else 9.2,
        ncol=1 if compact else 2,
    )
    ax.grid(True, linestyle="--", alpha=0.24)
    return True


def plot_method_comparison(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    if not methods:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    for axis, (metric_key, title, xlabel) in zip(axes.flatten(), PAPER_METRIC_PLOTS):
        draw_metric_bar_panel(axis, methods, metric_key=metric_key, title=title, xlabel=xlabel)

    fig.suptitle("HAN+MAPPO vs Baselines (Latency-Priority Metrics)", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir / "method_comparison.png")


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
    ax.grid(True, linestyle="--", alpha=0.24)
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
        axis.grid(True, linestyle="--", alpha=0.24)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir / "additional_metrics_episode_comparison.png")


def plot_delay_energy_tradeoff(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    if not methods:
        return None

    ordered = order_methods(methods)
    styles = build_method_styles(ordered)
    fig, ax = plt.subplots(figsize=(9, 6.8), dpi=220)
    delay_values = [float(method.get("avg_delay", 0.0)) for method in ordered]
    energy_values = [float(method.get("total_energy", 0.0)) for method in ordered]
    delay_span = max(max(delay_values) - min(delay_values), 1e-6)
    energy_span = max(max(energy_values) - min(energy_values), 1e-6)

    overlap_groups: Dict[tuple[float, float], List[Dict]] = defaultdict(list)
    for method in ordered:
        overlap_key = (
            round(float(method.get("avg_delay", 0.0)), 4),
            round(float(method.get("total_energy", 0.0)), 2),
        )
        overlap_groups[overlap_key].append(method)

    for method in ordered:
        style = styles[str(method.get("method", ""))]
        delay_value = float(method.get("avg_delay", 0.0))
        energy_value = float(method.get("total_energy", 0.0))
        completion_rate = float(method.get("task_completion_rate", 0.0))
        overlap_key = (round(delay_value, 4), round(energy_value, 2))
        siblings = overlap_groups[overlap_key]
        sibling_index = siblings.index(method)
        sibling_offset = sibling_index - (len(siblings) - 1) / 2.0
        plot_delay = delay_value + sibling_offset * delay_span * 0.012
        plot_energy = energy_value + sibling_offset * energy_span * 0.018
        marker = "*" if method.get("is_system") else style["marker"]
        size = 180 + 820 * max(completion_rate, 0.0)
        ax.scatter(
            plot_delay,
            plot_energy,
            s=size,
            color=style["color"],
            marker=marker,
            alpha=0.88,
            edgecolors=PAPER_COLORS["dark"],
            linewidths=1.0,
            label=method.get("display_name", method.get("method", "")),
        )
        ax.annotate(
            method.get("display_name", method.get("method", "")),
            xy=(plot_delay, plot_energy),
            xytext=(7, 7 if sibling_offset >= 0 else -11),
            textcoords="offset points",
            fontsize=8.2,
            color=PAPER_COLORS["dark"],
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "alpha": 0.76,
            },
        )

    ax.set_xlabel("Average Delay (s)")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Delay-Energy Trade-off (Bubble Size = Task Completion)")
    ax.grid(True, linestyle="--", alpha=0.24)
    ax.text(
        0.99,
        0.02,
        "Better region: lower-left",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=PAPER_COLORS["muted"],
    )
    fig.tight_layout()
    return save_figure(fig, output_dir / "delay_energy_tradeoff.png")


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
    has_reward_curve = draw_reward_curve_panel(ax_reward, history_path, methods, window=window, compact=True)
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
        xlabel="Avg Delay (s)",
        compact=True,
    )
    add_panel_label(ax_delay, "(b)")

    ax_task = fig.add_subplot(grid[1, 1])
    draw_metric_bar_panel(
        ax_task,
        methods,
        metric_key="service_continuity_rate",
        title="Service Continuity",
        xlabel="Rate (%)",
        compact=True,
    )
    add_panel_label(ax_task, "(c)")

    ax_energy = fig.add_subplot(grid[1, 2])
    draw_metric_bar_panel(
        ax_energy,
        methods,
        metric_key="service_availability_rate",
        title="Service Availability",
        xlabel="Rate (%)",
        compact=True,
    )
    add_panel_label(ax_energy, "(d)")

    fig.suptitle("Publication-Style Baseline Comparison (Latency-Priority View)", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.07, top=0.92, wspace=0.26, hspace=0.32)
    return save_figure(fig, output_dir / "paper_baseline_dashboard.png")


def plot_training_curve_vs_baselines(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
    window: int,
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=220)
    has_curve = draw_reward_curve_panel(ax, history_path, methods, window=window)
    if not has_curve:
        plt.close(fig)
        return None
    fig.tight_layout()
    return save_figure(fig, output_dir / "reward_curve_vs_baselines.png")


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
    ax.grid(axis="x", linestyle="--", alpha=0.24)
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
                        help="Directory used for training outputs and/or existing system artifacts.")
    parser.add_argument("--system-checkpoint", type=str, default=None,
                        help="Path to best_model.pt or final_model.pt.")
    parser.add_argument("--resume-system", action="store_true",
                        help="Resume training from an existing checkpoint in --system-run-dir or --system-checkpoint.")
    parser.add_argument("--exp-name", type=str, default=DEFAULT_SYSTEM_EXP_NAME,
                        help="Experiment name used when training from this unified entry script.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes for each method.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override episode length for training/evaluation/baselines.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Total system training steps. Default is 1,200,000 steps.")
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
    parser.add_argument("--skip-system-eval", action="store_true",
                        help="Skip checkpoint evaluation and only use history summary when available.")
    parser.add_argument("--plot-window", type=int, default=DEFAULT_PLOT_WINDOW,
                        help="Smoothing window used by the publication-style reward figure.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for JSON/CSV summaries and figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_publication_style()
    baselines = DEFAULT_BASELINES if "all" in args.baselines else args.baselines
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
        run_dir, checkpoint, history_path, config_data = run_system_training(
            config_data=config_data,
            objective=objective,
            system_run_dir=Path(args.system_run_dir).resolve(),
            device=args.device,
            episodes=args.episodes,
            max_steps=args.max_steps,
            total_timesteps=args.total_timesteps,
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
    elif history_path and history_path.exists():
        _, system_method = extract_history_method(history_path)

    methods: List[Dict] = []
    if system_method is not None:
        methods.append(system_method)

    for baseline_name in baselines:
        if baseline_name in {"random", "stay", "max_elev", "max_rvt", "min_distance"}:
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
    reward_distribution_plot = plot_reward_distribution(methods, output_dir)
    tradeoff_plot = plot_delay_energy_tradeoff(methods, output_dir)
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
    if reward_distribution_plot is not None:
        print(f"Reward distribution figure: {reward_distribution_plot}")
    if tradeoff_plot is not None:
        print(f"Delay-energy trade-off figure: {tradeoff_plot}")
    if dashboard_plot is not None:
        print(f"Paper dashboard figure: {dashboard_plot}")


if __name__ == "__main__":
    main()
