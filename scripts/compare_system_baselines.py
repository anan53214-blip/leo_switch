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
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.user import UserState
from scripts.train import HANMAPPOTrainer, TrainConfig
from scripts.train_delay_only import DelayOnlyEnv, DelayOnlyTrainer
from scripts.train_energy_only import EnergyOnlyEnv, EnergyOnlyTrainer


DEFAULT_BASELINES = [
    "random",
    "stay",
    "max_elev",
    "max_rvt",
    "min_distance",
    "threshold_rvt",
    "joint_greedy",
]

HIGHER_IS_BETTER = {
    "mean_reward": True,
    "handover_success_rate": True,
    "service_continuity_rate": True,
    "task_completion_rate": True,
    "task_resolution_rate": True,
    "avg_load_balance_score": True,
    "avg_delay": False,
    "total_energy": False,
    "pending_task_rate": False,
}

PLOT_METRICS = [
    ("mean_reward", "Mean Reward"),
    ("avg_delay", "Avg Delay (s)"),
    ("total_energy", "Total Energy (J)"),
    ("handover_success_rate", "Handover Success Rate"),
    ("service_continuity_rate", "Service Continuity Rate"),
    ("task_completion_rate", "Task Completion Rate"),
    ("pending_task_rate", "Pending Task Rate"),
    ("avg_load_balance_score", "Load Balance Score"),
]


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


def build_default_train_config(objective: str, seed: int, max_steps: int, num_users: int) -> Dict:
    config = asdict(TrainConfig())
    config["seed"] = seed
    config["max_steps"] = max_steps
    config["num_users"] = num_users
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
        config["exp_name"] = "han_mappo_leo"
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


def load_run_config(checkpoint: Optional[Path], history_path: Optional[Path]) -> Dict:
    if checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        return dict(payload.get("config", {}))
    if history_path and history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload.get("config", {}))
    raise FileNotFoundError("No checkpoint or training_history.json could be loaded.")


def summarize_results(
    name: str,
    rewards: Sequence[float],
    summaries: Sequence[Dict],
    extra: Optional[Dict] = None,
    is_system: bool = False,
) -> Dict:
    result = {
        "method": name,
        "episodes": len(rewards),
        "is_system": bool(is_system),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
    }
    metric_keys = {
        "avg_delay",
        "total_energy",
        "handover_success_rate",
        "service_continuity_rate",
        "task_completion_rate",
        "task_resolution_rate",
        "pending_task_rate",
        "avg_load_balance_score",
        "resolved_tasks",
        "pending_tasks",
        "total_tasks",
        "completed_tasks",
        "deadline_violations",
    }
    for key in metric_keys:
        values = [float(summary.get(key, 0.0)) for summary in summaries]
        result[key] = float(np.mean(values)) if values else 0.0
    if extra:
        result.update(extra)
    return result


def style_for_method(method: Dict) -> str:
    return "#D55E00" if method.get("is_system") else "#4C72B0"


def order_methods(methods: Sequence[Dict]) -> List[Dict]:
    systems = [method for method in methods if method.get("is_system")]
    baselines = [method for method in methods if not method.get("is_system")]
    baselines.sort(key=lambda item: item.get("mean_reward", 0.0), reverse=True)
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
        [candidate["mean_reward"] for candidate in candidates],
        higher_is_better=True,
    )
    best = dict(candidates[best_index])
    best["method"] = strategy
    best["method_variant"] = best["method"] + f"(best_offload={best.get('selected_offload', 0.0):.2f})"
    return best


def trainer_class_for_objective(objective: str):
    if objective == "delay_only":
        return DelayOnlyTrainer
    if objective == "energy_only":
        return EnergyOnlyTrainer
    return HANMAPPOTrainer


def train_config_from_dict(config_data: Dict, device: str, max_steps: Optional[int], episodes: int) -> TrainConfig:
    config = TrainConfig()
    for key, value in config_data.items():
        setattr(config, key, value)
    config.device = device
    config.eval_episodes = episodes
    if max_steps is not None:
        config.max_steps = int(max_steps)
    return config


def evaluate_system_checkpoint(
    checkpoint: Path,
    config_data: Dict,
    objective: str,
    episodes: int,
    device: str,
    max_steps: Optional[int],
) -> Dict:
    trainer_cls = trainer_class_for_objective(objective)
    config = train_config_from_dict(config_data, device=device, max_steps=max_steps, episodes=episodes)
    trainer = trainer_cls(config)
    trainer.load_checkpoint(str(checkpoint))

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

    best_record = max(evaluation_records, key=lambda record: record.get("eval_mean_reward", -float("inf")))
    result = {
        "method": str(config_data.get("exp_name", history_path.parent.name or "system")),
        "episodes": int(config_data.get("eval_episodes", 0)),
        "is_system": True,
        "mean_reward": float(best_record.get("eval_mean_reward", 0.0)),
        "std_reward": float(best_record.get("eval_std_reward", 0.0)),
        "avg_delay": float(best_record.get("avg_delay", 0.0)),
        "total_energy": float(best_record.get("total_energy", 0.0)),
        "handover_success_rate": float(best_record.get("handover_success_rate", 0.0)),
        "service_continuity_rate": float(best_record.get("service_continuity_rate", 0.0)),
        "task_completion_rate": float(best_record.get("task_completion_rate", 0.0)),
        "task_resolution_rate": float(best_record.get("task_resolution_rate", 0.0)),
        "pending_task_rate": float(best_record.get("pending_task_rate", 0.0)),
        "avg_load_balance_score": float(best_record.get("avg_load_balance_score", 0.0)),
        "resolved_tasks": float(best_record.get("resolved_tasks", 0.0)),
        "pending_tasks": float(best_record.get("pending_tasks", 0.0)),
        "total_tasks": float(best_record.get("total_tasks", 0.0)),
        "completed_tasks": float(best_record.get("completed_tasks", 0.0)),
        "deadline_violations": float(best_record.get("deadline_violations", 0.0)),
        "source": "training_history_best_eval",
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
        "is_system",
        "episodes",
        "mean_reward",
        "std_reward",
        "avg_delay",
        "total_energy",
        "handover_success_rate",
        "service_continuity_rate",
        "task_completion_rate",
        "task_resolution_rate",
        "pending_task_rate",
        "avg_load_balance_score",
        "selected_offload",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {key: method.get(key, "") for key in fieldnames}
            writer.writerow(row)
    return path


def annotate_bars(ax, bars, values: Sequence[float], percent: bool = False) -> None:
    for bar, value in zip(bars, values):
        label = f"{100.0 * value:.1f}%" if percent else f"{value:.3f}"
        if abs(value) >= 1000 and not percent:
            label = f"{value:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_method_comparison(methods: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    if not methods:
        return None

    ordered = order_methods(methods)
    labels = [method["method"] for method in ordered]
    colors = [style_for_method(method) for method in ordered]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=180)
    axes = axes.flatten()

    for axis, (key, title) in zip(axes, PLOT_METRICS):
        values = [float(method.get(key, 0.0)) for method in ordered]
        bars = axis.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.tick_params(axis="x", rotation=25)
        annotate_bars(axis, bars, values, percent=key.endswith("_rate") or "score" in key)

        if not HIGHER_IS_BETTER.get(key, True):
            axis.set_ylabel("Lower is better")
        else:
            axis.set_ylabel("Higher is better")

    fig.tight_layout(pad=1.4)
    path = output_dir / "method_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curve_vs_baselines(
    history_path: Optional[Path],
    methods: Sequence[Dict],
    output_dir: Path,
) -> Optional[Path]:
    if history_path is None or not history_path.exists():
        return None

    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training = payload.get("training", [])
    evaluation = payload.get("evaluation", [])
    if not training:
        return None

    steps = np.array([record.get("total_steps", 0) for record in training], dtype=float)
    rewards = np.array([record.get("recent_mean_reward", 0.0) for record in training], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.plot(steps, rewards, color="#D55E00", linewidth=2.0, label="System Training Reward")

    if evaluation:
        eval_steps = np.array([record.get("total_steps", 0) for record in evaluation], dtype=float)
        eval_rewards = np.array([record.get("eval_mean_reward", 0.0) for record in evaluation], dtype=float)
        ax.plot(
            eval_steps,
            eval_rewards,
            color="#CC79A7",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="System Eval Reward",
        )

    for method in methods:
        if method.get("is_system"):
            continue
        ax.axhline(
            y=float(method.get("mean_reward", 0.0)),
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"{method['method']} mean reward",
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("System Reward Curve vs Baseline Reward Levels")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = output_dir / "reward_curve_vs_baselines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the current system method against heuristic baselines.")
    parser.add_argument("--system-run-dir", type=str, default=str(PROJECT_ROOT / "results" / "full_train_delay_focus"),
                        help="Directory containing training_history.json and/or best_model.pt.")
    parser.add_argument("--system-checkpoint", type=str, default=None,
                        help="Path to best_model.pt or final_model.pt.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes for each method.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override episode length for system evaluation and baselines.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for baseline evaluation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device used when evaluating a trained checkpoint.")
    parser.add_argument("--objective", type=str, default="multi_objective",
                        choices=["multi_objective", "delay_only", "energy_only"],
                        help="Objective used when no system run is provided.")
    parser.add_argument("--num-users", type=int, default=5,
                        help="User count used when no system run is provided.")
    parser.add_argument("--baselines", type=str, nargs="+", default=["all"],
                        help="Baselines to evaluate. Use 'all' for the default suite.")
    parser.add_argument("--fixed-offload-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="Offload candidates for simple heuristic baselines.")
    parser.add_argument("--joint-offload-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Offload grid used by the joint greedy baseline.")
    parser.add_argument("--skip-system-eval", action="store_true",
                        help="Skip checkpoint evaluation and only use history summary when available.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for JSON/CSV summaries and figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            max_steps=args.max_steps or 200,
            num_users=args.num_users,
        )

    if checkpoint and not args.skip_system_eval:
        system_method = evaluate_system_checkpoint(
            checkpoint=checkpoint,
            config_data=config_data,
            objective=objective,
            episodes=args.episodes,
            device=args.device,
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

    methods = order_methods(methods)
    json_path = save_results_json(
        output_dir,
        {
            "generated_at": timestamp,
            "objective": objective,
            "system_run_dir": str(run_dir) if run_dir else None,
            "system_checkpoint": str(checkpoint) if checkpoint else None,
            "training_history": str(history_path) if history_path else None,
            "env_config": asdict(build_env_config_from_train_config(config_data, seed=args.seed, max_steps=args.max_steps)),
            "methods": methods,
        },
    )
    csv_path = save_results_csv(output_dir, methods)
    metrics_plot = plot_method_comparison(methods, output_dir)
    reward_curve_plot = plot_training_curve_vs_baselines(history_path, methods, output_dir)

    print(json.dumps(methods, ensure_ascii=False, indent=2))
    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")
    if metrics_plot is not None:
        print(f"Metric comparison figure: {metrics_plot}")
    if reward_curve_plot is not None:
        print(f"Reward curve comparison figure: {reward_curve_plot}")


if __name__ == "__main__":
    main()
