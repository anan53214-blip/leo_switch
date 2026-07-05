"""Fast satellite load/status features for candidate-attention policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.environment.gym_env import LEOSatelliteEnv


SATELLITE_LOAD_FEATURE_DIM = 8
SATELLITE_CONTEXT_FEATURE_DIM = 13
SHARED_CONSTRAINT_DIM = 6
SATELLITE_RISK_FEATURE_SLICE = slice(8, 13)


def build_satellite_load_features(env: "LEOSatelliteEnv", num_agents: int) -> np.ndarray:
    """Build per-satellite load, visibility, link, and demand features."""
    features = np.zeros(
        (env.num_satellites, SATELLITE_LOAD_FEATURE_DIM),
        dtype=np.float32,
    )
    visible_count = np.zeros(env.num_satellites, dtype=np.float32)
    snr_sum = np.zeros(env.num_satellites, dtype=np.float32)
    rvt_sum = np.zeros(env.num_satellites, dtype=np.float32)
    demand_sum = np.zeros(env.num_satellites, dtype=np.float32)

    for user in env.user_manager.users:
        task = env.user_tasks.get(user.user_id)
        task_demand = 0.0
        if task is not None:
            data_score = float(task.data_size) / 1e8
            compute_score = float(task.computation) / 1e10
            task_demand = 0.5 * data_score + 0.5 * compute_score
        for vis in env._get_visible_satellites(user):
            sid = int(vis.sat_id)
            visible_count[sid] += 1.0
            snr_sum[sid] += env.channel.compute_snr_db(
                vis.distance_km,
                vis.elevation_deg,
            )
            rvt_sum[sid] += float(vis.rvt_seconds)
            demand_sum[sid] += task_demand

    for sid in range(env.num_satellites):
        server = env.mec_manager.get_server(sid)
        utilization = 0.0
        queue_ratio = 0.0
        connected_ratio = 0.0
        if server is not None:
            utilization = float(server.utilization)
            queue_ratio = (
                float(server.queue_length) /
                max(float(server.config.max_queue_size), 1.0)
            )
            connected_ratio = (
                float(len(server.connected_users)) /
                max(float(num_agents), 1.0)
            )
        count = max(float(visible_count[sid]), 1.0)
        features[sid] = [
            sid / max(float(env.num_satellites), 1.0),
            np.clip(utilization, 0.0, 1.0),
            np.clip(queue_ratio, 0.0, 1.0),
            np.clip(connected_ratio, 0.0, 1.0),
            np.clip(visible_count[sid] / max(float(num_agents), 1.0), 0.0, 1.0),
            np.clip(demand_sum[sid] / count, 0.0, 1.0),
            np.clip((snr_sum[sid] / count) / 50.0, 0.0, 1.0),
            np.clip((rvt_sum[sid] / count) / 600.0, 0.0, 1.0),
        ]
    return features


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _queue_ratio(server) -> float:
    if server is None:
        return 0.0
    max_queue_size = max(float(getattr(server.config, "max_queue_size", 1.0)), 1.0)
    return _clip01(float(getattr(server, "queue_length", 0.0)) / max_queue_size)


def _task_deadline_pressure(env: "LEOSatelliteEnv", task) -> float:
    if task is None:
        return 0.0
    max_delay = max(float(getattr(task, "max_delay", 0.0)), 1e-6)
    elapsed = max(float(getattr(env, "current_time", 0.0)) - float(getattr(task, "creation_time", 0.0)), 0.0)
    remaining = max(max_delay - elapsed, 0.0)
    return 1.0 - _clip01(remaining / max_delay)


def _rvt_risk(env: "LEOSatelliteEnv", rvt_seconds: float) -> float:
    threshold = max(float(getattr(env.config, "pre_handover_rvt_sec", 1e-6)), 1e-6)
    return 1.0 - _clip01(float(rvt_seconds) / threshold)


def build_satellite_context_features(env: "LEOSatelliteEnv", num_agents: int) -> np.ndarray:
    """Build CPQ per-satellite context features.

    The first eight columns are exactly the legacy satellite load features.
    The last five columns add queue wait, compute headroom, deadline pressure,
    future visibility risk, and queue-full risk.
    """
    base_features = build_satellite_load_features(env, num_agents)
    features = np.zeros(
        (env.num_satellites, SATELLITE_CONTEXT_FEATURE_DIM),
        dtype=np.float32,
    )
    features[:, :SATELLITE_LOAD_FEATURE_DIM] = base_features

    deadline_pressure_sum = np.zeros(env.num_satellites, dtype=np.float32)
    rvt_risk_sum = np.zeros(env.num_satellites, dtype=np.float32)
    visible_count = np.zeros(env.num_satellites, dtype=np.float32)

    for user in env.user_manager.users:
        task = env.user_tasks.get(user.user_id)
        pressure = _task_deadline_pressure(env, task)
        for vis in env._get_visible_satellites(user):
            sid = int(vis.sat_id)
            visible_count[sid] += 1.0
            deadline_pressure_sum[sid] += pressure
            rvt_risk_sum[sid] += _rvt_risk(env, float(vis.rvt_seconds))

    for sid in range(env.num_satellites):
        server = env.mec_manager.get_server(sid)
        queue_ratio = _queue_ratio(server)
        queue_wait_ratio = 0.0
        compute_headroom = 1.0
        queue_full_risk = _clip01(queue_ratio / 0.85) if queue_ratio > 0.0 else 0.0
        if server is not None:
            queue_wait_ratio = _clip01(float(server.get_estimated_wait_time()) / 10.0)
            total_capacity = max(float(getattr(server, "total_capacity_ghz", 0.0)), 1e-6)
            compute_headroom = _clip01(float(getattr(server, "available_freq_ghz", 0.0)) / total_capacity)
            if bool(getattr(server, "is_full", False)):
                queue_full_risk = 1.0
        count = max(float(visible_count[sid]), 1.0)
        features[sid, 8:13] = [
            queue_wait_ratio,
            compute_headroom,
            _clip01(deadline_pressure_sum[sid] / count),
            _clip01(rvt_risk_sum[sid] / count),
            queue_full_risk,
        ]
    return features


def build_shared_constraint_vector(env: "LEOSatelliteEnv", num_agents: int) -> np.ndarray:
    """Build global CPQ congestion/deadline/visibility constraints."""
    queue_ratios = []
    overloaded = 0
    wait_ratios = []

    for sid in range(env.num_satellites):
        server = env.mec_manager.get_server(sid)
        ratio = _queue_ratio(server)
        queue_ratios.append(ratio)
        utilization = float(getattr(server, "utilization", 0.0)) if server is not None else 0.0
        is_full = bool(getattr(server, "is_full", False)) if server is not None else False
        if is_full or ratio >= 0.85 or utilization >= 0.85:
            overloaded += 1

    task_pressures = []
    min_wait_per_user = []
    handover_risk_users = 0
    rvt_threshold = max(float(getattr(env.config, "pre_handover_rvt_sec", 1e-6)), 1e-6)

    for user in env.user_manager.users:
        task_pressures.append(_task_deadline_pressure(env, env.user_tasks.get(user.user_id)))

        visible_waits = []
        for vis in env._get_visible_satellites(user):
            server = env.mec_manager.get_server(int(vis.sat_id))
            if server is not None:
                visible_waits.append(_clip01(float(server.get_estimated_wait_time()) / 10.0))
        if visible_waits:
            min_wait_per_user.append(min(visible_waits))

        serving_satellite = int(getattr(user, "serving_satellite", -1))
        serving_visibility = None
        if serving_satellite >= 0 and hasattr(env, "_get_satellite_visibility"):
            serving_visibility = env._get_satellite_visibility(user, serving_satellite)
        if (
            serving_satellite < 0
            or serving_visibility is None
            or not bool(getattr(serving_visibility, "is_visible", False))
            or float(getattr(serving_visibility, "rvt_seconds", 0.0)) < rvt_threshold
        ):
            handover_risk_users += 1

    queue_array = np.asarray(queue_ratios, dtype=np.float32)
    return np.asarray(
        [
            _clip01(float(queue_array.mean()) if queue_array.size else 0.0),
            _clip01(float(queue_array.max()) if queue_array.size else 0.0),
            _clip01(float(overloaded) / max(float(env.num_satellites), 1.0)),
            _clip01(float(np.mean(min_wait_per_user)) if min_wait_per_user else 0.0),
            _clip01(float(np.mean(task_pressures)) if task_pressures else 0.0),
            _clip01(float(handover_risk_users) / max(float(num_agents), 1.0)),
        ],
        dtype=np.float32,
    )
